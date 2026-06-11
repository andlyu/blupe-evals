"""Blupe relay — outbound-only transport between operator and robot nodes + fleet UI.

One stdlib file, three roles (PLAN "Customer transport"):

  serve     the hosted rendezvous: robots register, operators request channels, the relay
            authenticates (per-robot token), splices bytes, and serves a FLEET UI (web)
            that lists arms/operators and can turn an arm ON (with preflight verification)
            or OFF (kill serve + guaranteed torque-off).
  robot     runs at the robot site; dials OUT to the relay, bridges relay channels to local
            services on an allowlist, and executes fleet commands (preflight/arm_on/arm_off).
  operator  runs at the operator site; dials OUT and exposes the robot's services on LOCAL
            ports (5599 -> 15599, 8089 -> 18089), so the eval runs unchanged.

Both ends only ever make OUTBOUND connections — no VPN, no port-forwards at either site.
Robot-side safety (clamp, hold-on-drop, watchdog) rides underneath, untouched.

Wire (newline-JSON hello, then either a held control channel or a raw spliced data pipe):
  robot ctrl   -> {"role": "robot", "robot": ID, "token": T}
  operator     -> {"role": "operator", "robot": ID, "token": T, "port": 5599}
  robot data   -> {"role": "data", "conn": C}
  relay->robot:   {"open": PORT, "conn": C} | {"cmd": NAME, "req": N}
  robot->relay:   {"ping": t} | {"resp": N, "data": {...}}

Fleet UI: http://<relay>:8080/?token=<RELAY_ADMIN_TOKEN>   (set both env vars on the host:
RELAY_TOKENS="yam-1:tok,..." RELAY_ADMIN_TOKEN="...").

Run:
  relay:    RELAY_TOKENS="yam-1:tok" RELAY_ADMIN_TOKEN="atok" \
                python3 relay.py serve --port 8443 --ui-port 8080
  robot:    python3 relay.py robot --relay HOST:8443 --robot yam-1 --token tok
  operator: python3 relay.py operator --relay HOST:8443 --robot yam-1 --token tok
"""

import argparse
import asyncio
import itertools
import json
import os
import shlex
import time
import urllib.parse

PING_S = 15
OPEN_TIMEOUT_S = 10
CMD_TIMEOUT_S = 45


async def _read_hello(reader):
    line = await asyncio.wait_for(reader.readline(), timeout=10)
    return json.loads(line.decode())


def _send(writer, obj):
    writer.write((json.dumps(obj) + "\n").encode())


async def _splice(a_reader, a_writer, b_reader, b_writer, tag, on_close=None):
    async def pipe(r, w):
        try:
            while True:
                data = await r.read(65536)
                if not data:
                    break
                w.write(data)
                await w.drain()
        except (ConnectionError, asyncio.IncompleteReadError, OSError):
            pass

    await asyncio.gather(pipe(a_reader, b_writer), pipe(b_reader, a_writer))
    for w in (a_writer, b_writer):
        try:
            w.close()
        except OSError:
            pass
    if on_close:
        on_close()
    print(f"[relay] {tag} closed", flush=True)


# --------------------------------------------------------------------------- #
# role: serve (rendezvous + fleet UI)
# --------------------------------------------------------------------------- #
class Robot:
    def __init__(self, writer):
        self.writer = writer
        self.since = time.time()
        self.channels = {}            # conn_id -> {"port": p, "peer": str, "since": t}
        self.last_preflight = None    # {"t": ..., "data": {...}}


class Relay:
    def __init__(self, tokens, admin_token):
        self.tokens = tokens
        self.admin_token = admin_token
        self.robots = {}              # robot_id -> Robot
        self.pending = {}             # conn_id -> Future[(reader, writer)]
        self.cmd_waiters = {}         # req_id -> Future[dict]
        self.ids = itertools.count(1)

    def auth(self, robot, token):
        return self.tokens.get(robot) is not None and self.tokens[robot] == token

    # ---- transport ---------------------------------------------------------- #
    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            hello = await _read_hello(reader)
        except Exception:
            writer.close()
            return
        role = hello.get("role")

        if role == "robot":
            robot_id, token = hello.get("robot"), hello.get("token")
            if not self.auth(robot_id, token):
                _send(writer, {"err": "auth"})
                writer.close()
                return
            old = self.robots.get(robot_id)
            if old is not None:
                old.writer.close()
            rob = self.robots[robot_id] = Robot(writer)
            print(f"[relay] robot '{robot_id}' online from {peer}", flush=True)
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        msg = json.loads(line.decode())
                    except ValueError:
                        continue
                    req = msg.get("resp")
                    if req is not None:
                        fut = self.cmd_waiters.pop(req, None)
                        if fut is not None and not fut.done():
                            fut.set_result(msg.get("data", {}))
            finally:
                if self.robots.get(robot_id) is rob:
                    del self.robots[robot_id]
                print(f"[relay] robot '{robot_id}' offline", flush=True)

        elif role == "data":
            fut = self.pending.pop(hello.get("conn"), None)
            if fut is None or fut.done():
                writer.close()
                return
            fut.set_result((reader, writer))

        elif role == "operator":
            robot_id, token, port = hello.get("robot"), hello.get("token"), hello.get("port")
            if not self.auth(robot_id, token):
                _send(writer, {"err": "auth"})
                writer.close()
                return
            rob = self.robots.get(robot_id)
            if rob is None:
                _send(writer, {"err": f"robot '{robot_id}' offline"})
                writer.close()
                return
            conn_id = next(self.ids)
            fut = asyncio.get_running_loop().create_future()
            self.pending[conn_id] = fut
            _send(rob.writer, {"open": port, "conn": conn_id})
            try:
                await rob.writer.drain()
                r2, w2 = await asyncio.wait_for(fut, timeout=OPEN_TIMEOUT_S)
            except Exception as e:
                self.pending.pop(conn_id, None)
                _send(writer, {"err": f"robot did not answer: {e}"})
                writer.close()
                return
            _send(writer, {"ok": True})
            await writer.drain()
            rob.channels[conn_id] = {"port": port, "peer": str(peer), "since": time.time()}
            await _splice(reader, writer, r2, w2, f"{robot_id}:{port}#{conn_id}",
                          on_close=lambda: rob.channels.pop(conn_id, None))
        else:
            writer.close()

    # ---- fleet commands ------------------------------------------------------ #
    async def command(self, robot_id, cmd):
        rob = self.robots.get(robot_id)
        if rob is None:
            return {"ok": False, "err": f"robot '{robot_id}' offline"}
        req = next(self.ids)
        fut = asyncio.get_running_loop().create_future()
        self.cmd_waiters[req] = fut
        _send(rob.writer, {"cmd": cmd, "req": req})
        try:
            await rob.writer.drain()
            data = await asyncio.wait_for(fut, timeout=CMD_TIMEOUT_S)
        except Exception as e:
            self.cmd_waiters.pop(req, None)
            return {"ok": False, "err": f"no answer: {e}"}
        if cmd in ("preflight", "arm_on"):
            rob.last_preflight = {"t": time.time(), "data": data}
        return {"ok": True, "data": data}

    def status(self):
        return {rid: {
            "online_s": round(time.time() - r.since),
            "channels": [{"port": c["port"], "peer": c["peer"],
                          "for_s": round(time.time() - c["since"])} for c in r.channels.values()],
            "last_preflight": r.last_preflight,
        } for rid, r in self.robots.items()}


UI_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Blupe fleet</title>
<style>
 body{font-family:-apple-system,Arial;margin:40px;background:#f7f8fa;color:#1a2233}
 h1{font-size:22px} .arm{background:#fff;border:1px solid #d8dee8;border-radius:10px;
 padding:18px 22px;margin:14px 0;max-width:760px}
 .name{font-size:17px;font-weight:700} .on{color:#1a7a3a} .off{color:#999}
 .ops{font-size:13px;color:#556;margin:8px 0} button{margin-right:8px;padding:7px 14px;
 border-radius:7px;border:1px solid #889;background:#fff;cursor:pointer;font-size:13px}
 button.primary{background:#1a7a3a;color:#fff;border-color:#1a7a3a}
 button.danger{background:#b33;color:#fff;border-color:#b33}
 pre{background:#f2f4f7;border-radius:6px;padding:10px;font-size:12px;overflow-x:auto}
 .muted{color:#889;font-size:12px}
</style></head><body>
<h1>Blupe fleet</h1><div id="arms">loading…</div>
<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
async function api(path){ const r = await fetch(path + (path.includes('?')?'&':'?') + 'token=' + TOKEN);
  return r.json(); }
async function cmd(robot, c, btn){ btn.disabled = true; btn.textContent = c + '…';
  const r = await api('/api/cmd?robot=' + robot + '&cmd=' + c);
  document.getElementById('out-' + robot).textContent = JSON.stringify(r, null, 2);
  btn.disabled = false; btn.textContent = ({arm_on:'Turn ON',arm_off:'Turn OFF',preflight:'Check'})[c];
  refresh(); }
async function refresh(){
  const s = await api('/api/status'); const div = document.getElementById('arms');
  const ids = Object.keys(s); if(!ids.length){ div.innerHTML = '<p class="muted">no arms online</p>'; return; }
  div.innerHTML = ids.map(id => {
    const r = s[id];
    const ops = r.channels.length
      ? r.channels.map(c => `port ${c.port} ← ${c.peer} (${c.for_s}s)`).join('<br>')
      : 'no operator connected';
    const pf = r.last_preflight ? JSON.stringify(r.last_preflight.data, null, 2) : '';
    return `<div class="arm"><span class="name">${id}</span>
      <span class="on">● online ${r.online_s}s</span>
      <div class="ops">${ops}</div>
      <button class="primary" onclick="cmd('${id}','arm_on',this)">Turn ON</button>
      <button class="danger" onclick="cmd('${id}','arm_off',this)">Turn OFF</button>
      <button onclick="cmd('${id}','preflight',this)">Check</button>
      <pre id="out-${id}">${pf}</pre></div>`; }).join('');
}
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


async def _http_ui(relay, args):
    async def handle(reader, writer):
        try:
            req_line = (await asyncio.wait_for(reader.readline(), 10)).decode()
            while (await reader.readline()).strip():            # drain headers
                pass
            _, target, _ = req_line.split(" ", 2)
            url = urllib.parse.urlparse(target)
            q = urllib.parse.parse_qs(url.query)
            token_ok = (q.get("token", [""])[0] == relay.admin_token)

            if url.path == "/" :
                body, ctype, code = UI_HTML, "text/html", 200
            elif not token_ok:
                body, ctype, code = '{"err":"bad admin token"}', "application/json", 403
            elif url.path == "/api/status":
                body, ctype, code = json.dumps(relay.status()), "application/json", 200
            elif url.path == "/api/cmd":
                robot = q.get("robot", [""])[0]
                cmd = q.get("cmd", [""])[0]
                if cmd not in ("preflight", "arm_on", "arm_off"):
                    body, ctype, code = '{"err":"bad cmd"}', "application/json", 400
                else:
                    body = json.dumps(await relay.command(robot, cmd))
                    ctype, code = "application/json", 200
            else:
                body, ctype, code = "not found", "text/plain", 404
            payload = body.encode()
            writer.write((f"HTTP/1.1 {code} OK\r\nContent-Type: {ctype}\r\n"
                          f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode())
            writer.write(payload)
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except OSError:
                pass

    srv = await asyncio.start_server(handle, "0.0.0.0", args.ui_port)
    print(f"[relay] fleet UI on :{args.ui_port}", flush=True)
    async with srv:
        await srv.serve_forever()


async def role_serve(args):
    tokens = dict(p.split(":", 1) for p in os.environ.get("RELAY_TOKENS", "").split(",") if p)
    if not tokens and args.token and args.robot:
        tokens = {args.robot: args.token}
    if not tokens:
        raise SystemExit("set RELAY_TOKENS='robot:token,...' (or --robot/--token)")
    admin = os.environ.get("RELAY_ADMIN_TOKEN", "")
    if not admin:
        raise SystemExit("set RELAY_ADMIN_TOKEN for the fleet UI")
    relay = Relay(tokens, admin)
    srv = await asyncio.start_server(relay.handle, "0.0.0.0", args.port)
    print(f"[relay] serving on :{args.port} for robots {sorted(tokens)}", flush=True)
    async with srv:
        await asyncio.gather(srv.serve_forever(), _http_ui(relay, args))


# --------------------------------------------------------------------------- #
# role: robot (agent: bridge channels + execute fleet commands)
# --------------------------------------------------------------------------- #
class Arm:
    """Owns the serve subprocess + the verification ('is everything working?') checks."""

    def __init__(self, args):
        self.args = args
        self.proc = None

    async def _sh(self, cmd, timeout=10):
        p = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
        return p.returncode, out.decode(errors="replace").strip()

    async def _handshake(self, timeout=6):
        """Connect to the serve and read start_joints — proves motors enumerate + respond."""
        try:
            r, w = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.args.serve_port), timeout=3)
            line = await asyncio.wait_for(r.readline(), timeout=timeout)
            w.close()
            return json.loads(line.decode()).get("start_joints")
        except Exception:
            return None

    async def preflight(self):
        checks = {}
        rc, out = await self._sh(f"ip -br link show {self.args.can}")
        checks["can"] = "UP" if (rc == 0 and "UP" in out) else f"DOWN ({out or 'no interface'})"
        sj = await self._handshake()
        checks["serve"] = {"up": sj is not None, "start_joints": sj}
        rc, out = await self._sh(
            f"curl -s -m 3 -o /dev/null -w '%{{http_code}}' {self.args.camera_url}", timeout=6)
        checks["camera"] = "ok" if out == "200" else f"fail ({out})"
        checks["ok"] = checks["can"] == "UP" and checks["camera"] == "ok"
        return checks

    async def arm_on(self):
        pre = await self.preflight()
        if pre["serve"]["up"]:
            return {"result": "already on", "preflight": pre}
        if pre["can"] != "UP":
            return {"result": "REFUSED: CAN is down (run setup_can.sh)", "preflight": pre}
        self.proc = await asyncio.create_subprocess_shell(
            f"exec {self.args.serve_cmd}", start_new_session=True,
            stdout=open("/tmp/serve_managed.log", "ab"), stderr=asyncio.subprocess.STDOUT)
        for _ in range(25):                       # motor init + gripper calibration takes a while
            await asyncio.sleep(1)
            sj = await self._handshake()
            if sj is not None:
                return {"result": "ON", "start_joints": sj}
            if self.proc.returncode is not None:
                _, tail = await self._sh("tail -3 /tmp/serve_managed.log")
                return {"result": f"serve exited rc={self.proc.returncode}", "log": tail}
        return {"result": "TIMEOUT waiting for serve handshake (arm powered?)"}

    async def arm_off(self):
        killed = []
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()                       # SIGINT is swallowed by i2rt; kill + turn_off
            killed.append("managed serve")
        rc, _ = await self._sh("pkill -9 -f 'yam_real_serve'")
        if rc == 0:
            killed.append("external serve")
        await asyncio.sleep(0.5)
        rc, out = await self._sh(self.args.turnoff_cmd, timeout=30)
        return {"result": "OFF (torque cut)" if rc == 0 else f"turn_off rc={rc}",
                "killed": killed or ["nothing running"], "turnoff_tail": out.splitlines()[-1] if out else ""}


async def _robot_data(args, conn_id, port):
    try:
        lr, lw = await asyncio.open_connection("127.0.0.1", port)
    except OSError as e:
        print(f"[agent] local :{port} refused ({e})", flush=True)
        return
    host, rport = args.relay.rsplit(":", 1)
    rr, rw = await asyncio.open_connection(host, int(rport))
    _send(rw, {"role": "data", "conn": conn_id})
    await rw.drain()
    await _splice(rr, rw, lr, lw, f"local:{port}#{conn_id}")


async def role_robot(args):
    allow = set(args.allow)
    host, rport = args.relay.rsplit(":", 1)
    arm = Arm(args)
    cmds = {"preflight": arm.preflight, "arm_on": arm.arm_on, "arm_off": arm.arm_off}
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, int(rport))
            _send(writer, {"role": "robot", "robot": args.robot, "token": args.token})
            await writer.drain()
            print(f"[agent] registered '{args.robot}' at {args.relay}", flush=True)

            async def ping():
                while True:
                    await asyncio.sleep(PING_S)
                    _send(writer, {"ping": time.time()})
                    await writer.drain()

            async def run_cmd(name, req):
                try:
                    data = await cmds[name]()
                except Exception as e:
                    data = {"result": f"agent error: {e}"}
                _send(writer, {"resp": req, "data": data})
                await writer.drain()
                print(f"[agent] cmd {name} -> {json.dumps(data)[:120]}", flush=True)

            ping_task = asyncio.create_task(ping())
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    msg = json.loads(line.decode())
                    if msg.get("cmd") in cmds:
                        asyncio.create_task(run_cmd(msg["cmd"], msg.get("req")))
                    elif msg.get("open") is not None:
                        port = msg["open"]
                        if port in allow:
                            asyncio.create_task(_robot_data(args, msg.get("conn"), port))
                        else:
                            print(f"[agent] refused channel :{port}", flush=True)
            finally:
                ping_task.cancel()
        except OSError as e:
            print(f"[agent] relay unreachable ({e})", flush=True)
        print("[agent] reconnecting in 3s", flush=True)
        await asyncio.sleep(3)


# --------------------------------------------------------------------------- #
# role: operator (expose the robot's ports locally)
# --------------------------------------------------------------------------- #
async def role_operator(args):
    maps = [tuple(int(x) for x in m.split(":")) for m in args.map]
    host, rport = args.relay.rsplit(":", 1)

    def make_handler(robot_port):
        async def handler(lr, lw):
            try:
                rr, rw = await asyncio.open_connection(host, int(rport))
                _send(rw, {"role": "operator", "robot": args.robot, "token": args.token,
                           "port": robot_port})
                await rw.drain()
                resp = json.loads((await rr.readline()).decode())
                if not resp.get("ok"):
                    print(f"[client] :{robot_port} -> {resp.get('err')}", flush=True)
                    lw.close()
                    return
                await _splice(lr, lw, rr, rw, f"op:{robot_port}")
            except OSError as e:
                print(f"[client] relay error ({e})", flush=True)
                lw.close()
        return handler

    servers = []
    for robot_port, local_port in maps:
        srv = await asyncio.start_server(make_handler(robot_port), "127.0.0.1", local_port)
        servers.append(srv)
        print(f"[client] localhost:{local_port} -> {args.robot}:{robot_port} via {args.relay}",
              flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="role", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8443)
    s.add_argument("--ui-port", type=int, default=8080)
    s.add_argument("--robot"), s.add_argument("--token")

    r = sub.add_parser("robot")
    r.add_argument("--relay", required=True)
    r.add_argument("--robot", required=True)
    r.add_argument("--token", required=True)
    r.add_argument("--allow", type=int, nargs="+", default=[5599, 8089])
    r.add_argument("--can", default="can0")
    r.add_argument("--serve-port", type=int, default=5599)
    r.add_argument("--camera-url", default="http://127.0.0.1:8089/0")
    r.add_argument("--serve-cmd", default=os.path.expanduser(
        "~/i2rt/.venv/bin/python ~/blupe-evals/YAM_control/yam_real_serve.py --channel can0"))
    r.add_argument("--turnoff-cmd", default=os.path.expanduser(
        "~/i2rt/.venv/bin/python ~/blupe-evals/YAM_control/turn_off.py --channel can0"))

    o = sub.add_parser("operator")
    o.add_argument("--relay", required=True)
    o.add_argument("--robot", required=True)
    o.add_argument("--token", required=True)
    o.add_argument("--map", nargs="+", default=["5599:15599", "8089:18089"])

    args = ap.parse_args()
    asyncio.run({"serve": role_serve, "robot": role_robot, "operator": role_operator}[args.role](args))


if __name__ == "__main__":
    main()
