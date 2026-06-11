"""Blupe relay v1 — outbound-only transport between operator and robot nodes (no VPN).

One stdlib file, three roles (PLAN "Customer transport"):

  serve     the hosted rendezvous: robots register, operators request channels, the relay
            authenticates (per-robot token), splices bytes, and can kill any robot's session.
  robot     runs at the robot site; dials OUT to the relay, and bridges relay channels to
            local services on an allowlist (the joints serve :5599, the camera relay :8089).
  operator  runs at the operator site; dials OUT to the relay and exposes the robot's
            services on LOCAL ports (5599 -> 15599, 8089 -> 18089 by default), so the eval
            runs unchanged:  --serve-host 127.0.0.1 --serve-port 15599
                             --cameras http://127.0.0.1:18089/0 http://127.0.0.1:18089/2

Both ends only ever make OUTBOUND connections to the relay — nothing at either site needs
a reachable address, port-forward, or VPN. Robot-side safety (clamp, hold-on-drop,
watchdog) rides underneath, untouched: a relay/WAN drop looks like a client disconnect to
the serve, which holds pose.

Wire: every connection starts with one newline-JSON hello, then either stays as the robot's
control channel (relay -> robot "open" requests, robot -> relay pings) or becomes a raw
spliced byte pipe (data connections).

  robot ctrl  -> {"role": "robot", "robot": ID, "token": T}
  operator    -> {"role": "operator", "robot": ID, "token": T, "port": 5599}
  robot data  -> {"role": "data", "conn": C}
  relay->robot ctrl: {"open": PORT, "conn": C}   relay->operator: {"ok": true} | {"err": ...}

v1 scope: one shared token per robot (RELAY_TOKENS env: "robot1:tok1,robot2:tok2"), plain
TCP (put TLS in front via the host, or upgrade to WSS in v2), no persistence.

Run:
  relay:    RELAY_TOKENS="yam-1:secret" python relay/relay.py serve --port 8443
  robot:    python relay/relay.py robot --relay HOST:8443 --robot yam-1 --token secret \
                --allow 5599 8089
  operator: python relay/relay.py operator --relay HOST:8443 --robot yam-1 --token secret \
                --map 5599:15599 8089:18089
"""

import argparse
import asyncio
import itertools
import json
import os
import time

PING_S = 15           # robot ctrl keepalive
OPEN_TIMEOUT_S = 10   # operator waits this long for the robot's data dial-back


async def _read_hello(reader):
    line = await asyncio.wait_for(reader.readline(), timeout=10)
    return json.loads(line.decode())


def _send(writer, obj):
    writer.write((json.dumps(obj) + "\n").encode())


async def _splice(a_reader, a_writer, b_reader, b_writer, tag):
    """Pipe bytes both ways until either side closes; then close both."""
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
    print(f"[relay] {tag} closed", flush=True)


# --------------------------------------------------------------------------- #
# role: serve (the hosted rendezvous)
# --------------------------------------------------------------------------- #
class Relay:
    def __init__(self, tokens):
        self.tokens = tokens                  # robot_id -> token
        self.robots = {}                      # robot_id -> control writer
        self.pending = {}                     # conn_id -> Future[(reader, writer)]
        self.ids = itertools.count(1)

    def auth(self, robot, token):
        return self.tokens.get(robot) is not None and self.tokens[robot] == token

    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            hello = await _read_hello(reader)
        except Exception:
            writer.close()
            return
        role = hello.get("role")

        if role == "robot":
            robot, token = hello.get("robot"), hello.get("token")
            if not self.auth(robot, token):
                _send(writer, {"err": "auth"})
                writer.close()
                return
            old = self.robots.get(robot)
            if old is not None:
                old.close()
            self.robots[robot] = writer
            print(f"[relay] robot '{robot}' online from {peer}", flush=True)
            try:                                            # hold ctrl conn; robot pings
                while await reader.readline():
                    pass
            finally:
                if self.robots.get(robot) is writer:
                    del self.robots[robot]
                print(f"[relay] robot '{robot}' offline", flush=True)

        elif role == "data":
            fut = self.pending.pop(hello.get("conn"), None)
            if fut is None or fut.done():
                writer.close()
                return
            fut.set_result((reader, writer))                # operator side completes splice

        elif role == "operator":
            robot, token, port = hello.get("robot"), hello.get("token"), hello.get("port")
            if not self.auth(robot, token):
                _send(writer, {"err": "auth"})
                writer.close()
                return
            ctrl = self.robots.get(robot)
            if ctrl is None:
                _send(writer, {"err": f"robot '{robot}' offline"})
                writer.close()
                return
            conn_id = next(self.ids)
            fut = asyncio.get_running_loop().create_future()
            self.pending[conn_id] = fut
            _send(ctrl, {"open": port, "conn": conn_id})
            try:
                await ctrl.drain()
                r2, w2 = await asyncio.wait_for(fut, timeout=OPEN_TIMEOUT_S)
            except Exception as e:
                self.pending.pop(conn_id, None)
                _send(writer, {"err": f"robot did not answer: {e}"})
                writer.close()
                return
            _send(writer, {"ok": True})
            await writer.drain()
            await _splice(reader, writer, r2, w2, f"{robot}:{port}#{conn_id}")
        else:
            writer.close()

    def kill(self, robot):
        ctrl = self.robots.get(robot)
        if ctrl is not None:
            ctrl.close()


async def role_serve(args):
    tokens = dict(p.split(":", 1) for p in os.environ.get("RELAY_TOKENS", "").split(",") if p)
    if not tokens and args.token and args.robot:
        tokens = {args.robot: args.token}
    if not tokens:
        raise SystemExit("set RELAY_TOKENS='robot:token,...' (or --robot/--token)")
    relay = Relay(tokens)
    srv = await asyncio.start_server(relay.handle, "0.0.0.0", args.port)
    print(f"[relay] serving on :{args.port} for robots {sorted(tokens)}", flush=True)
    async with srv:
        await srv.serve_forever()


# --------------------------------------------------------------------------- #
# role: robot (robot-site agent; dials out, bridges allowlisted local ports)
# --------------------------------------------------------------------------- #
async def _robot_data(args, conn_id, port):
    try:
        lr, lw = await asyncio.open_connection("127.0.0.1", port)
    except OSError as e:
        print(f"[agent] local :{port} refused ({e}) — is the service up?", flush=True)
        return
    host, rport = args.relay.rsplit(":", 1)
    rr, rw = await asyncio.open_connection(host, int(rport))
    _send(rw, {"role": "data", "conn": conn_id})
    await rw.drain()
    print(f"[agent] channel open :{port} (conn {conn_id})", flush=True)
    await _splice(rr, rw, lr, lw, f"local:{port}#{conn_id}")


async def role_robot(args):
    allow = set(args.allow)
    host, rport = args.relay.rsplit(":", 1)
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, int(rport))
            _send(writer, {"role": "robot", "robot": args.robot, "token": args.token})
            await writer.drain()
            print(f"[agent] registered '{args.robot}' at {args.relay}; allow {sorted(allow)}",
                  flush=True)

            async def ping():
                while True:
                    await asyncio.sleep(PING_S)
                    _send(writer, {"ping": time.time()})
                    await writer.drain()

            ping_task = asyncio.create_task(ping())
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    msg = json.loads(line.decode())
                    port, conn_id = msg.get("open"), msg.get("conn")
                    if port is None:
                        continue
                    if port not in allow:
                        print(f"[agent] refused channel :{port} (not in allowlist)", flush=True)
                        continue
                    asyncio.create_task(_robot_data(args, conn_id, port))
            finally:
                ping_task.cancel()
        except OSError as e:
            print(f"[agent] relay unreachable ({e})", flush=True)
        print("[agent] reconnecting in 3s", flush=True)
        await asyncio.sleep(3)


# --------------------------------------------------------------------------- #
# role: operator (operator-site client; exposes the robot's ports locally)
# --------------------------------------------------------------------------- #
async def role_operator(args):
    maps = [tuple(int(x) for x in m.split(":")) for m in args.map]   # (robot_port, local_port)
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
    s.add_argument("--robot"), s.add_argument("--token")

    r = sub.add_parser("robot")
    r.add_argument("--relay", required=True)
    r.add_argument("--robot", required=True)
    r.add_argument("--token", required=True)
    r.add_argument("--allow", type=int, nargs="+", default=[5599, 8089])

    o = sub.add_parser("operator")
    o.add_argument("--relay", required=True)
    o.add_argument("--robot", required=True)
    o.add_argument("--token", required=True)
    o.add_argument("--map", nargs="+", default=["5599:15599", "8089:18089"])

    args = ap.parse_args()
    asyncio.run({"serve": role_serve, "robot": role_robot, "operator": role_operator}[args.role](args))


if __name__ == "__main__":
    main()
