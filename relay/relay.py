"""Blupe relay — outbound-only transport between operator and robot nodes + fleet UI.

One stdlib file, three roles (PLAN "Customer transport"):

  serve     the hosted rendezvous: robots register, operators request channels, the relay
            authenticates operators, splices bytes, and serves a FLEET UI (web)
            that lists arms/operators and can turn an arm ON (with preflight verification)
            or OFF (kill serve + guaranteed torque-off).
  robot     runs at the robot site; dials OUT to the relay, bridges relay channels to local
            services on an allowlist, and executes fleet commands (preflight/arm_on/arm_off).
  operator  runs at the operator site; dials OUT and exposes the robot's services on LOCAL
            ports (5599 -> 15599, 8089 -> 18089), so the eval runs unchanged.

Both ends only ever make OUTBOUND connections — no VPN, no port-forwards at either site.
Robot-side safety (clamp, hold-on-drop, watchdog) rides underneath, untouched.

Wire (newline-JSON hello, then either a held control channel or a raw spliced data pipe):
  robot ctrl   -> {"role": "robot", "robot": ID}
  operator     -> {"role": "operator", "robot": ID, "token": T, "port": 5599}
  robot data   -> {"role": "data", "conn": C}
  relay->robot:   {"open": PORT, "conn": C} | {"cmd": NAME, "req": N}
  robot->relay:   {"ping": t} | {"resp": N, "data": {...}}

Fleet UI: http://<relay>:8080/?token=<RELAY_ADMIN_TOKEN>   (set both env vars on the host:
RELAY_TOKENS="yam-1:tok,..." RELAY_ADMIN_TOKEN="...").

Run:
  relay:    RELAY_TOKENS="yam-1:tok" RELAY_ADMIN_TOKEN="atok" \
                python3 relay.py serve --port 8443 --ui-port 8080
  robot:    python3 relay.py robot --relay HOST:8443 --robot yam-1
  operator: python3 relay.py operator --relay HOST:8443 --robot yam-1 --token tok
"""

import argparse
import asyncio
import itertools
import json
import os
import secrets
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
    def __init__(self, state_path, env_tokens, admin_token):
        self.state_path = state_path
        self.admin_token = admin_token
        # Fleet registry, PERSISTED: robots (id -> token+label) and users/customers
        # (id -> token+label+linked robot ids). Admin mutations take effect live — no restart.
        self.fleet = {"robots": {}, "users": {}}
        if state_path and os.path.exists(state_path):
            try:
                f = json.load(open(state_path))
                self.fleet = {"robots": f.get("robots", {}), "users": f.get("users", {})}
            except (ValueError, OSError) as e:
                print(f"[relay] WARNING: {state_path} unreadable ({e}); starting empty", flush=True)
        for rid, tok in (env_tokens or {}).items():     # legacy RELAY_TOKENS -> registry
            self.fleet["robots"].setdefault(rid, {"token": tok, "label": rid})
        self._save()
        self.robots = {}              # robot_id -> Robot (LIVE connections)
        self.pending = {}             # conn_id -> Future[(reader, writer)]
        self.cmd_waiters = {}         # req_id -> Future[dict]
        self.ids = itertools.count(1)

    def _save(self):
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.fleet, f, indent=2)
        os.replace(tmp, self.state_path)                # atomic: never a torn fleet file

    # ---- auth ---------------------------------------------------------------- #
    def _valid_robot_id(self, robot):
        return isinstance(robot, str) and bool(robot.strip())

    def ensure_robot(self, robot, label=None):
        if not self._valid_robot_id(robot):
            return False
        if robot not in self.fleet["robots"]:
            self.fleet["robots"][robot] = {"label": label or robot}
            self._save()
        return True

    def auth_robot(self, robot, token=None):
        """Robot agents are authenticated by the outbound control channel, not a token."""
        return self.ensure_robot(robot)

    def auth_robot_token(self, robot, token):
        """Legacy operator access: accept old per-robot tokens if still present."""
        r = self.fleet["robots"].get(robot)
        return r is not None and bool(token) and bool(r.get("token")) and r.get("token") == token

    def auth_operator(self, robot, token):
        """Data-plane access to a robot: its own token, or a user LINKED to it."""
        if self.auth_robot_token(robot, token):
            return True
        return any(bool(token) and u.get("token") == token and robot in u.get("robots", [])
                   for u in self.fleet["users"].values())

    def viewer(self, token):
        """UI access level: ('admin', None) | ('user', uid) | None."""
        if token and token == self.admin_token:
            return ("admin", None)
        for uid, u in self.fleet["users"].items():
            if token and u.get("token") == token:
                return ("user", uid)
        return None

    def visible_robots(self, who):
        kind, uid = who
        if kind == "admin":
            return set(self.fleet["robots"]) | set(self.robots)
        return set(self.fleet["users"].get(uid, {}).get("robots", []))

    # ---- fleet admin (UI: add arms, add customers, link/unlink) -------------- #
    def fleet_action(self, action, q):
        rid = q.get("robot", [""])[0]
        uid = q.get("user", [""])[0]
        label = q.get("label", [""])[0]
        if action == "add_robot":
            if not rid or rid in self.fleet["robots"]:
                return {"ok": False, "err": "missing or duplicate robot id"}
            self.fleet["robots"][rid] = {"label": label or rid}
            self._save()
            return {"ok": True, "robot": rid,
                    "install": (f"python3 relay.py robot --relay <RELAY_HOST>:8443 "
                                f"--robot {rid}")}
        if action == "add_user":
            if not uid or uid in self.fleet["users"]:
                return {"ok": False, "err": "missing or duplicate user id"}
            tok = secrets.token_hex(16)
            self.fleet["users"][uid] = {"token": tok, "label": label or uid, "robots": []}
            self._save()
            return {"ok": True, "user": uid, "token": tok,
                    "ui": f"http://<RELAY_HOST>:8080/?token={tok}"}
        if action in ("link", "unlink"):
            u = self.fleet["users"].get(uid)
            if u is None or rid not in self.fleet["robots"]:
                return {"ok": False, "err": "unknown user or robot"}
            robots = set(u.get("robots", []))
            (robots.add if action == "link" else robots.discard)(rid)
            u["robots"] = sorted(robots)
            self._save()
            return {"ok": True, "user": uid, "robots": u["robots"]}
        if action == "list":                            # admin-only; includes tokens
            return {"ok": True, "fleet": self.fleet}
        return {"ok": False, "err": "bad action"}

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
            if not self.auth_robot(robot_id, token):
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
            if not self.auth_operator(robot_id, token):   # legacy robot token OR linked user token
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

    async def open_channel(self, robot_id, port):
        """Open a data channel to a robot's local port (same machinery operators use)."""
        rob = self.robots.get(robot_id)
        if rob is None:
            raise ConnectionError(f"robot '{robot_id}' offline")
        conn_id = next(self.ids)
        fut = asyncio.get_running_loop().create_future()
        self.pending[conn_id] = fut
        _send(rob.writer, {"open": port, "conn": conn_id})
        await rob.writer.drain()
        try:
            return await asyncio.wait_for(fut, timeout=OPEN_TIMEOUT_S)
        except Exception:
            self.pending.pop(conn_id, None)
            raise

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

    def status(self, visible=None):
        """Registered + live robots merged; offline arms still get a card (onboarding UX:
        add the arm -> see it -> install the agent -> watch it come online)."""
        out = {}
        for rid in sorted(set(self.fleet["robots"]) | set(self.robots)):
            if visible is not None and rid not in visible:
                continue
            r = self.robots.get(rid)
            reg = self.fleet["robots"].get(rid, {})
            out[rid] = {
                "online": r is not None,
                "label": reg.get("label", rid),
                "linked": sorted(u for u, d in self.fleet["users"].items()
                                 if rid in d.get("robots", [])),
                "online_s": round(time.time() - r.since) if r else 0,
                "channels": [{"port": c["port"], "peer": c["peer"],
                              "for_s": round(time.time() - c["since"])}
                             for c in r.channels.values()] if r else [],
                "last_preflight": r.last_preflight if r else None,
            }
        return out


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
<h1>Blupe fleet</h1>
<div class="arm" id="admin-card" style="display:none">
  <span class="name">Fleet admin</span>
  <button class="primary" onclick="addArm()">Add arm</button>
  <button class="primary" onclick="addCustomer()">Add customer</button>
  <pre id="admin-out"></pre></div>
<div class="arm" id="sim-card" style="display:none"><span class="name">Operator launcher</span>
  <div class="ops">choose what the operator console/headset controls on the Mac:
  <b>yam</b> = real YAM arm, other profiles = sim-only standards. Same console (:8810).</div>
  <div id="sim-arms">loading…</div>
  <pre id="sim-out"></pre></div>
<div class="arm" id="report-card" style="display:none"><span class="name">Eval report</span>
  <div class="ops">runs on the operator Mac: rotates to a fresh trial session + records the
  full operator view (session.mp4); Finish renders report.html</div>
  <button class="primary" onclick="report('new',this)">New report</button>
  <button onclick="report('finish',this)">Finish report</button>
  <button onclick="report('status',this)">Status</button>
  <a href="/reports" target="_blank" style="margin-left:12px">Public reports &#8599;</a>
  <pre id="report-out"></pre></div>
<div id="arms">loading…</div>
<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
let ME = {role: 'user'}, USERS = [];
async function api(path){ const r = await fetch(path + (path.includes('?')?'&':'?') + 'token=' + TOKEN);
  return r.json(); }
async function whoami(){ ME = await api('/api/me');
  if (ME.role === 'admin'){
    document.getElementById('admin-card').style.display = '';
    document.getElementById('report-card').style.display = '';
    document.getElementById('sim-card').style.display = '';
    simRefresh(); }
  if (ME.role === 'admin') await loadFleet(); }
async function simRefresh(){
  const s = await api('/api/sim?action=status');
  if (!s.ok){ document.getElementById('sim-arms').textContent = s.err || 'launcher offline'; return; }
  const label = n => n === 'yam' ? 'yam (real)' : n + ' (sim)';
  const policy = a => a && a.policy ? a.policy : 'none';
  const currentPolicy = s.running && s.arm ? (s.policy || policy(s.arms[s.arm])) : 'none';
  const current = s.running && s.arm ? `current: ${label(s.arm)}` : 'current: none';
  document.getElementById('sim-arms').innerHTML =
    `<div class="ops">${current}; policy: ${currentPolicy}</div>` +
    Object.entries(s.arms).map(([n, a]) =>
    `<div class="ops"><button ${a.status !== 'ready' ? 'disabled' : ''}
       class="${s.arm === n ? 'primary' : ''}"
       onclick="simLaunch('${n}', this)">${label(n)}${s.arm === n && s.running ? ' \\u25cf running' : ''}</button>
       <span class="muted">policy: ${policy(a)}</span></div>`).join('');
}
async function simLaunch(n, btn){ btn.disabled = true; btn.textContent = n + '\\u2026';
  const r = await api('/api/sim?action=launch&arm=' + encodeURIComponent(n));
  document.getElementById('sim-out').textContent = JSON.stringify(r, null, 2);
  setTimeout(simRefresh, 9000); simRefresh(); }
async function loadFleet(){ const r = await api('/api/fleet?action=list');
  if (r.ok) USERS = Object.keys(r.fleet.users); }
async function fleetAct(params){
  const out = document.getElementById('admin-out');
  const r = await api('/api/fleet?' + params);
  out.textContent = JSON.stringify(r, null, 2);
  if (!r.ok) return;
  await loadFleet(); refresh(); }
function addArm(){ const id = prompt('arm id (e.g. yam-2):');
  if (id) fleetAct('action=add_robot&robot=' + encodeURIComponent(id.trim())); }
function addCustomer(){ const id = prompt('customer id (e.g. acme):');
  if (id) fleetAct('action=add_user&user=' + encodeURIComponent(id.trim())); }
function linkSel(rid){ const uid = document.getElementById('link-' + rid).value;
  if (uid) fleetAct('action=link&robot=' + encodeURIComponent(rid) +
                    '&user=' + encodeURIComponent(uid)); }
function unlink(rid, uid){ fleetAct('action=unlink&robot=' + encodeURIComponent(rid) +
                                    '&user=' + encodeURIComponent(uid)); }
async function report(action, btn){ btn.disabled = true;
  try { const r = await api('/api/report?action=' + action);
        document.getElementById('report-out').textContent = JSON.stringify(r, null, 2); }
  finally { btn.disabled = false; } }
async function cmd(robot, c, btn){ btn.disabled = true; btn.textContent = c + '…';
  const labels = {arm_on:'Turn ON',arm_off:'Turn OFF',preflight:'Check'};
  const r = await api('/api/cmd?robot=' + encodeURIComponent(robot) + '&cmd=' + encodeURIComponent(c));
  document.getElementById('out-' + robot).textContent = JSON.stringify(r, null, 2);
  btn.disabled = false; btn.textContent = labels[c];
  if (c === 'arm_on' && r.ok && r.data && (r.data.result === 'ON' || r.data.result === 'already on'))
    showCams(robot, true);                                  // ON => show what the arm sees
  refresh(); }
function showCams(id, on){
  const d = document.getElementById('cams-' + id);
  const btn = document.getElementById('camsbtn-' + id);
  if (on === undefined) on = !d.hasChildNodes();
  d.innerHTML = on ? ['0','2'].map(i =>
    `<img src="/cam/${id}/${i}?token=${TOKEN}" width="340"
          style="border-radius:6px;margin:6px 8px 0 0" alt="camera ${i}">`).join('') : '';
  btn.textContent = on ? 'Hide cameras' : 'Cameras'; }
function card(id){                                          // built ONCE; refresh only edits text
  const el = document.createElement('div'); el.className = 'arm'; el.id = 'arm-' + id;
  const admin = ME.role === 'admin' ? `
    <div class="ops">customers: <span id="links-${id}"></span>
      <select id="link-${id}"></select>
      <button id="linkbtn-${id}" onclick="linkSel('${id}')">Link</button>
      <span class="muted" id="linkmsg-${id}"></span></div>` : '';
  el.innerHTML = `<span class="name" id="nm-${id}">${id}</span> <span class="on" id="st-${id}"></span>
    <div class="ops" id="ops-${id}"></div>${admin}
    <button class="primary" onclick="cmd('${id}','arm_on',this)">Turn ON</button>
    <button class="danger" onclick="cmd('${id}','arm_off',this)">Turn OFF</button>
    <button onclick="cmd('${id}','preflight',this)">Check</button>
    <button id="camsbtn-${id}" onclick="showCams('${id}')">Cameras</button>
    <div id="cams-${id}"></div>
    <pre id="out-${id}"></pre>`;
  document.getElementById('arms').appendChild(el); return el; }
async function refresh(){
  const s = await api('/api/status'); const div = document.getElementById('arms');
  const ids = Object.keys(s).filter(id => !id.startsWith('mac-'));  // operator nodes: no arm card
  if (div.textContent === 'loading…' || (!ids.length && !div.querySelector('.arm')))
    div.innerHTML = ids.length ? '' : '<p class="muted">no arms registered</p>';
  for (const id of ids){
    const el = document.getElementById('arm-' + id) || card(id);
    const on = s[id].online;
    el.style.opacity = on ? 1 : 0.55;
    document.getElementById('nm-' + id).textContent = s[id].label || id;
    const st = document.getElementById('st-' + id);
    st.textContent = on ? `● online ${s[id].online_s}s` : '○ offline';
    st.className = on ? 'on' : 'off';
    document.getElementById('ops-' + id).innerHTML = !on
      ? 'agent not connected — run the install one-liner on the robot computer'
      : s[id].channels.length
        ? s[id].channels.map(c => `port ${c.port} ← ${c.peer} (${c.for_s}s)`).join('<br>')
        : 'no operator connected';
    if (ME.role === 'admin'){
      document.getElementById('links-' + id).innerHTML = (s[id].linked || []).map(u =>
        `<b>${u}</b> <button onclick="unlink('${id}','${u}')" title="unlink">✕</button>`)
        .join(' ') || '<span class="muted">none</span>';
      const sel = document.getElementById('link-' + id);
      const opts = [''].concat(USERS.filter(u => !(s[id].linked || []).includes(u)));
      sel.innerHTML = opts.map(u => `<option value="${u}">${u || 'customer…'}</option>`).join('');
      const btn = document.getElementById('linkbtn-' + id);
      const msg = document.getElementById('linkmsg-' + id);
      btn.disabled = opts.length <= 1;
      msg.textContent = USERS.length ? (opts.length <= 1 ? 'all customers linked' : '') : 'add a customer first';
    }
    const pre = document.getElementById('out-' + id);
    if (!pre.textContent && s[id].last_preflight)
      pre.textContent = JSON.stringify(s[id].last_preflight.data, null, 2);
  }
}
whoami().then(refresh); setInterval(refresh, 5000);
</script></body></html>"""


async def _proxy_mac(relay, op, port, path):
    """One HTTP GET to an operator-node service over a relay channel; returns the body.
    Reads by Content-Length, NOT to EOF — the agent-side splice keeps the channel open
    until both directions close, so EOF never comes."""
    try:
        rr, rw = await relay.open_channel(op, port)
        rw.write(f"GET {path} HTTP/1.1\r\nHost: op\r\nConnection: close\r\n\r\n".encode())
        await rw.drain()
        raw = b""
        while b"\r\n\r\n" not in raw:
            data = await asyncio.wait_for(rr.read(65536), CMD_TIMEOUT_S)
            if not data:
                break
            raw += data
        head, _, payload = raw.partition(b"\r\n\r\n")
        clen = 0
        for h in head.split(b"\r\n"):
            if h.lower().startswith(b"content-length:"):
                clen = int(h.split(b":")[1])
        while len(payload) < clen:
            data = await asyncio.wait_for(rr.read(65536), CMD_TIMEOUT_S)
            if not data:
                break
            payload += data
        rw.close()
        return payload.decode() if payload else '{"ok": false, "err": "bad upstream response"}'
    except Exception as e:
        return json.dumps({"ok": False, "err": f"operator node: {e}"})


_CTYPES = {".html": "text/html", ".mp4": "video/mp4", ".json": "application/json",
           ".png": "image/png", ".jpg": "image/jpeg", ".css": "text/css"}


async def _serve_report(writer, reports_dir, path, range_hdr):
    """Static file server for PUBLIC published reports under reports_dir. /reports = an
    index of published sessions; /reports/<session>/<file> = the file, with Range support
    so report videos seek in the browser. Path-traversal is contained to reports_dir."""
    def respond(code, body, ctype="text/html", extra=b""):
        if isinstance(body, str):
            body = body.encode()
        writer.write(f"HTTP/1.1 {code} OK\r\nContent-Type: {ctype}\r\n".encode() + extra +
                     f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)

    rel = urllib.parse.unquote(path[len("/reports"):]).lstrip("/")
    base = os.path.realpath(reports_dir)
    if not rel:                                             # index of published sessions
        try:
            sessions = sorted((d for d in os.listdir(base)
                               if os.path.exists(os.path.join(base, d, "report.html"))),
                              reverse=True)
        except OSError:
            sessions = []
        links = "".join(f'<li><a href="/reports/{s}/report.html">{s}</a></li>'
                        for s in sessions) or "<li>no reports published yet</li>"
        respond(200, f"<!doctype html><title>Blupe reports</title><body style='font:15px "
                     f"-apple-system;margin:40px'><h1>Published eval reports</h1><ul>{links}"
                     f"</ul></body>")
        await writer.drain()
        return

    full = os.path.realpath(os.path.join(base, rel))
    if not full.startswith(base + os.sep) or not os.path.isfile(full):
        respond(404, "not found", "text/plain")
        await writer.drain()
        return

    ctype = _CTYPES.get(os.path.splitext(full)[1].lower(), "application/octet-stream")
    size = os.path.getsize(full)
    start, end = 0, size - 1
    if range_hdr and range_hdr.startswith("bytes="):        # video seeking -> 206 partial
        s, _, e = range_hdr[6:].partition("-")
        start = int(s) if s else 0
        end = int(e) if e else size - 1
        end = min(end, size - 1)
    partial = range_hdr is not None
    with open(full, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)
    code = "206 Partial Content" if partial else "200 OK"
    extra = (f"Accept-Ranges: bytes\r\nContent-Range: bytes {start}-{end}/{size}\r\n".encode()
             if partial else b"Accept-Ranges: bytes\r\n")
    writer.write(f"HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\n".encode() + extra +
                 f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
    await writer.drain()


async def _http_ui(relay, args):
    async def handle(reader, writer):
        try:
            req_line = (await asyncio.wait_for(reader.readline(), 10)).decode()
            req_headers = {}
            while True:                                         # read headers (keep Range)
                line = (await reader.readline()).decode()
                if not line.strip():
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    req_headers[k.strip().lower()] = v.strip()
            _, target, _ = req_line.split(" ", 2)
            url = urllib.parse.urlparse(target)
            q = urllib.parse.parse_qs(url.query)
            who = relay.viewer(q.get("token", [""])[0])   # ('admin',None)|('user',uid)|None
            token_ok = who is not None
            is_admin = who is not None and who[0] == "admin"

            if url.path == "/reports" or url.path.startswith("/reports/"):
                # PUBLIC (no token): published eval reports — static files under args.reports_dir.
                await _serve_report(writer, args.reports_dir, url.path,
                                    req_headers.get("range"))
                return

            if url.path.startswith("/cam/") and token_ok:
                # /cam/<robot>/<idx> -> open a channel to the robot's camera relay (:8089),
                # send a raw GET, splice the multipart-MJPEG response verbatim to the browser.
                _, _, rid, idx = url.path.split("/", 3)
                if rid not in relay.visible_robots(who):
                    writer.close()
                    return
                rr, rw = await relay.open_channel(rid, 8089)
                rw.write(f"GET /{idx} HTTP/1.1\r\nHost: cam\r\n\r\n".encode())
                await rw.drain()
                try:
                    while True:
                        data = await rr.read(65536)
                        if not data:
                            break
                        writer.write(data)
                        await writer.drain()
                except (ConnectionError, OSError):
                    pass
                finally:
                    rw.close()
                return

            if url.path == "/" :
                body, ctype, code = UI_HTML, "text/html", 200
            elif not token_ok:
                body, ctype, code = '{"err":"bad token"}', "application/json", 403
            elif url.path == "/api/me":
                body, ctype, code = json.dumps({"role": who[0], "id": who[1]}), \
                    "application/json", 200
            elif url.path == "/api/status":
                body, ctype, code = json.dumps(relay.status(relay.visible_robots(who))), \
                    "application/json", 200
            elif url.path == "/api/fleet":
                if not is_admin:
                    body, ctype, code = '{"err":"admin only"}', "application/json", 403
                else:
                    body = json.dumps(relay.fleet_action(q.get("action", [""])[0], q))
                    ctype, code = "application/json", 200
            elif url.path == "/api/report" and not is_admin:
                body, ctype, code = '{"err":"admin only"}', "application/json", 403
            elif url.path == "/api/report":
                # Proxy to the operator Mac's preview server (:8810) over a relay channel —
                # the Mac registers like a robot (id mac-*) so the same machinery applies.
                action = q.get("action", [""])[0]
                op = q.get("op", ["mac-1"])[0]
                if action not in ("new", "finish", "status") or not op.startswith("mac-"):
                    body, ctype, code = '{"err":"bad report request"}', "application/json", 400
                else:
                    body = await _proxy_mac(relay, op, 8810, f"/report/{action}")
                    ctype, code = "application/json", 200
            elif url.path == "/api/sim" and not is_admin:
                body, ctype, code = '{"err":"admin only"}', "application/json", 403
            elif url.path == "/api/sim":
                # Sim-arm launcher on the operator Mac (eval_launcher.py :8809): list the
                # registered arm standards / relaunch the eval with one (sim, no hardware).
                action = q.get("action", [""])[0]
                arm = q.get("arm", [""])[0]
                op = q.get("op", ["mac-1"])[0]
                if action not in ("status", "launch") or not op.startswith("mac-") or \
                        (action == "launch" and not arm.replace("-", "").isalnum()):
                    body, ctype, code = '{"err":"bad sim request"}', "application/json", 400
                else:
                    path = "/status" if action == "status" else f"/launch?arm={arm}"
                    body = await _proxy_mac(relay, op, 8809, path)
                    ctype, code = "application/json", 200
            elif url.path == "/api/cmd":
                robot = q.get("robot", [""])[0]
                cmd = q.get("cmd", [""])[0]
                if cmd not in ("preflight", "arm_on", "arm_off"):
                    body, ctype, code = '{"err":"bad cmd"}', "application/json", 400
                elif robot not in relay.visible_robots(who):
                    body, ctype, code = '{"err":"not your arm"}', "application/json", 403
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
    admin = os.environ.get("RELAY_ADMIN_TOKEN", "")
    if not admin:
        raise SystemExit("set RELAY_ADMIN_TOKEN for the fleet UI")
    relay = Relay(args.state, tokens, admin)
    srv = await asyncio.start_server(relay.handle, "0.0.0.0", args.port)
    print(f"[relay] serving on :{args.port}; fleet: {sorted(relay.fleet['robots'])} "
          f"(state: {args.state})", flush=True)
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
        self.lock = asyncio.Lock()

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

    async def _shutdown_serve(self):
        """Ask the serve to torque off and exit via its own protocol."""
        try:
            r, w = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.args.serve_port), timeout=3)
            _send(w, {"shutdown": True})
            await w.drain()
            w.close()
            await w.wait_closed()
        except Exception as e:
            return False, f"shutdown request failed: {e}"

        if self.proc is not None and self.proc.returncode is None:
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=12)
            except asyncio.TimeoutError:
                return False, "shutdown request sent; managed serve did not exit"

        for _ in range(20):
            if await self._handshake(timeout=1) is None:
                return True, "serve shutdown"
            await asyncio.sleep(0.25)
        return False, "shutdown request sent; serve still answers"

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
        async with self.lock:
            return await self._arm_on()

    async def _arm_on(self):
        """ON = headset-ready: cameras streaming + serve up with motors answering. The
        operator still explicitly CONNECTs in the headset — that handoff stays deliberate."""
        pre = await self.preflight()
        if pre["camera"] != "ok":                  # cameras are part of 'operable'
            await asyncio.create_subprocess_shell(
                f"exec {self.args.camera_cmd}", start_new_session=True,
                stdout=open("/tmp/camera_managed.log", "ab"), stderr=asyncio.subprocess.STDOUT)
            await asyncio.sleep(3)
            pre = await self.preflight()
            if pre["camera"] != "ok":
                return {"result": "REFUSED: cameras won't start", "preflight": pre}
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
        async with self.lock:
            return await self._arm_off()

    async def _arm_off(self):
        killed = []
        graceful, detail = await self._shutdown_serve()
        if graceful:
            self.proc = None
            return {"result": "OFF (serve shutdown)", "killed": [detail], "turnoff_tail": ""}

        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()                       # SIGINT is swallowed by i2rt; kill + turn_off
            killed.append("managed serve")
        try:
            rc, _ = await self._sh(
                "ps -eo pid=,args= | "
                "awk '/yam_real_serve[.]py|lerobot_robot_yam[.]yam_serve/ && "
                "!/relay[.]py robot/ {print $1}' | xargs -r kill -9"
            )
            if rc == 0:
                killed.append("external serve")
        except asyncio.TimeoutError:
            killed.append("pkill timed out")
        await asyncio.sleep(0.5)
        try:
            rc, out = await self._sh(self.args.turnoff_cmd, timeout=40)
            result = "OFF (torque cut)" if rc == 0 else f"turn_off rc={rc}"
            tail = out.splitlines()[-1] if out else ""
        except asyncio.TimeoutError:
            result, tail = "serve killed; turn_off TIMED OUT (arm powered?)", ""
        if detail:
            killed.insert(0, detail)
        return {"result": result, "killed": killed or ["nothing running"], "turnoff_tail": tail}


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
                    data = {"result": f"agent error: {type(e).__name__}: {e}"}
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
    s.add_argument("--state", default=os.environ.get("RELAY_STATE", "fleet.json"),
                   help="fleet registry file (robots+users+links; admin edits persist here)")
    s.add_argument("--reports-dir", default=os.environ.get("RELAY_REPORTS", "/opt/reports"),
                   help="published eval reports, served PUBLIC at /reports/")

    r = sub.add_parser("robot")
    r.add_argument("--relay", required=True)
    r.add_argument("--robot", required=True)
    r.add_argument("--token", default="", help="legacy; robot agents no longer require tokens")
    r.add_argument("--allow", type=int, nargs="+", default=[5599, 8089])
    r.add_argument("--can", default="can0")
    r.add_argument("--serve-port", type=int, default=5599)
    r.add_argument("--camera-url", default="http://127.0.0.1:8089/0")
    r.add_argument("--serve-cmd", default=os.path.expanduser(
        "~/i2rt/.venv/bin/python ~/blupe-evals/YAM_control/yam_real_serve.py --channel can0"))
    r.add_argument("--turnoff-cmd", default=os.path.expanduser(
        "~/i2rt/.venv/bin/python ~/blupe-evals/YAM_control/turn_off.py --channel can0"))
    r.add_argument("--camera-cmd", default=os.path.expanduser(
        "~/miniforge3/envs/xr/bin/python ~/blupe-evals/YAM_control/camera_relay.py --devices 0 2"))

    o = sub.add_parser("operator")
    o.add_argument("--relay", required=True)
    o.add_argument("--robot", required=True)
    o.add_argument("--token", required=True)
    o.add_argument("--map", nargs="+", default=["5599:15599", "8089:18089"])

    args = ap.parse_args()
    asyncio.run({"serve": role_serve, "robot": role_robot, "operator": role_operator}[args.role](args))


if __name__ == "__main__":
    main()
