"""Mac <-> Quest bridge launcher — switch the running arm from a browser.

A tiny supervisor that owns the Mac-to-Quest bridge process: pick an arm from
scripts/arms.py and it kills the current bridge and boots the chosen one. The operator
console stays at :8810 (the new bridge re-binds it). Sim arms run serve-less; real
profiles can use relay camera and serve tunnels.

Run:  XR_INPUT=bridge .venv/bin/python scripts/eval_launcher.py     # from the repo root
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms

EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mac_quest_bridge.py")
LOG = "/tmp/eval_live.log"


def _read_token(path="/tmp/relay_token"):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def _yam_profile():
    token = os.environ.get("RELAY_OPERATOR_TOKEN") or _read_token()
    return ["--serve-host", "127.0.0.1", "--serve-port", "15599",
            "--cameras", "http://127.0.0.1:18089/0", "http://127.0.0.1:18089/2",
            "--task", "red-plate-demo", "--stages", "reach", "grasp", "lift", "place",
            "--relay-policy-url", os.environ.get("RELAY_POLICY_URL", "http://35.203.190.87"),
            "--relay-policy-token", token,
            "--relay-policy-robot", os.environ.get("RELAY_POLICY_ROBOT", "yam-1"),
            "--relay-policy", os.environ.get("RELAY_POLICY", "pick_place")]


# Per-arm launch profiles. Sim arms: no cameras, dead-end serve port (CONNECT is a no-op).
# yam = the production profile through the relay operator tunnel.
PROFILES = {
    "yam": _yam_profile,
}
SIM_ARGS = ["--cameras", "none", "--serve-port", "5596"]


def _policy_from_args(args):
    for flag in ("--relay-policy", "--policy"):
        try:
            value = args[args.index(flag) + 1]
            return value if flag == "--policy" else f"relay:{value or 'default'}"
        except (ValueError, IndexError):
            pass
    return None


def _profile_args(arm):
    profile = PROFILES.get(arm)
    if profile is None:
        return SIM_ARGS
    return profile() if callable(profile) else profile


def _profile_policy(arm):
    return _policy_from_args(_profile_args(arm))


class Launcher:
    def __init__(self, quest_ip):
        self.quest_ip = quest_ip
        self.proc = None
        self.arm = None
        self.lock = threading.Lock()

    def launch(self, arm):
        spec = arms.ARMS.get(arm)
        if spec is None:
            return {"ok": False, "err": f"unknown arm {arm!r}"}
        if getattr(spec, "status", "ready") != "ready":
            return {"ok": False, "err": f"arm {arm!r} is {spec.status}"}
        with self.lock:
            self._stop_locked()
            for pat in ("mac_quest_bridge[.]py", "eval_yam_vr[.]py"):
                subprocess.run(["pkill", "-f", pat], capture_output=True)
            time.sleep(1.5)                       # let :8810/:13579 free up
            argv = [sys.executable, EVAL, "--quest-ip", self.quest_ip, "--arm", arm]
            argv += _profile_args(arm)
            env = dict(os.environ, XR_INPUT=os.environ.get("XR_INPUT", "bridge"))
            with open(LOG, "ab") as log:
                self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True, env=env)
            self.arm = arm
            policy = _profile_policy(arm)
            print(f"[launcher] {arm} -> pid {self.proc.pid}; policy={policy or 'none'}",
                  flush=True)
            return {"ok": True, "arm": arm, "pid": self.proc.pid,
                    "policy": policy, "policy_attached": policy is not None,
                    "console": "http://this-host:8810/"}

    def _stop_locked(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        self.proc = None

    def status(self):
        alive = self.proc is not None and self.proc.poll() is None
        current_arm = self.arm if alive else None
        current_policy = _profile_policy(current_arm) if current_arm else None
        return {"ok": True, "arm": self.arm if alive else None, "running": alive,
                "policy": current_policy, "policy_attached": current_policy is not None,
                "arms": {n: {"status": s.status, "dof": s.dof, "notes": s.notes,
                             "policy": _profile_policy(n),
                             "policy_attached": _profile_policy(n) is not None}
                         for n, s in arms.ARMS.items()}}


PAGE = """<!doctype html><title>blupe eval launcher</title>
<body style="font:15px -apple-system;background:#16181c;color:#dde;margin:32px">
<h2>Eval launcher</h2>
<p id="cur" style="color:#8ad"></p>
<div id="cards"></div>
<p><a href="http://HOST:8810/" style="color:#8ad">operator console (:8810)</a> —
the same page after every switch; reload it once the new arm is up (~8 s).</p>
<script>
async function refresh(){
  const s = await (await fetch('/status')).json();
  document.getElementById('cur').textContent =
    s.running ? `running: ${s.arm} · policy: ${s.policy || 'none'}` : 'nothing running';
  document.getElementById('cards').innerHTML = Object.entries(s.arms).map(([n, a]) =>
    `<div style="background:#1e2128;border-radius:10px;padding:12px 16px;margin:10px 0;max-width:640px">
     <b>${n}</b> <span style="color:#889">dof ${a.dof}${a.status !== 'ready' ? ' · ' + a.status : ''}</span>
     <button style="float:right;padding:6px 14px;cursor:pointer" ${a.status !== 'ready' ? 'disabled' : ''}
       onclick="launch('${n}', this)">${s.arm === n && s.running ? 'Relaunch' : 'Launch'}</button>
     <div style="color:#8ad;font-size:13px;margin-top:6px">policy: ${a.policy || 'none'}</div>
     <div style="color:#99a;font-size:13px;margin-top:6px">${a.notes}</div></div>`).join('');
}
async function launch(n, btn){
  btn.disabled = true; btn.textContent = 'launching…';
  const r = await (await fetch('/launch?arm=' + n)).json();
  if (!r.ok) alert(r.err);
  setTimeout(refresh, 9000); setTimeout(refresh, 2000);
}
refresh(); setInterval(refresh, 10000);
</script></body>"""


def main(port: int = 8809, quest_ip: str = "192.168.0.30"):
    launcher = Launcher(quest_ip)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = PAGE.replace("HOST", self.headers.get("Host", "").split(":")[0]
                                    or "localhost").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/launch":
                arm = parse_qs(url.query).get("arm", [""])[0]
                self._json(launcher.launch(arm))
            elif url.path == "/status":
                self._json(launcher.status())
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"[launcher] http://0.0.0.0:{port}/  (arms: {', '.join(arms.ARMS)})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
