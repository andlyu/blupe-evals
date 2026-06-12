"""Browser mirror of the headset canvas — iterate on the VR layout without wearing the Quest.

The eval submits the SAME post-draw_hud frame it sends to the headset; this serves it as
multipart MJPEG (renders in a plain <img>, every browser, no JS):

    http://<eval-host>:8810/        page wrapping the live stream
    http://<eval-host>:8810/stream  the raw MJPEG (embed anywhere, e.g. the fleet UI)

`active` is True while any browser is connected — the eval uses it to render the canvas
even when no Quest is streaming and no trial is recording (headset-free iteration).

Transport rules inherited from camera_relay/stereo_sender (don't regress them):
each frame sent ONCE (seq gate), small SO_SNDBUF + drop-on-backpressure so a dozing
browser tab can never queue a stale backlog — slow consumers get FEWER frames, never
OLDER frames.
"""

import json
import os
import select
import socket
import subprocess
import sys
import threading
import time

try:
    import cv2
except ImportError:
    cv2 = None

_PAGE = b"""HTTP/1.1 200 OK\r
Content-Type: text/html\r
Connection: close\r
\r
<!doctype html><title>blupe headset mirror</title>
<body style="margin:0;background:#111;display:flex;flex-direction:column;align-items:center">
<img src="/stream" style="max-width:100vw;max-height:94vh">
<p style="color:#888;font:13px monospace">
<button id="rec" onclick="rec()" style="font:13px monospace;padding:4px 12px;cursor:pointer;
border-radius:5px;border:1px solid #666;background:#222;color:#ddd">&#9210; REC</button>
&nbsp; arrows = move &nbsp; Enter = select &nbsp;
a/x/b/y = TELEOP/POLICY/GO_HOME/QUIT &nbsp; (same canvas the Quest sees)</p>
<script>
const KEYS = ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter','a','x','b','y'];
document.addEventListener('keydown', e => {
  if (e.repeat || !KEYS.includes(e.key)) return;
  e.preventDefault();
  fetch('/key?k=' + encodeURIComponent(e.key));
});
async function rec(){
  const r = await (await fetch('/rec')).json();
  const b = document.getElementById('rec');
  if (r.err) { b.textContent = r.err; return; }
  b.textContent = r.recording ? '\\u23F9 STOP \\u25CF recording' : '\\u23FA REC';
  b.style.background = r.recording ? '#a22' : '#222';
  if (r.saved) b.title = 'saved: ' + r.saved;
}
</script>
</body>"""

_STREAM_HDR = (b"HTTP/1.1 200 OK\r\n"
               b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
               b"Cache-Control: no-store\r\nConnection: close\r\n\r\n")


class SessionTape:
    """Continuous recording of the operator view -> session.mp4: the full "what I was
    seeing while teleoperating" track, independent of the per-trial recorder. Threaded
    latest-frame encoder (never blocks the eval loop); wall-clock pts = true playback
    speed however the render rate varies."""

    def __init__(self, fps=30):
        self.fps = fps
        self.path = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def recording(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, path):
        self.stop()
        self.path = path
        self._latest = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, rgb):
        with self._lock:
            self._latest = rgb

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self):
        import av
        from fractions import Fraction
        container = stream = t0 = None
        last = None
        period = 1.0 / self.fps
        while not self._stop.is_set():
            time.sleep(period)
            with self._lock:
                rgb = self._latest
            if rgb is None or rgb is last:             # nothing new this tick
                continue
            last = rgb
            if container is None:
                h, w = rgb.shape[:2]
                container = av.open(self.path, "w")
                stream = container.add_stream("h264", rate=self.fps)
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "23"}
                stream.codec_context.time_base = Fraction(1, 1000)
                t0 = time.monotonic()
                print(f"[preview] session tape -> {self.path}", flush=True)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = int((time.monotonic() - t0) * 1000)
            for pkt in stream.encode(frame):
                container.mux(pkt)
        if container is not None:
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            print(f"[preview] session tape saved {self.path}", flush=True)


class PreviewServer:
    """submit(rgb) the headset frame; browsers watch it at http://:port/."""

    def __init__(self, port=8810, quality=80):
        self.port, self.quality = port, quality
        self._jpeg = None
        self._seq = 0
        self._clients = 0              # stream watchers (drives `active`)
        self._keys = []                # browser keydowns, drained by the eval loop
        self.tape = SessionTape()      # continuous operator-view recording (report sessions)
        self.on_new_session = None     # set by the eval: rotate trial session, return its dir
        self._session_dir = None       # current report session (set by /report/new)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def active(self):
        return self._clients > 0 or self.tape.recording   # a running tape forces rendering

    def take_keys(self):
        """Drain keydown events sent by the page (e.g. 'ArrowLeft', 'Enter', 'a')."""
        with self._lock:
            keys, self._keys = self._keys, []
        return keys

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.tape.stop()               # finalize session.mp4 if a report was left open

    def submit(self, rgb):
        """Encode ONCE per frame, only while someone is watching."""
        if self.tape.recording:
            self.tape.submit(rgb)
        if cv2 is None or self._clients == 0:
            return
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def _rec_toggle(self, conn):
        """Page REC button: ad-hoc recording of the mirror to runs/demo/. Refuses to touch
        a tape owned by a report session (that one is stopped by Finish report)."""
        try:
            if self.tape.recording:
                if self._session_dir and self.tape.path and \
                        self.tape.path.startswith(self._session_dir):
                    out = {"err": "report session is recording (use Finish report)"}
                else:
                    path = self.tape.path
                    self.tape.stop()
                    out = {"ok": True, "recording": False, "saved": path}
            else:
                os.makedirs("runs/demo", exist_ok=True)
                self.tape.start(time.strftime("runs/demo/mirror_%Y%m%d_%H%M%S.mp4"))
                out = {"ok": True, "recording": True}
        except Exception as e:
            out = {"ok": False, "err": str(e)}
        body = json.dumps(out).encode()
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() +
                     b"\r\nConnection: close\r\n\r\n" + body)
        print(f"[preview] rec: {out}", flush=True)

    def _report(self, conn, action):
        """Report-session control (also reachable from the fleet UI via the relay):
        new    = rotate the eval to a fresh trial session + start the session tape
        finish = stop the tape + render report.html for the session
        status = current session dir / tape state / trial count"""
        out = {"ok": False}
        try:
            if action == "new":
                if self.on_new_session is None:
                    out["err"] = "eval not in --task mode (no recorder)"
                else:
                    sd = self.on_new_session()
                    self._session_dir = sd
                    self.tape.start(os.path.join(sd, "session.mp4"))
                    out = {"ok": True, "session": sd, "tape": self.tape.path}
            elif action == "finish":
                sd = self._session_dir
                if sd is None:
                    out["err"] = "no report session started"
                else:
                    self.tape.stop()
                    r = subprocess.run(
                        [sys.executable, os.path.join(os.path.dirname(__file__),
                                                      "eval_report.py"),
                         "render", "--session", sd],
                        capture_output=True, text=True, timeout=120)
                    trials = sorted(d for d in os.listdir(sd) if d.startswith("trial_"))
                    out = {"ok": r.returncode == 0, "session": sd, "trials": len(trials),
                           "report": os.path.join(sd, "report.html"),
                           "session_video": os.path.join(sd, "session.mp4"),
                           "render": (r.stdout + r.stderr).strip()[-400:]}
            elif action == "status":
                sd = self._session_dir
                trials = sorted(d for d in os.listdir(sd)
                                if d.startswith("trial_")) if sd else []
                out = {"ok": True, "session": sd, "recording": self.tape.recording,
                       "trials": len(trials)}
            else:
                out["err"] = f"unknown action {action!r}"
        except Exception as e:
            out = {"ok": False, "err": str(e)}
        body = json.dumps(out).encode()
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() +
                     b"\r\nConnection: close\r\n\r\n" + body)
        print(f"[preview] report/{action}: {out}", flush=True)

    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", self.port))
        except OSError as e:
            print(f"[preview] disabled: cannot bind :{self.port} ({e})", flush=True)
            return
        srv.listen(4)
        srv.settimeout(1.0)
        print(f"[preview] headset mirror at http://0.0.0.0:{self.port}/", flush=True)
        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._client, args=(conn, addr[0]), daemon=True).start()
        srv.close()

    def _client(self, conn, peer):
        counted = False
        try:
            conn.settimeout(5.0)
            req = conn.recv(2048).decode(errors="replace")
            path = req.split(" ")[1] if " " in req else "/"
            if path.startswith("/report/"):
                self._report(conn, path.split("?")[0].split("/")[2])
                return
            if path.startswith("/rec"):
                self._rec_toggle(conn)
                return
            if path.startswith("/key"):
                key = path.split("k=", 1)[1].split("&")[0] if "k=" in path else ""
                key = key.replace("%20", " ")
                if key:
                    with self._lock:
                        self._keys.append(key)
                conn.sendall(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                return
            if not path.startswith("/stream"):
                conn.sendall(_PAGE)
                return
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
            conn.sendall(_STREAM_HDR)
            with self._lock:
                self._clients += 1
                counted = True
            print(f"[preview] {peer} watching ({self._clients})", flush=True)
            sent_seq = -1
            while not self._stop.is_set():
                with self._cond:
                    if self._seq == sent_seq:
                        self._cond.wait(timeout=1.0)
                    jpeg, seq = self._jpeg, self._seq
                if jpeg is None or seq == sent_seq:
                    continue
                sent_seq = seq
                _, writable, _ = select.select([], [conn], [], 0)
                if not writable:                   # stalled tab -> drop, never backlog
                    time.sleep(0.03)
                    continue
                conn.sendall(b"--frame\r\nContent-Type: image/jpeg\r\n"
                             b"Content-Length: " + str(len(jpeg)).encode() +
                             b"\r\n\r\n" + jpeg + b"\r\n")
        except (OSError, ValueError, IndexError):
            pass
        finally:
            if counted:
                with self._lock:
                    self._clients -= 1
                print(f"[preview] {peer} left ({self._clients})", flush=True)
            try:
                conn.close()
            except OSError:
                pass
