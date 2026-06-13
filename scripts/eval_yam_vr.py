"""Eval state machine + LIVE headset video in ONE process.

Runs the same gate as eval_yam_states (TELEOP / POLICY / GO_HOME / QUIT, menu + A/X/B/Y),
but headless: instead of an on-screen MuJoCo window it streams video to the Quest. The VIEW
menu option toggles what the headset shows: the robot's real cameras or the live sim render.

Two video transports (--video):
  stereo (default)  ZEDMINI flow — the QUEST DIALS US (stereo_sender.py listens on :13579),
                    then we stream a double-wide H.264 canvas back: both cameras side by
                    side at full size, or a wide sim render, with ONE HUD bar across the
                    whole frame. View it in the app's FLAT mode (B toggles flat<->3D; the
                    canvas is a dashboard, not an eye pair). Quest: Camera panel -> source
                    ZEDMINI -> Listen -> this host's IP. Survives the Quest re-opening the
                    panel (fresh OPEN_CAMERA each time) — unlike the mono LISTEN port.
  mono              legacy Remote Vision flow — we dial the Quest's LISTEN port (:12345).
                    Quest: Remote Vision -> camera-source IP = this host -> LISTEN.

Verified coexistence: Remote Vision video + controller tracking run at once.
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")  # EGL flaky on this Jetson; GLFW+DISPLAY works

import json
import socket
import struct
import threading
import time
from fractions import Fraction
from typing import Literal, Optional

import mujoco
import numpy as np
import tyro

try:
    import cv2
except ImportError:
    cv2 = None

SHORTCUT_KEY = {"TELEOP": "A", "POLICY": "X", "GO_HOME": "B", "QUIT": "Y", "CONNECT": "stick",
                "RELAUNCH": "menu"}
WAYPOINTS = "scripts/policies/waypoints.json"   # MARK A/B capture; pick_place.py prefers these
# Teleop, policy, and go-home all share ONE joint-speed cap: E.MAX_VEL (eval_yam_states.py,
# imported below). Change it there and every motion path here is bounded by the same number.


def draw_hud(rgb, menu, highlight, state, connect_arm, link_status=None, rec_label=None,
             judging=None, input_stale=False, grip=None):
    """Overlay the menu/state HUD onto the RGB frame (so the buttons show in the headset).
    Menu = compact 2x4 grid centered below the cameras (frame edges are hard to read in VR).
    Active state = green fill; stick cursor = yellow border. judging = policy-verdict modal
    (0=SUCCESS, 1=FAIL highlighted). rgb is RGB uint8."""
    if cv2 is None:
        return rgb
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    cv2.putText(rgb, f"STATE: {state}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 230, 90), 2)
    if input_stale:               # controller data stopped (doze / panel disconnect / bridge)
        msg = "NO CONTROLLER INPUT - Network panel: tap popup IP + Send ON"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        bx = (w - tw) // 2
        cv2.rectangle(rgb, (bx - 14, 40), (bx + tw + 14, 40 + th + 22), (200, 40, 40), -1)
        cv2.putText(rgb, msg, (bx, 40 + th + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)
    if rec_label:                                       # trial recording indicator
        cv2.circle(rgb, (22, 54), 8, (255, 60, 60), -1)
        cv2.putText(rgb, f"REC {rec_label}", (38, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 80, 80), 2)
    if grip is not None:          # sim has no gripper DOF -> show the commanded state instead
        col = (90, 220, 110) if grip == "OPEN" else (255, 110, 90)
        cv2.putText(rgb, f"GRIP: {grip}", (12, 88 if rec_label else 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    n = len(menu)
    cols = 4
    rows = (n + cols - 1) // cols
    bw, bh, pad = int(w * 0.13), 34, 8                  # compact centered block
    gw = cols * bw + (cols - 1) * pad
    x0g = (w - gw) // 2
    y0g = h - rows * (bh + pad) - 10
    ov = rgb.copy()                                     # translucent backing, grid-sized only
    cv2.rectangle(ov, (x0g - 12, y0g - 12), (x0g + gw + 12, h - 4), (25, 25, 30), -1)
    cv2.addWeighted(ov, 0.55, rgb, 0.45, 0, dst=rgb)
    for i, opt in enumerate(menu):
        r_, c_ = divmod(i, cols)
        x0 = x0g + c_ * (bw + pad); y0 = y0g + r_ * (bh + pad)
        x1, y1 = x0 + bw, y0 + bh
        active = (opt == state) or (opt == "CONNECT" and connect_arm)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (40, 150, 70) if active else (55, 55, 60), -1)
        if i == highlight and judging is None:
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (255, 230, 0), 3)
        key = SHORTCUT_KEY.get(opt, "")
        cv2.putText(rgb, f"{opt} ({key})" if key else opt, (x0 + 6, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    if judging is not None:                             # policy verdict modal, dead center
        mw, mh = int(w * 0.36), 116
        mx, my = (w - mw) // 2, (h - mh) // 2
        cv2.rectangle(rgb, (mx, my), (mx + mw, my + mh), (20, 22, 28), -1)
        cv2.rectangle(rgb, (mx, my), (mx + mw, my + mh), (255, 230, 0), 2)
        cv2.putText(rgb, "POLICY RUN: success?", (mx + 16, my + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        obw = (mw - 48) // 2
        for i, (lab, col) in enumerate([("SUCCESS", (40, 170, 80)), ("FAIL", (190, 60, 50))]):
            ox, oy = mx + 16 + i * (obw + 16), my + 52
            cv2.rectangle(rgb, (ox, oy), (ox + obw, oy + 44), col, -1)
            if i == judging:
                cv2.rectangle(rgb, (ox, oy), (ox + obw, oy + 44), (255, 230, 0), 3)
            cv2.putText(rgb, lab, (ox + 16, oy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
    if connect_arm and link_status != "connected":      # direct CONNECT on but not linked
        msg = "CONNECTING..." if link_status == "connecting" else "DIRECT LINK OFF"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(rgb, (w - tw - 26, 8), (w - 6, 8 + th + 18), (200, 40, 40), -1)
        cv2.putText(rgb, msg, (w - tw - 16, 8 + th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return rgb


def letterbox(frame, scale):
    """Shrink the picture into a black border for the HEADSET-bound frame only: the Quest
    app fills its panel edge-to-edge (RawImageRectSize in video_source.yml), so scaling the
    content is how we 'move the screen back' without touching the app. Recordings stay full."""
    if scale >= 1.0 or cv2 is None:
        return frame
    h, w = frame.shape[:2]
    sw, sh = max(2, int(w * scale)) & ~1, max(2, int(h * scale)) & ~1
    out = np.zeros_like(frame)
    x0, y0 = (w - sw) // 2, (h - sh) // 2
    out[y0:y0 + sh, x0:x0 + sw] = cv2.resize(frame, (sw, sh))
    return out
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

import arms                  # arm registry: --arm picks models/EE/dof (scripts/arms.py)
import eval_yam_states as E  # reuse SimRobot/SafeRobot/Console/load_run/_sample_run + constants
import xrobotoolkit_sdk as _xr_sdk        # input-age probe (the shim in this dir)
from stereo_sender import LatencyStats

LAT = LatencyStats()   # data-flow stage timings (input/IK/loop/command/camera); [lat] every 2 s


class VideoStreamer:
    """Background thread: H.264-encode the latest submitted RGB frame and send it to the Quest
    as [4-byte big-endian length][Annex-B]. Reconnects on drop; fresh encoder per connection
    (leads with SPS/PPS + IDR). Proven config: 960x540, baseline, ultrafast/zerolatency."""

    def __init__(self, ip, port, w, h, fps):
        self.ip, self.port, self.w, self.h, self.fps = ip, port, w, h, fps
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, rgb):
        with self._lock:
            self._latest = rgb

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _connect(self):
        while not self._stop.is_set():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            try:
                s.connect((self.ip, self.port))
                s.settimeout(5.0)   # send timeout: a stalled Quest -> reconnect, not blocked forever
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print("[stream] connected", flush=True)
                return s
            except OSError:
                s.close()
                time.sleep(0.5)
        return None

    def _run(self):
        import av
        pts = 0
        print(f"[stream] waiting for Quest LISTEN at {self.ip}:{self.port} ...", flush=True)
        while not self._stop.is_set():
            sock = self._connect()
            if sock is None:
                return
            enc = av.CodecContext.create("libx264", "w")
            enc.width, enc.height, enc.pix_fmt = self.w, self.h, "yuv420p"
            enc.time_base = Fraction(1, self.fps)
            enc.options = {"preset": "ultrafast", "tune": "zerolatency",
                           "g": "15", "profile": "baseline"}
            period = 1.0 / self.fps
            try:
                while not self._stop.is_set():
                    t0 = time.monotonic()
                    with self._lock:
                        rgb = self._latest
                    if rgb is None:
                        time.sleep(0.01)
                        continue
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame.pts = pts
                    pts += 1
                    for pkt in enc.encode(frame):
                        b = bytes(pkt)
                        sock.sendall(struct.pack(">I", len(b)) + b)
                    dt = period - (time.monotonic() - t0)
                    if dt > 0:
                        time.sleep(dt)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"[stream] dropped ({e}); reconnecting", flush=True)
                try:
                    sock.close()
                except OSError:
                    pass
                time.sleep(0.5)


class RobotLink:
    """CONNECT: stream the sim's joint command to yam_real_serve.py so the real arm follows.
    Background thread; newline-JSON. Reads the no-jump handshake (start_joints) on connect."""

    def __init__(self, host="127.0.0.1", port=5599):
        self.host, self.port = host, port
        self.status = "idle"           # idle | connecting | connected | error
        self.start_joints = None
        self._latest = None
        self._gripper = None
        self._t_target = 0.0
        self._lock = threading.Lock()
        self._wlock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._file = None

    def connect_async(self):
        self.status = "connecting"
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_target(self, q, gripper=None):
        with self._lock:
            self._latest = [float(x) for x in q]
            self._gripper = None if gripper is None else float(gripper)
            self._t_target = time.monotonic()

    def _read_acks(self, f):
        """Drain {"ack": t} echoes from the serve: arm_rtt = Mac-clock time from socket write
        to applied-on-the-robot-and-echoed-back. An old serve sends nothing -> blocks harmlessly."""
        try:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line.decode()).get("ack")
                if t is not None:
                    LAT.note("arm_rtt", time.monotonic() - float(t))
        except (OSError, ValueError):
            pass                                      # link closed under us; thread exits

    def _run(self):
        s = None
        try:
            s = socket.create_connection((self.host, self.port), timeout=4.0)
            f = s.makefile("rwb")
            self._file = f
            line = f.readline()
            if not line:
                raise ConnectionError("no handshake from serve")
            self.start_joints = json.loads(line.decode()).get("start_joints")
            self.status = "connected"
            print(f"[connect] real arm linked; start_joints={np.round(self.start_joints, 3)}", flush=True)
            threading.Thread(target=self._read_acks, args=(f,), daemon=True).start()
            while not self._stop.is_set():
                with self._lock:
                    latest = self._latest
                    grip = self._gripper
                    t_target = self._t_target
                if latest is not None:
                    now = time.monotonic()
                    LAT.note("cmd_queue", now - t_target)       # set_target -> socket write
                    msg = {"q": latest, "t": round(now, 4)}
                    if grip is not None:
                        msg["g"] = grip
                    with self._wlock:
                        f.write((json.dumps(msg) + "\n").encode())
                        f.flush()
                    LAT.note("cmd_write", time.monotonic() - now)
                self._stop.wait(0.02)                 # ~50 Hz
        except Exception as e:
            self.status = "error"
            print(f"[connect] link error: {e}", flush=True)
        finally:
            self._file = None
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass

    def close(self, shutdown=False):
        if shutdown and self._file is not None:
            try:
                with self._wlock:
                    self._file.write((json.dumps({"shutdown": True}) + "\n").encode())
                    self._file.flush()
            except OSError:
                pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.status = "idle"


class CameraGrabber:
    """Grab webcam frames in a thread so the control loop never blocks on cv2.read. Composites
    one OR several cameras side-by-side into a single (width x height) RGB frame; None if no
    camera opened (-> sim-render fallback). Sequential reads => ~camera_fps / n_cameras."""

    def __init__(self, devices, width, height, fps=30):
        self.width, self.height = width, height
        self._latest = None
        self._eyes = []                # full-size per-camera RGB frames (stereo: one per eye)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.caps = []
        self.devs = []                 # parsed device per cap, for reopen-on-death
        if cv2 is None:
            print("[camera] cv2 unavailable -> sim render", flush=True)
            return
        for dev in devices:
            dev = int(dev) if str(dev).lstrip("-").isdigit() else str(dev)
            if isinstance(dev, int) and dev < 0:
                continue
            cap = self._open(dev)
            self.caps.append(cap)              # keep the slot even if closed: the drain
            self.devs.append(dev)              # thread retries (robot relay may boot later)
            print(f"[camera] {dev} {'open' if cap.isOpened() else 'failed to open — will retry'}",
                  flush=True)
        if self.caps:
            print(f"[camera] {len(self.caps)} camera(s) -> headset (side-by-side)", flush=True)
        else:
            print("[camera] no cameras -> sim render", flush=True)

    def start(self):
        if self.caps:
            self._frames = [None] * len(self.caps)   # latest raw BGR per camera
            self._times = [0.0] * len(self.caps)     # monotonic arrival time per camera
            for i, cap in enumerate(self.caps):
                threading.Thread(target=self._read_one, args=(i, cap), daemon=True).start()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _open(self, dev):
        """Open one camera (local index or network URL) with our standard props."""
        cap = cv2.VideoCapture(dev)
        if isinstance(dev, int):                       # local device: force MJPG + size
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # latest frame, low latency
        return cap

    def _read_one(self, i, cap):
        """One blocking-read loop PER camera, draining to the newest frame. Sequential shared
        reads consume slower than the streams produce; TCP backpressure then queues frames
        server-side -> unbounded staleness (measured: 32 s old after 25 s of runtime). A
        dedicated drain per camera keeps consumption >= production, so 'latest' means NOW.
        A network stream that DIES returns EOF forever -> REOPEN after 3 s with no frames
        (relay channels die on network hops; NO SIGNAL must heal itself, not persist)."""
        last = time.monotonic()
        while not self._stop.is_set():
            if not cap.isOpened():                     # never opened / died: retry forever
                time.sleep(2.0)
                cap.release()
                cap = self._open(self.devs[i])
                self.caps[i] = cap                     # so stop() releases the live one
                if cap.isOpened():
                    print(f"[camera] {self.devs[i]} open (recovered)", flush=True)
                    last = time.monotonic()
                continue
            ok, bgr = cap.read()
            now = time.monotonic()
            if ok:
                with self._lock:
                    self._frames[i] = bgr
                    self._times[i] = now
                last = now
                continue
            if now - last < 3.0:
                time.sleep(0.05)
                continue
            print(f"[camera] {self.devs[i]}: no frames for {now - last:.0f}s -> reopening",
                  flush=True)
            cap.release()
            time.sleep(1.0)
            cap = self._open(self.devs[i])
            self.caps[i] = cap
            last = time.monotonic()                    # fresh grace period per attempt

    def _run(self):
        n = len(self.caps)
        cw = self.width // n
        period = 1.0 / 30
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._lock:
                frames, times = list(self._frames), list(self._times)
            if not any(times):             # nothing has EVER arrived -> keep latest()=None
                time.sleep(period)         # (sim-render fallback, e.g. --cameras none)
                continue
            LAT.note("cam_age", t0 - max(t for t in times if t))   # freshest frame's age
            full = []
            for i in range(n):
                f = np.zeros((self.height, self.width, 3), np.uint8) if frames[i] is None \
                    else cv2.resize(frames[i], (self.width, self.height))
                if t0 - times[i] > 1.5:        # dead/stalled stream: say so, never freeze silently
                    cv2.rectangle(f, (0, self.height // 2 - 28),
                                  (self.width, self.height // 2 + 28), (0, 0, 0), -1)
                    cv2.putText(f, f"NO SIGNAL cam{i} ({0 if not times[i] else t0 - times[i]:.0f}s)",
                                (self.width // 2 - 160, self.height // 2 + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                full.append(f)
            comp = full[0] if n == 1 else \
                np.hstack([cv2.resize(f, (cw, self.height)) for f in full])
            if comp.shape[1] != self.width or comp.shape[0] != self.height:
                comp = cv2.resize(comp, (self.width, self.height))   # rounding: cw*n != width
            rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            eyes = [np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in full]
            with self._lock:
                self._latest = np.ascontiguousarray(rgb)
                self._eyes = eyes
            dt = period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def latest_eyes(self):
        """(left, right) FULL-SIZE RGB pair for the stereo canvas — cameras[0] = left panel,
        cameras[1] = right panel. One camera -> same frame in both panels. None until the
        first frame (-> wide sim render fallback)."""
        with self._lock:
            if self._latest is None or not self._eyes:
                return None
            if len(self._eyes) == 1:
                return self._eyes[0].copy(), self._eyes[0].copy()
            return self._eyes[0].copy(), self._eyes[1].copy()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for cap in self.caps:
            cap.release()


class TrialRecorder:
    """Record eval trials for the report pipeline: the operator-view canvas (cameras/sim +
    HUD) -> runs/<session>/trial_NNN/video.mp4 + meta.json. Threaded latest-frame encoder
    (same pattern as the video senders) so the control loop never blocks on encoding.
    One instance per session; start()/stop() per trial; judge later with eval_report.py."""

    def __init__(self, session_dir, fps=30):
        self.session_dir = session_dir
        self.fps = fps
        self.trial_dir = None          # set while recording
        self._latest, self._seq = None, 0
        self._lock = threading.Lock()
        self._worker = None
        self._stop_evt = None
        self._meta = None
        self._t0 = 0.0

    @property
    def recording(self):
        return self.trial_dir is not None

    @property
    def label(self):
        return os.path.basename(self.trial_dir) if self.trial_dir else None

    def start(self):
        nums = [int(d.split("_")[-1]) for d in os.listdir(self.session_dir)
                if d.startswith("trial_") and d.split("_")[-1].isdigit()]
        n = max(nums, default=0) + 1
        self.trial_dir = os.path.join(self.session_dir, f"trial_{n:03d}")
        os.makedirs(self.trial_dir, exist_ok=True)
        self._meta = {"trial": n, "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "events": [], "result": None, "failed_stage": None,
                      "score": None, "notes": ""}
        self._t0 = time.monotonic()
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        print(f"[trial] REC {self.label}", flush=True)

    def event(self, kind, value):
        """Timestamped breadcrumb into meta.json (state changes, gripper, connect...)."""
        if self._meta is not None:
            self._meta["events"].append([round(time.monotonic() - self._t0, 2), kind, value])

    def set_result(self, result):
        """In-VR verdict (policy modal): lands in meta.json when the trial saves."""
        if self._meta is not None:
            self._meta["result"] = result

    def submit(self, rgb):
        with self._lock:
            self._latest = rgb
            self._seq += 1

    def _run(self):
        import av
        stop, path = self._stop_evt, os.path.join(self.trial_dir, "video.mp4")
        out = stream = None
        sent, pts = -1, 0
        while not stop.is_set():
            with self._lock:
                rgb, seq = self._latest, self._seq
            if rgb is None or seq == sent:             # only NEW frames -> wall-clock-ish rate
                time.sleep(0.005)
                continue
            sent = seq
            if out is None:                            # lazy open: knows frame size now
                out = av.open(path, "w")
                stream = out.add_stream("h264", rate=self.fps)
                stream.width, stream.height = rgb.shape[1], rgb.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"preset": "ultrafast", "crf": "23"}
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = pts
            pts += 1
            for pkt in stream.encode(frame):
                out.mux(pkt)
        if out is not None:
            for pkt in stream.encode():                # flush
                out.mux(pkt)
            out.close()

    def stop(self):
        if not self.recording:
            return
        label = self.label
        self._stop_evt.set()
        self._worker.join(timeout=5.0)
        self._meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._meta["duration_s"] = round(time.monotonic() - self._t0, 2)
        with open(os.path.join(self.trial_dir, "meta.json"), "w") as f:
            json.dump(self._meta, f, indent=2)
        print(f"[trial] saved {label} ({self._meta['duration_s']}s)", flush=True)
        self.trial_dir = None
        self._meta = None
        with self._lock:
            self._latest, self._seq = None, 0


def main(quest_ip: str, port: int = 12345, policy: Optional[str] = None,
         width: int = 960, height: int = 540, fps: int = 30, scale_factor: float = 1.0,
         cameras: list[str] = ["0", "2"],
         video: Literal["stereo", "mono"] = "stereo",
         screen_scale: float = 0.85,   # <1 letterboxes the headset view ("move the screen back")
         task: Optional[str] = None, stages: list[str] = [],
         serve_host: str = "127.0.0.1", serve_port: int = 5599,
         preview_port: int = 8810,     # browser mirror of the headset canvas; 0 = off
         arm: str = "yam",             # arm standard from scripts/arms.py (yam | so101 | ...)
         direct_serve_control: bool = False):
    spec = arms.get(arm)
    E.N_ARM = spec.dof                 # eval_yam_states constants follow the spec: every
    E.EE = spec.ee_body                # E.N_ARM/E.EE reference below resolves per-arm
    E.MAX_VEL = spec.max_vel
    if arm != "yam" and policy:
        print(f"[eval] WARNING: bundled policies assume the YAM (assets/yam paths); "
              f"--policy with --arm {arm} is untested", flush=True)
    c = MujocoTeleopController(xml_path=spec.mjcf,
                              robot_urdf_path=spec.urdf,
                              manipulator_config=spec.manipulators, scale_factor=scale_factor,
                              visualize_placo=False)
    jt = c.solver.add_joints_task()
    jt.set_joints({j: 0.0 for j in c.placo_robot.joint_names()})
    jt.configure("reg", "soft", 1e-4)
    if spec.orientation_weight != 1.0:         # <6-DOF arms: position pinned, orientation
        for ee_name, ee_task in c.effector_task.items():   # best-effort via the wrist's DOFs
            ee_task.configure(ee_name, "soft", 1.0, spec.orientation_weight)

    m, d = c.mj_model, c.mj_data
    grip_aid = None                    # sim gripper actuator (arms that model one, e.g. so101)
    if spec.gripper_joint:
        grip_aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.gripper_joint)
        if grip_aid < 0:
            grip_aid = None
        else:
            print(f"[eval] sim gripper: right trigger TOGGLES actuator "
                  f"{spec.gripper_joint!r} (press = open<->close)", flush=True)
    home_qpos = m.key("home").qpos[:E.N_ARM].copy()
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)

    robot = E.SimRobot(c)
    console = E.Console(E.SafeRobot(robot))
    run_loop = E.load_run(policy) if policy else E._sample_run()
    print(f"[eval] policy: {policy}" if policy else "[eval] no --policy: built-in sample move")

    state, connect_arm, highlight, centered, prev_sel = "HOLD", False, 0, True, False
    prev = {b: False for b in E.SHORTCUT}
    menu = [m for m in E.MENU if direct_serve_control or m != "CONNECT"] + ["VIEW", "RELAUNCH", "MARK A"]
    # VIEW = cameras<->sim; RELAUNCH =
    # soft reset; MARK A/B = capture the CURRENT pose as a policy waypoint (teleop there first)

    recorder = None                              # --task arms trial recording (eval report v1)
    if task:
        session_dir = os.path.join("runs", f"{time.strftime('%Y-%m-%d')}_{task}")
        os.makedirs(session_dir, exist_ok=True)
        with open(os.path.join(session_dir, "task.json"), "w") as f:
            json.dump({"task": task, "stages": stages}, f, indent=2)
        recorder = TrialRecorder(session_dir, fps)
        print(f"[trial] session {session_dir} stages={stages or '(none)'} — "
              f"every POLICY run records one trial", flush=True)
    view_sim = False                     # False = camera composite (when available); True = sim render
    link = None                          # RobotLink when CONNECT is on (sim joints -> real arm)
    gripper_open = True                   # gripper toggle state (starts open); flips on each trigger press
    trig_prev = False                    # right index trigger as a button, for rising-edge detection
    judging = None                       # policy verdict modal: None | 0 (SUCCESS) | 1 (FAIL)
    centered_y = True                    # stick-y rising edge, for 2x4 grid row jumps
    head_prev, head_live_t = None, time.monotonic()   # input liveness: a worn headset's pose
    input_stale = False                  # always jitters; frozen >2 s = controller data dead

    def print_menu():
        row = "   ".join(f"[{o}]" if i == highlight else f" {o} " for i, o in enumerate(menu))
        print(f"[menu] {row}   (state={state})", flush=True)

    def enter(target):
        nonlocal state, judging
        if target == state:
            return
        if state == "POLICY":
            console.stop()
            judging = 0                       # every policy run ends with a verdict prompt
        c.active = {k: False for k in c.active}
        state = target
        print(f"[eval] state = {state}", flush=True)
        if recorder is not None and recorder.recording:
            recorder.event("state", state)
        if state == "POLICY":
            if recorder is not None:          # every policy run = one auto-recorded trial
                if recorder.recording:        # previous run never judged -> save it as-is
                    recorder.stop()
                judging = None                # a new run cancels any pending verdict modal
                recorder.start()
                recorder.event("policy", policy or "sample")
                recorder.event("connect", connect_arm)
                recorder.event("view", "sim" if view_sim else "cameras")
            console.start(run_loop)

    def choose(opt):
        nonlocal connect_arm, state, link, view_sim
        if opt == "VIEW":
            view_sim = not view_sim
            print(f"[eval] view -> {'SIM' if view_sim else 'CAMERAS'}", flush=True)
        elif opt == "CONNECT":
            if not direct_serve_control:
                print("[eval] CONNECT disabled; use fleet UI robot-side policy/serve control", flush=True)
                return
            connect_arm = not connect_arm
            if connect_arm:
                link = RobotLink(serve_host, serve_port)
                link.connect_async()
                print("[eval] CONNECT on -> streaming sim joints to real arm (yam_real_serve)", flush=True)
            else:
                if link is not None:
                    link.close(shutdown=False)
                    link = None
                print("[eval] CONNECT off -> sim only", flush=True)
        elif opt.startswith("MARK"):                  # capture current pose as a policy waypoint
            slot = "A" if opt.endswith("A") else "B"
            wp = {}
            if os.path.exists(WAYPOINTS):
                try:
                    wp = json.load(open(WAYPOINTS))
                except (json.JSONDecodeError, OSError):
                    wp = {}
            wp[slot] = {"q": [float(x) for x in d.qpos[:E.N_ARM]],
                        "marked": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(WAYPOINTS, "w") as f:
                json.dump(wp, f, indent=2)
            print(f"[mark] {slot} = {np.round(d.qpos[:E.N_ARM], 3)} -> {WAYPOINTS}", flush=True)
            menu[highlight] = "MARK B" if slot == "A" else "MARK A"   # alternate slots
        elif opt == "RELAUNCH":                       # soft reset: re-home, drop policy + clutch
            console.stop()
            c.active = {k: False for k in c.active}
            mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
            mujoco.mj_forward(m, d)
            state = "HOLD"
            print("[eval] RELAUNCH -> reset to home, HOLD", flush=True)
        else:
            enter(opt)

    renderer = mujoco.Renderer(m, height=height,            # stereo sends a double-wide canvas
                               width=2 * width if video == "stereo" else width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0.0, 0.3]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 120.0, -20.0

    if video == "stereo":
        from stereo_sender import StereoVisionServer
        streamer, stereo = None, StereoVisionServer()
        stereo.start()
        print("[eval] video: STEREO — Quest: Camera panel -> source ZEDMINI -> Listen -> this host's IP")
    else:
        streamer, stereo = VideoStreamer(quest_ip, port, width, height, fps), None
        streamer.start()
        print("[eval] video: MONO — Quest: Remote Vision -> source IP = this host -> LISTEN")
    grabber = CameraGrabber(cameras, width, height, fps)  # real camera(s) -> headset
    grabber.start()
    preview = None
    if preview_port:
        from preview_server import PreviewServer
        preview = PreviewServer(preview_port)              # browser mirror; a watcher alone
        preview.start()                                    # is enough to render the canvas
        if recorder is not None:
            def _new_session():
                """Fleet-UI "New report": close any in-flight trial, rotate to a fresh
                session dir so the next runs (and their report) stand alone."""
                if recorder.recording:
                    recorder.stop()
                sd = os.path.join(
                    "runs", f"{time.strftime('%Y-%m-%d')}_{task}_{time.strftime('%H%M%S')}")
                os.makedirs(sd, exist_ok=True)
                with open(os.path.join(sd, "task.json"), "w") as f:
                    json.dump({"task": task, "stages": stages}, f, indent=2)
                recorder.session_dir = sd
                print(f"[trial] NEW SESSION {sd}", flush=True)
                return sd
            preview.on_new_session = _new_session

    target_hz = 50.0
    loop_period = 1.0 / target_hz
    sim_steps = max(1, round(loop_period / m.opt.timestep))  # keep the sim ~real-time
    render_period = 1.0 / fps
    last_render = 0.0
    prev_t = time.monotonic()
    loops, last_dbg = 0, prev_t
    print("[eval] right stick = menu, click = select (shortcuts A/X/B/Y); headset = live view")
    print_menu()
    try:
        while not c._stop_event.is_set():
            t0 = time.monotonic()
            dt = min(max(t0 - prev_t, 1e-4), 0.1)   # real elapsed -> rate-independent caps
            prev_t = t0
            if state == "POLICY" and console._thread is not None \
                    and not console._thread.is_alive():    # run() returned on its own
                enter("HOLD")                              # -> enter() pops the verdict modal
            js = c.xr_client.get_joystick_state(E.NAV_STICK)
            jx, jy = float(js[0]), float(js[1])
            if judging is not None:                        # the verdict modal owns the stick
                if centered and abs(jx) > E.NAV_THRESH:
                    judging = 1 - judging
                    centered = False
                elif abs(jx) < E.NAV_DEADZONE:
                    centered = True
            else:
                if centered and abs(jx) > E.NAV_THRESH:
                    highlight = (highlight + (1 if jx > 0 else -1)) % len(menu)
                    centered = False
                    print_menu()
                elif abs(jx) < E.NAV_DEADZONE:
                    centered = True
                if centered_y and abs(jy) > E.NAV_THRESH:  # 2x4 grid: up/down jumps a row
                    highlight = (highlight + (-4 if jy > 0 else 4)) % len(menu)
                    centered_y = False
                    print_menu()
                elif abs(jy) < E.NAV_DEADZONE:
                    centered_y = True
            kb_select = False                              # browser keyboard (preview page)
            for k in (preview.take_keys() if preview is not None else ()):
                if k == "Enter":
                    kb_select = True
                elif k in ("ArrowLeft", "ArrowRight"):
                    if judging is not None:
                        judging = 1 - judging
                    else:
                        highlight = (highlight + (1 if k == "ArrowRight" else -1)) % len(menu)
                        print_menu()
                elif k in ("ArrowUp", "ArrowDown") and judging is None:
                    highlight = (highlight + (4 if k == "ArrowDown" else -4)) % len(menu)
                    print_menu()
                elif k.upper() in E.SHORTCUT:              # a/x/b/y = same shortcuts as the Quest
                    enter(E.SHORTCUT[k.upper()])
            sel = c.xr_client.get_button_state_by_name(E.SELECT_BTN)
            if (sel and not prev_sel) or kb_select:
                if judging is not None:                    # confirm the policy verdict
                    verdict = "success" if judging == 0 else "fail"
                    print(f"[eval] policy run -> {verdict.upper()}", flush=True)
                    if recorder is not None and recorder.recording:
                        recorder.event("policy_result", verdict)
                        recorder.set_result(verdict)
                        recorder.stop()                    # one trial per policy run
                    judging = None
                else:
                    choose(menu[highlight])
            prev_sel = sel
            loops += 1
            if t0 - last_dbg >= 2.0:
                a_btn = c.xr_client.get_button_state_by_name("A")
                print(f"[dbg] {loops / (t0 - last_dbg):.0f}Hz jx={jx:+.2f} A={a_btn} "
                      f"centered={centered} link={link.status if link is not None else '-'}", flush=True)
                rep = LAT.report()
                if rep:
                    print(f"[lat] {rep}", flush=True)
                loops, last_dbg = 0, t0
            age = _xr_sdk.get_input_age_s()
            if age > 0.0:                              # 0.0 = sdk/stub backend (no hop to measure)
                LAT.note("input_age", age)
                try:                                   # bridge mode: liveness via head-pose jitter
                    hp = tuple(_xr_sdk.get_headset_pose() or ())
                except Exception:
                    hp = ()
                if hp and hp != head_prev:
                    head_prev, head_live_t = hp, t0
                was_stale, input_stale = input_stale, (t0 - head_live_t) > 2.0
                if input_stale != was_stale:
                    print(f"[eval] controller input {'LOST (banner up)' if input_stale else 'back'}",
                          flush=True)
            for b, target in E.SHORTCUT.items():
                now = c.xr_client.get_button_state_by_name(b)
                if now and not prev[b]:
                    enter(target)
                prev[b] = now

            c._update_robot_state()
            if state == "TELEOP":
                prev_ctrl = d.ctrl[:E.N_ARM].copy()
                t_ik = time.monotonic()
                c._update_ik()
                c._update_gripper_target()
                c._update_mocap_target()
                c._send_command()                      # raw IK target -> d.ctrl
                LAT.note("ik", time.monotonic() - t_ik)
                if any(c.active.values()):              # any clutch held: drive to IK target,
                    step = E.MAX_VEL * dt               # rate-capped (bimanual: either hand)
                    d.ctrl[:E.N_ARM] = prev_ctrl + np.clip(d.ctrl[:E.N_ARM] - prev_ctrl, -step, step)
                else:                                   # no clutch: HOLD current pose (no move)
                    d.ctrl[:E.N_ARM] = prev_ctrl
            elif state in ("GO_HOME", "QUIT"):
                cur = d.ctrl[:E.N_ARM].copy()
                step = E.MAX_VEL * dt
                d.ctrl[:E.N_ARM] = cur + np.clip(home_qpos - cur, -step, step)
                if np.max(np.abs(home_qpos - d.qpos[:E.N_ARM])) < E.HOME_EPS:
                    if state == "QUIT":
                        robot.turn_off()           # park: home + motors off, but KEEP running
                    state = "HOLD"                 # stay alive so the headset can reconnect
                    print("[eval] state = HOLD (home reached)", flush=True)

            # Gripper = YAM pattern everywhere: trigger press TOGGLES open<->closed.
            trig = (c.xr_client.get_key_value_by_name("right_trigger") or 0.0) > 0.5
            if trig and not trig_prev:                           # rising edge -> toggle
                gripper_open = not gripper_open
                print(f"[eval] gripper -> {'OPEN' if gripper_open else 'CLOSED'}", flush=True)
                if recorder is not None and recorder.recording:
                    recorder.event("gripper", "open" if gripper_open else "closed")
            trig_prev = trig
            if grip_aid is not None:                             # sim-modeled gripper (so101/droid)
                glo, ghi = m.actuator_ctrlrange[grip_aid]
                open_v, closed_v = (ghi, glo) if spec.gripper_open_high else (glo, ghi)
                d.ctrl[grip_aid] = open_v if gripper_open else closed_v

            if connect_arm and link is not None and link.status == "connected":
                if state == "POLICY" and robot.gripper is not None:
                    gripper = robot.gripper                      # the policy owns the gripper
                else:
                    gripper = 1.0 if gripper_open else 0.0       # 1=open, 0=closed (serve walks it)
                link.set_target(d.ctrl[:E.N_ARM], gripper)       # arm joints + gripper -> real arm

            for _ in range(sim_steps):
                mujoco.mj_step(m, d)

            if t0 - last_render >= render_period:
                link_status = link.status if link is not None else None
                rec = recorder.label if recorder is not None and recorder.recording else None
                gv = robot.gripper if state == "POLICY" and robot.gripper is not None \
                    else (1.0 if gripper_open else 0.0)
                grip = "OPEN" if gv > 0.5 else "CLOSED"
                watching = preview is not None and preview.active
                if stereo is not None:
                    if stereo.status == "streaming" or rec or watching:  # headset/tape/browser
                        pair = None if view_sim else grabber.latest_eyes()  # VIEW toggle
                        if pair is not None:               # both cameras, full size, side by side
                            canvas = np.hstack(pair)
                        else:                              # fallback: one wide sim render
                            renderer.update_scene(d, cam)
                            canvas = renderer.render().copy()
                        frame = draw_hud(canvas, menu, highlight, state, connect_arm,
                                         link_status, rec, judging, input_stale, grip)
                        if stereo.status == "streaming":
                            stereo.submit(letterbox(frame, screen_scale))
                        if rec:
                            recorder.submit(frame)        # recordings stay full-frame
                        if watching:
                            preview.submit(frame)         # browser mirror, full-frame
                else:
                    cam_rgb = None if view_sim else grabber.latest()   # VIEW toggle; None -> sim render
                    if cam_rgb is None:                    # fallback: render the sim
                        renderer.update_scene(d, cam)
                        cam_rgb = renderer.render().copy()
                    frame = draw_hud(cam_rgb, menu, highlight, state, connect_arm,
                                     link_status, rec, judging, input_stale, grip)
                    streamer.submit(letterbox(frame, screen_scale))
                    if rec:
                        recorder.submit(frame)
                    if watching:
                        preview.submit(frame)
                last_render = t0

            LAT.note("loop_busy", time.monotonic() - t0)   # >20ms avg = loop can't hold 50Hz
            dt = loop_period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)
    finally:
        if recorder is not None:
            recorder.stop()                      # no-op unless mid-trial; saves what we have
        if link is not None:
            link.close(shutdown=True)            # exit -> tell the serve to torque-off the real arm
        grabber.stop()
        if streamer is not None:
            streamer.stop()
        if stereo is not None:
            stereo.stop()
        if preview is not None:
            preview.stop()
        console.stop()
        c.xr_client.close()


if __name__ == "__main__":
    tyro.cli(main)
