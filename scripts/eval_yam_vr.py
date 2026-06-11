"""Eval state machine + LIVE headset video in ONE process.

Runs the same gate as eval_yam_states (TELEOP / POLICY / GO_HOME / QUIT, menu + A/X/B/Y),
but headless: instead of an on-screen MuJoCo window it renders the live sim offscreen and
streams it to the Quest's Remote Vision. So the headset shows the arm you are driving.

Verified coexistence: Remote Vision video + controller tracking run at once.

On the Quest: Remote Vision -> camera-source IP = this host -> LISTEN.  Then on the Orin:
    MUJOCO_GL=glfw DISPLAY=:0 python scripts/eval_yam_vr.py --quest-ip 192.168.0.30
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")  # EGL flaky on this Jetson; GLFW+DISPLAY works

import json
import socket
import struct
import threading
import time
from fractions import Fraction
from typing import Optional

import mujoco
import numpy as np
import tyro

try:
    import cv2
except ImportError:
    cv2 = None

SHORTCUT_KEY = {"TELEOP": "A", "POLICY": "X", "GO_HOME": "B", "QUIT": "Y", "CONNECT": "stick",
                "RELAUNCH": "menu"}
# Teleop, policy, and go-home all share ONE joint-speed cap: E.MAX_VEL (eval_yam_states.py,
# imported below). Change it there and every motion path here is bounded by the same number.


def draw_hud(rgb, menu, highlight, state, connect_arm, link_status=None):
    """Overlay the menu/state HUD onto the RGB frame (so the buttons show in the headset).
    Active state = green fill; stick cursor = yellow border. rgb is RGB uint8."""
    if cv2 is None:
        return rgb
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    bar = 64
    ov = rgb.copy()
    cv2.rectangle(ov, (0, h - bar), (w, h), (25, 25, 30), -1)
    cv2.addWeighted(ov, 0.55, rgb, 0.45, 0, dst=rgb)
    cv2.putText(rgb, f"STATE: {state}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 230, 90), 2)
    n = len(menu); pad = 8
    bw = (w - pad * (n + 1)) // n
    y0, y1 = h - bar + 8, h - 10
    for i, opt in enumerate(menu):
        x0 = pad + i * (bw + pad); x1 = x0 + bw
        active = (opt == state) or (opt == "CONNECT" and connect_arm)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (40, 150, 70) if active else (55, 55, 60), -1)
        if i == highlight:
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (255, 230, 0), 3)
        cv2.putText(rgb, f"{opt} ({SHORTCUT_KEY.get(opt, '')})", (x0 + 6, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    if connect_arm and link_status != "connected":      # CONNECT on but arm not actually linked
        msg = "CONNECTING..." if link_status == "connecting" else "ROBOT OFF - start serve"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(rgb, (w - tw - 26, 8), (w - 6, 8 + th + 18), (200, 40, 40), -1)
        cv2.putText(rgb, msg, (w - tw - 16, 8 + th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return rgb
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

import eval_yam_states as E  # reuse SimRobot/SafeRobot/Console/load_run/_sample_run + constants


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
            while not self._stop.is_set():
                with self._lock:
                    latest = self._latest
                    grip = self._gripper
                if latest is not None:
                    msg = {"q": latest} if grip is None else {"q": latest, "g": grip}
                    with self._wlock:
                        f.write((json.dumps(msg) + "\n").encode())
                        f.flush()
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
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.caps = []
        if cv2 is None:
            print("[camera] cv2 unavailable -> sim render", flush=True)
            return
        for dev in devices:
            dev = int(dev) if str(dev).lstrip("-").isdigit() else str(dev)
            if isinstance(dev, int):                   # local device index
                if dev < 0:
                    continue
                cap = cv2.VideoCapture(dev)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
            else:                                      # network stream (camera_relay.py URL)
                cap = cv2.VideoCapture(dev)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # latest frame, low latency
            if cap.isOpened():
                self.caps.append(cap)
                print(f"[camera] {dev} open", flush=True)
            else:
                print(f"[camera] {dev} failed to open", flush=True)
                cap.release()
        if self.caps:
            print(f"[camera] {len(self.caps)} camera(s) -> headset (side-by-side)", flush=True)
        else:
            print("[camera] no cameras -> sim render", flush=True)

    def start(self):
        if self.caps:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        n = len(self.caps)
        cw = self.width // n
        tiles = [np.zeros((self.height, cw, 3), np.uint8) for _ in self.caps]
        while not self._stop.is_set():
            for i, cap in enumerate(self.caps):
                ok, bgr = cap.read()
                if ok:
                    tiles[i] = cv2.resize(bgr, (cw, self.height))
            comp = tiles[0] if n == 1 else np.hstack(tiles)
            if comp.shape[1] != self.width or comp.shape[0] != self.height:
                comp = cv2.resize(comp, (self.width, self.height))   # rounding: cw*n != width
            rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest = np.ascontiguousarray(rgb)

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for cap in self.caps:
            cap.release()


def main(quest_ip: str, port: int = 12345, policy: Optional[str] = None,
         width: int = 960, height: int = 540, fps: int = 30, scale_factor: float = 1.0,
         cameras: list[str] = ["0", "2"],
         serve_host: str = "127.0.0.1", serve_port: int = 5599):
    cfg = {"right_hand": {"link_name": E.EE, "pose_source": "right_controller",
                          "control_trigger": "right_grip", "vis_target": "right_target",
                          "control_mode": "pose"}}
    c = MujocoTeleopController(xml_path="assets/yam/scene.xml",
                              robot_urdf_path="assets/yam/yam.urdf",
                              manipulator_config=cfg, scale_factor=scale_factor,
                              visualize_placo=False)
    jt = c.solver.add_joints_task()
    jt.set_joints({j: 0.0 for j in c.placo_robot.joint_names()})
    jt.configure("reg", "soft", 1e-4)

    m, d = c.mj_model, c.mj_data
    home_qpos = m.key("home").qpos[:E.N_ARM].copy()
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)

    robot = E.SimRobot(c)
    console = E.Console(E.SafeRobot(robot))
    run_loop = E.load_run(policy) if policy else E._sample_run()
    print(f"[eval] policy: {policy}" if policy else "[eval] no --policy: built-in sample move")

    state, connect_arm, highlight, centered, prev_sel = "HOLD", False, 0, True, False
    prev = {b: False for b in E.SHORTCUT}
    menu = list(E.MENU) + ["RELAUNCH"]   # RELAUNCH = soft reset after QUIT (re-home, ready)
    link = None                          # RobotLink when CONNECT is on (sim joints -> real arm)
    gripper_open = True                   # gripper toggle state (starts open); flips on each trigger press
    trig_prev = False                    # right index trigger as a button, for rising-edge detection

    def print_menu():
        row = "   ".join(f"[{o}]" if i == highlight else f" {o} " for i, o in enumerate(menu))
        print(f"[menu] {row}   (state={state})", flush=True)

    def enter(target):
        nonlocal state
        if target == state:
            return
        if state == "POLICY":
            console.stop()
        c.active = {k: False for k in c.active}
        state = target
        print(f"[eval] state = {state}", flush=True)
        if state == "POLICY":
            console.start(run_loop)

    def choose(opt):
        nonlocal connect_arm, state, link
        if opt == "CONNECT":
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
        elif opt == "RELAUNCH":                       # soft reset: re-home, drop policy + clutch
            console.stop()
            c.active = {k: False for k in c.active}
            mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
            mujoco.mj_forward(m, d)
            state = "HOLD"
            print("[eval] RELAUNCH -> reset to home, HOLD", flush=True)
        else:
            enter(opt)

    renderer = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0.0, 0.3]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 120.0, -20.0

    streamer = VideoStreamer(quest_ip, port, width, height, fps)
    streamer.start()
    grabber = CameraGrabber(cameras, width, height, fps)  # real camera(s) -> headset (side-by-side)
    grabber.start()

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
            jx = float(c.xr_client.get_joystick_state(E.NAV_STICK)[0])
            if centered and abs(jx) > E.NAV_THRESH:
                highlight = (highlight + (1 if jx > 0 else -1)) % len(menu)
                centered = False
                print_menu()
            elif abs(jx) < E.NAV_DEADZONE:
                centered = True
            sel = c.xr_client.get_button_state_by_name(E.SELECT_BTN)
            if sel and not prev_sel:
                choose(menu[highlight])
            prev_sel = sel
            loops += 1
            if t0 - last_dbg >= 2.0:
                a_btn = c.xr_client.get_button_state_by_name("A")
                print(f"[dbg] {loops / (t0 - last_dbg):.0f}Hz jx={jx:+.2f} A={a_btn} "
                      f"centered={centered} link={link.status if link is not None else '-'}", flush=True)
                loops, last_dbg = 0, t0
            for b, target in E.SHORTCUT.items():
                now = c.xr_client.get_button_state_by_name(b)
                if now and not prev[b]:
                    enter(target)
                prev[b] = now

            c._update_robot_state()
            if state == "TELEOP":
                prev_ctrl = d.ctrl[:E.N_ARM].copy()
                c._update_ik()
                c._update_gripper_target()
                c._update_mocap_target()
                c._send_command()                      # raw IK target -> d.ctrl
                if c.active.get("right_hand", False):   # gripping: drive to IK target, rate-capped
                    step = E.MAX_VEL * dt
                    d.ctrl[:E.N_ARM] = prev_ctrl + np.clip(d.ctrl[:E.N_ARM] - prev_ctrl, -step, step)
                else:                                   # not gripping: HOLD current pose (no move)
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

            if connect_arm and link is not None and link.status == "connected":
                trig = (c.xr_client.get_key_value_by_name("right_trigger") or 0.0) > 0.5  # index finger as a button
                if trig and not trig_prev:                       # rising edge -> toggle open<->closed
                    gripper_open = not gripper_open
                    print(f"[eval] gripper -> {'OPEN' if gripper_open else 'CLOSED'}", flush=True)
                trig_prev = trig
                gripper = 1.0 if gripper_open else 0.0           # 1=open, 0=closed (serve walks it, bounded)
                link.set_target(d.ctrl[:E.N_ARM], gripper)       # arm joints + gripper -> real arm

            for _ in range(sim_steps):
                mujoco.mj_step(m, d)

            if t0 - last_render >= render_period:
                cam_rgb = grabber.latest()                 # real camera frame (RGB), or None
                if cam_rgb is None:                        # fallback: render the sim
                    renderer.update_scene(d, cam)
                    cam_rgb = renderer.render().copy()
                frame = draw_hud(cam_rgb, menu, highlight, state, connect_arm,
                                 link.status if link is not None else None)
                streamer.submit(frame)
                last_render = t0

            dt = loop_period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)
    finally:
        if link is not None:
            link.close(shutdown=True)            # exit -> tell the serve to torque-off the real arm
        grabber.stop()
        streamer.stop()
        console.stop()
        c.xr_client.close()


if __name__ == "__main__":
    tyro.cli(main)
