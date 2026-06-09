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


def draw_hud(rgb, menu, highlight, state, connect_arm):
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
                s.settimeout(None)
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


def main(quest_ip: str, port: int = 12345, policy: Optional[str] = None,
         width: int = 960, height: int = 540, fps: int = 30, scale_factor: float = 1.0):
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
        nonlocal connect_arm, state
        if opt == "CONNECT":
            connect_arm = not connect_arm
            print(f"[eval] connect-to-arm = {connect_arm} (real backend is M2; still sim)", flush=True)
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

    target_hz = 50.0
    loop_period = 1.0 / target_hz
    sim_steps = max(1, round(loop_period / m.opt.timestep))  # keep the sim ~real-time
    render_period = 1.0 / fps
    last_render = 0.0
    prev_t = time.monotonic()
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

            for _ in range(sim_steps):
                mujoco.mj_step(m, d)

            if t0 - last_render >= render_period:
                renderer.update_scene(d, cam)
                frame = draw_hud(renderer.render().copy(), menu, highlight, state, connect_arm)
                streamer.submit(frame)
                last_render = t0

            dt = loop_period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)
    finally:
        streamer.stop()
        console.stop()
        c.xr_client.close()


if __name__ == "__main__":
    tyro.cli(main)
