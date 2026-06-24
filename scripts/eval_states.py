"""Generic eval state machine switched by Quest controller buttons.

One driver at a time (the Console gate ported from blupe-eval-console).

Selection is by the RIGHT thumbstick: tilt L/R to move the highlight across the menu
[ TELEOP | POLICY | GO_HOME | CONNECT | QUIT ], press the stick (right_axis_click) to
select. A/X/B/Y are optional direct shortcuts.

  TELEOP   grip-clutch IK teleop (ours; full 6-DOF, XRoboToolkit)
  POLICY   run the user's external run(robot, stop) loop, gated by SafeRobot
  GO_HOME  ease to the home pose under a rate clamp, then hold
  QUIT     go home, turn the arm off, exit
  CONNECT  toggle connect-to-arm. In this on-monitor variant, that only flips intent
           and logs; the arm stays the MuJoCo sim either way.

For operator-side policies, pass:
  --policy /path/to/your_policy.py:run   (a `run(robot, stop)` callable).
With no --policy a built-in sample move is used so X is visible in sim.

The same state machine + gate can drive different arm profiles by swapping adapters.

Run on an operator desktop:  DISPLAY=:0 python scripts/eval_states.py
"""

import importlib.util
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

N_ARM = 6
EE = "grasp"                       # default tool frame body; IK target + FK readout
MAX_VEL = 0.6                      # rad/s -- THE single joint-speed cap: teleop, policy, AND go-home
HOME_EPS = 0.01                    # rad: "reached home" tolerance

# Right thumbstick drives a menu: tilt L/R to move the highlight, press (click) to select.
# SDK getters verified in common/xr_client.py:
#   get_joystick_state("left"|"right") -> [x, y]; get_button_state_by_name("right_axis_click") -> bool.
MENU = ["TELEOP", "POLICY", "GO_HOME", "CONNECT", "QUIT"]
NAV_STICK = "right"
SELECT_BTN = "right_axis_click"
NAV_THRESH = 0.6                   # tilt past this = step the highlight
NAV_DEADZONE = 0.3                 # return inside this to re-arm a step (one tilt = one step)
SHORTCUT = {"A": "TELEOP", "X": "POLICY", "B": "GO_HOME", "Y": "QUIT"}  # optional direct buttons


# --------------------------------------------------------------------------- #
# Robot seam (ported from blupe-eval-console: interface.py + console.py)
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    joint_pos: np.ndarray          # (N_ARM,) rad
    ee_pos: np.ndarray             # (3,) base-frame xyz, m
    gripper: float = 0.0           # sim has no gripper DOF (linear_4310 is real/M2)


class SimRobot:
    """Robot adapter over the XRoboToolkit controller's MuJoCo sim. read() = joint
    qpos + EE FK; command() writes the position-actuator targets (d.ctrl). No sim
    gripper DOF to move, but the last commanded gripper is KEPT (self.gripper) so the
    eval's real-arm link and HUD can honor a policy's gripper commands."""

    def __init__(self, c: MujocoTeleopController):
        self.m, self.d = c.mj_model, c.mj_data
        self.gripper = None

    def read(self) -> Observation:
        try:
            ee = self.d.body(EE).xpos.copy()
        except Exception:
            ee = np.zeros(3)
        return Observation(joint_pos=self.d.qpos[:N_ARM].copy(), ee_pos=ee)

    def command(self, joint_pos, gripper=None) -> None:
        self.d.ctrl[:N_ARM] = np.asarray(joint_pos, dtype=float)[:N_ARM]
        if gripper is not None:
            self.gripper = float(gripper)

    def turn_off(self) -> None:
        print("[eval] (sim) arm disarmed -- real i2rt motor-off is M2")


class SafeRobot:
    """Wraps a Robot: every command is velocity-clamped (a rate-independent rad/s
    cap) and gated by an `armed` flag -- the kill switch. When disarmed, commands
    are dropped so a runaway policy is stopped even if it ignores stop()."""

    def __init__(self, robot, max_vel: float = MAX_VEL):
        self.robot = robot
        self.max_vel = max_vel
        self._armed = False
        self._last_t = None
        self._last = np.asarray(robot.read().joint_pos, dtype=float)[:N_ARM].copy()

    def arm(self) -> None:
        # re-seed from the current pose so the first command delta is ~0 (no jump)
        self._last = np.asarray(self.robot.read().joint_pos, dtype=float)[:N_ARM].copy()
        self._last_t = None
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def read(self) -> Observation:
        return self.robot.read()

    def command(self, joint_pos, gripper=None) -> None:
        if not self._armed:
            return                                       # kill switch: command dropped
        now = time.monotonic()
        dt = (now - self._last_t) if self._last_t is not None else 0.005
        dt = min(max(dt, 1e-4), 0.1)                     # bound dt (no huge step after a stall)
        self._last_t = now
        step = self.max_vel * dt                         # |dq| <= max_vel * dt
        desired = np.asarray(joint_pos, dtype=float)[:N_ARM]
        self._last = self._last + np.clip(desired - self._last, -step, step)
        self.robot.command(self._last, gripper)


class Console:
    """Owns who drives the arm. start(run) hands off to the policy loop in a daemon
    thread; stop() disarms instantly (the kill switch), signals the loop, joins it."""

    def __init__(self, safe: SafeRobot):
        self.safe = safe
        self._stop = threading.Event()
        self._thread = None

    def start(self, run) -> None:
        self.stop()                                      # exclusive: one driver at a time
        self._stop.clear()
        self.safe.arm()
        self._thread = threading.Thread(target=self._wrap, args=(run,), daemon=True)
        self._thread.start()

    def _wrap(self, run) -> None:
        try:
            run(self.safe, self._stop.is_set)
        finally:
            self.safe.disarm()

    def stop(self) -> None:
        self._stop.set()
        self.safe.disarm()                               # immediate, independent of the loop
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


# --------------------------------------------------------------------------- #
# Policy loading (external; lives on the Jetson, not in this repo)
# --------------------------------------------------------------------------- #
def load_run(spec: str) -> Callable:
    """spec = '/path/to/policy.py:run' or 'module.path:run' -> the run callable."""
    path, _, attr = spec.partition(":")
    attr = attr or "run"
    if path.endswith(".py"):
        p = Path(path).expanduser()
        s = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        import importlib as _il
        mod = _il.import_module(path)
    return getattr(mod, attr)


def _sample_run(turn_deg: float = 20.0, gain: float = 0.25, rate: float = 50.0):
    """Built-in stand-in for the user's policy: turn joint1 by turn_deg and hold.
    Commands through SafeRobot (rate-clamped, killable) -- safe in sim and on real."""
    def run(robot, stop):
        start = np.asarray(robot.read().joint_pos, dtype=float)
        tgt = start.copy()
        tgt[0] += np.deg2rad(turn_deg)
        while not stop():
            q = np.asarray(robot.read().joint_pos, dtype=float)
            robot.command(q + gain * (tgt - q))
            time.sleep(1.0 / rate)
    return run


# --------------------------------------------------------------------------- #
def main(policy: Optional[str] = None, scale_factor: float = 1.0,
         visualize_placo: bool = False):
    cfg = {"right_hand": {"link_name": EE, "pose_source": "right_controller",
                          "control_trigger": "right_grip", "vis_target": "right_target",
                          "control_mode": "pose"}}
    c = MujocoTeleopController(xml_path="assets/yam/scene.xml",
                              robot_urdf_path="assets/yam/yam.urdf",
                              manipulator_config=cfg, scale_factor=scale_factor,
                              visualize_placo=visualize_placo)
    jt = c.solver.add_joints_task()
    jt.set_joints({j: 0.0 for j in c.placo_robot.joint_names()})
    jt.configure("reg", "soft", 1e-4)

    m, d = c.mj_model, c.mj_data
    home_qpos = m.key("home").qpos[:N_ARM].copy()
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)

    robot = SimRobot(c)
    console = Console(SafeRobot(robot))
    run_loop = load_run(policy) if policy else _sample_run()
    print(f"[eval] policy: {policy}" if policy
          else "[eval] no --policy: using built-in sample move (turn joint1)")

    state = "HOLD"
    connect_arm = False
    highlight = 0
    centered = True
    prev_sel = False
    prev = {b: False for b in SHORTCUT}

    def print_menu():
        row = "   ".join(f"[{o}]" if i == highlight else f" {o} " for i, o in enumerate(MENU))
        print(f"[menu] {row}   (state={state})")

    def enter(target):
        nonlocal state
        if target == state:
            return
        if state == "POLICY":                            # leaving policy: cut + reclaim
            console.stop()
        c.active = {k: False for k in c.active}           # drop teleop clutch on any switch
        state = target
        print(f"[eval] state = {state}")
        if state == "POLICY":
            console.start(run_loop)

    def choose(opt):
        nonlocal connect_arm
        if opt == "CONNECT":
            connect_arm = not connect_arm
            print(f"[eval] connect-to-arm = {connect_arm} "
                  f"({'real backend is M2 -- still running sim' if connect_arm else 'sim'})")
        else:
            enter(opt)

    print("[eval] right stick = move menu, click = select   (shortcuts: A/X/B/Y)")
    print_menu()
    prev_t = time.monotonic()

    with mj_viewer.launch_passive(m, d) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 120, -20, 1.3
        viewer.cam.lookat = [0.3, 0, 0.3]
        while not c._stop_event.is_set() and viewer.is_running():
            t0 = time.monotonic()
            dt = min(max(t0 - prev_t, 1e-4), 0.1)         # real elapsed -> rate-independent vel caps
            prev_t = t0
            # --- right-stick menu: tilt L/R moves the highlight, click selects ---
            jx = float(c.xr_client.get_joystick_state(NAV_STICK)[0])
            if centered and abs(jx) > NAV_THRESH:
                highlight = (highlight + (1 if jx > 0 else -1)) % len(MENU)
                centered = False
                print_menu()
            elif abs(jx) < NAV_DEADZONE:
                centered = True
            sel = c.xr_client.get_button_state_by_name(SELECT_BTN)
            if sel and not prev_sel:
                choose(MENU[highlight])
            prev_sel = sel

            # --- optional direct shortcuts (A/X/B/Y) ---
            for b, target in SHORTCUT.items():
                now = c.xr_client.get_button_state_by_name(b)
                if now and not prev[b]:
                    enter(target)
                prev[b] = now

            c._update_robot_state()                       # keep placo synced for clean teleop re-entry

            if state == "TELEOP":
                prev_ctrl = d.ctrl[:N_ARM].copy()
                c._update_ik()
                c._update_gripper_target()
                c._update_mocap_target()
                c._send_command()                         # raw IK targets -> d.ctrl
                step = MAX_VEL * dt                        # cap joint speed (rad/s, rate-independent)
                d.ctrl[:N_ARM] = prev_ctrl + np.clip(d.ctrl[:N_ARM] - prev_ctrl, -step, step)
            elif state in ("GO_HOME", "QUIT"):
                cur = d.ctrl[:N_ARM].copy()
                step = MAX_VEL * dt
                d.ctrl[:N_ARM] = cur + np.clip(home_qpos - cur, -step, step)
                if np.max(np.abs(home_qpos - d.qpos[:N_ARM])) < HOME_EPS:
                    if state == "QUIT":
                        robot.turn_off()
                        break
                    state = "HOLD"
                    print("[eval] state = HOLD (home reached)")
            # POLICY: the console thread drives via SafeRobot. HOLD: ctrl is held.

            mujoco.mj_step(m, d)
            viewer.sync()

    console.stop()
    c.xr_client.close()


if __name__ == "__main__":
    tyro.cli(main)
