"""Scripted pick-and-place policy:
move to A1 (10 cm above A) -> descend to A -> close gripper -> move to B -> open -> home.
The vertical approach keeps the gripper from sweeping the object sideways on the way in;
the grab happens only once A itself is reached.

Loaded by the eval's POLICY state:
    --policy scripts/policies/pick_place.py:run

WAYPOINTS — two sources, in priority order:
  1. MARKED (scripts/policies/waypoints.json): teleop the gripper where you want it, select
     MARK A in the eval menu; repeat for MARK B. The captured sim JOINT configs are replayed
     exactly (they came from your teleop IK — no offset guessing, no IK branch surprises).
  2. Fallback: A_OFFSET/B_OFFSET below, relative to the HOME EE pose, solved with placo IK
     (deterministic, but generic — mark real waypoints for a real task).

The policy RETURNS when the sequence ends -> the eval pops the SUCCESS/FAIL verdict modal
(each POLICY run = one recorded, judged trial). Gripper: 1=open, 0=closed (serve walks it).
"""

import json
import os
import time

import mujoco
import numpy as np
import placo

WAYPOINTS = "scripts/policies/waypoints.json"

EE = "grasp"
URDF = "assets/yam/yam.urdf"
MJCF = "assets/yam/scene.xml"                # home pose from its `home` keyframe, never hardcoded
A_OFFSET = np.array([0.25, 0.00, 0.00])      # pick: 25 cm ahead of home EE (known reachable)
B_OFFSET = np.array([0.25, 0.20, 0.00])      # place: same depth, 20 cm to the side
OPEN, CLOSED = 1.0, 0.0
APPROACH_LIFT = 0.10                         # m above A for the A1 approach point
A1_MAX_RESID = 0.05                          # m; worse IK than this -> skip A1, go straight to A
EPS = 0.03                                   # rad, per-joint arrival threshold
LEG_TIMEOUT = 12.0                           # s per motion leg (velocity-capped moves are slow)
GRIP_DWELL = 1.2                             # s holding still while the serve walks the gripper


def _make_solver(home_q):
    """placo IK anchored at HOME (same recipe as gripper_forward.py, reusable for N targets).
    Returns (solve, fk): solve(offset, seed_q) -> (q_goal[6], residual_m), with offset
    relative to the HOME EE position; fk(q6) -> world EE position (so a marked joint
    config can be turned into an offset, e.g. A1 = fk(q_a) - fk(home) + lift)."""
    r = placo.RobotWrapper(URDF)
    solver = placo.KinematicsSolver(r)
    solver.dt = 0.01
    nq = len(r.state.q)

    def _seed(q6):
        qq = r.state.q.copy()
        if nq > 6:                           # defensive: only if a free-flyer is present
            qq[:7] = [0, 0, 0, 0, 0, 0, 1]
        qq[nq - 6:] = q6
        r.state.q = qq
        r.update_kinematics()

    if nq > 6:
        solver.mask_fbase(True)
    _seed(home_q)
    T_home = np.array(r.get_T_world_frame(EE))
    frame = solver.add_frame_task(EE, T_home)
    frame.configure("ee", "soft", 1.0)
    reg = solver.add_joints_task()
    reg.set_joints({j: float(home_q[i]) for i, j in enumerate(r.joint_names())})
    reg.configure("reg", "soft", 1e-3)       # bias toward natural home-like configs

    def solve(offset, seed_q):
        _seed(seed_q)
        T = T_home.copy()
        T[:3, 3] += offset
        frame.T_world_frame = T
        for _ in range(400):
            solver.solve(True)
            r.update_kinematics()
        q = np.asarray(r.state.q[nq - 6:], dtype=float)
        resid = float(np.linalg.norm(np.array(r.get_T_world_frame(EE))[:3, 3] - T[:3, 3]))
        return q, resid

    def fk(q6):
        _seed(q6)
        return np.array(r.get_T_world_frame(EE))[:3, 3].copy()

    return solve, fk


def _goto(robot, stop, q, gripper, label):
    """Command q (+gripper) until arrival or timeout; SafeRobot does the velocity ramp."""
    t0 = time.monotonic()
    while not stop():
        robot.command(q, gripper)
        err = np.max(np.abs(np.asarray(robot.read().joint_pos, dtype=float)[:6] - q))
        if err < EPS:
            return True
        if time.monotonic() - t0 > LEG_TIMEOUT:
            print(f"[policy] {label}: TIMEOUT (residual {err:.3f} rad) — continuing", flush=True)
            return False
        time.sleep(0.02)
    return False


def _dwell(robot, stop, q, gripper, secs):
    """Hold q while commanding the gripper for a fixed dwell (the serve walks it gradually)."""
    t0 = time.monotonic()
    while not stop() and time.monotonic() - t0 < secs:
        robot.command(q, gripper)
        time.sleep(0.02)


def _load_waypoints():
    """Marked joint configs from MARK A/B in the eval, or None."""
    if not os.path.exists(WAYPOINTS):
        return None
    try:
        wp = json.load(open(WAYPOINTS))
        return (np.asarray(wp["A"]["q"], dtype=float)[:6],
                np.asarray(wp["B"]["q"], dtype=float)[:6])
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as e:
        print(f"[policy] waypoints.json unusable ({e}) -> IK fallback", flush=True)
        return None


def run(robot, stop):
    home_q = mujoco.MjModel.from_xml_path(MJCF).key("home").qpos[:6].copy()
    marked = _load_waypoints()
    if marked is not None:
        q_a, q_b = marked
        # A1 = the marked A lifted APPROACH_LIFT in world z: FK the marked joints, raise the
        # target, IK back seeded FROM q_a (stays on the operator's arm branch).
        solve, fk = _make_solver(home_q)
        off_a1 = fk(q_a) - fk(home_q)
        off_a1[2] += APPROACH_LIFT
        q_a1, res_a1 = solve(off_a1, q_a)
        print(f"[policy] pick_place: MARKED waypoints A={np.round(q_a, 2)} "
              f"B={np.round(q_b, 2)} (A1 res {res_a1:.3f} m)", flush=True)
    else:
        solve, fk = _make_solver(home_q)
        q_a1, res_a1 = solve(A_OFFSET + [0.0, 0.0, APPROACH_LIFT], home_q)
        q_a, res_a = solve(A_OFFSET, q_a1)   # seed the descent from A1 -> continuous branch
        q_b, res_b = solve(B_OFFSET, q_a)
        print(f"[policy] pick_place: no marked waypoints -> IK offsets "
              f"(A1 res {res_a1:.3f} m, A res {res_a:.3f} m, B res {res_b:.3f} m). "
              f"Teleop + MARK A/B to set real ones.", flush=True)
    if res_a1 > A1_MAX_RESID:                # unreachable approach: don't visit a wrong pose
        print(f"[policy] A1 unreachable (res {res_a1:.3f} m) -> direct to A", flush=True)
        q_a1 = None

    if q_a1 is not None:
        print(f"[policy] -> A1 (approach, {APPROACH_LIFT * 100:.0f} cm above A)", flush=True)
        _goto(robot, stop, q_a1, OPEN, "to A1")
    print("[policy] -> A (descend, gripper open)", flush=True)
    if not _goto(robot, stop, q_a, OPEN, "to A"):
        print("[policy] A not reached — grabbing anyway (timeout)", flush=True)
    print("[policy] close gripper", flush=True)
    _dwell(robot, stop, q_a, CLOSED, GRIP_DWELL)
    print("[policy] -> B (carrying)", flush=True)
    _goto(robot, stop, q_b, CLOSED, "to B")
    print("[policy] open gripper", flush=True)
    _dwell(robot, stop, q_b, OPEN, GRIP_DWELL)
    print("[policy] -> home", flush=True)
    _goto(robot, stop, home_q, OPEN, "home")
    print("[policy] pick_place done", flush=True)
