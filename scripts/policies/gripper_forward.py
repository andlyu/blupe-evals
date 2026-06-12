"""Demo policy: move the gripper to 25 cm ahead of the HOME position, then hold.

Loaded by the eval's POLICY state:
    --policy scripts/policies/gripper_forward.py:run

A policy is just `run(robot, stop)`: it commands JOINT targets via `robot.command(q)` and reads
`robot.read()` (joint_pos, ee_pos). It owns the arm while active; commands go through SafeRobot, so
they're velocity-clamped and the kill switch (stop()) cuts it instantly.

The target is anchored to the HOME pose (not wherever the arm currently is): we FK the home joint
config, translate the EE 25 cm forward (orientation held), and solve that with placo IK on the same
yam.urdf the eval uses -- so the goal is deterministic run-to-run. FORWARD is world +X; flip the
axis/sign if "forward" for your setup differs.
"""

import time

import mujoco
import numpy as np
import placo

EE = "grasp"
FORWARD = np.array([0.25, 0.0, 0.0])                # +25 cm in world X, measured from the HOME EE pose
URDF = "assets/yam/yam.urdf"
MJCF = "assets/yam/scene.xml"                        # home pose read from its `home` keyframe, never hardcoded


def run(robot, stop):
    r = placo.RobotWrapper(URDF)
    solver = placo.KinematicsSolver(r)
    solver.dt = 0.01
    nq = len(r.state.q)

    # HOME joints from the model's `home` keyframe -- single source of truth, the same one the eval reads
    home_q = mujoco.MjModel.from_xml_path(MJCF).key("home").qpos[:6].copy()

    # seed placo with the HOME joints so the target is anchored to the home pose (not the handoff pose)
    qq = r.state.q.copy()
    if nq > 6:                                  # defensive: only if a free-flyer is present
        qq[:7] = [0, 0, 0, 0, 0, 0, 1]
        solver.mask_fbase(True)
    qq[nq - 6:] = home_q
    r.state.q = qq
    r.update_kinematics()

    # target = HOME EE pose, translated 26 cm forward (orientation held)
    T = np.array(r.get_T_world_frame(EE))
    T[:3, 3] += FORWARD
    frame = solver.add_frame_task(EE, T)
    frame.configure("ee", "soft", 1.0)
    reg = solver.add_joints_task()
    reg.set_joints({j: float(home_q[i]) for i, j in enumerate(r.joint_names())})  # bias toward HOME
    reg.configure("reg", "soft", 1e-3)          # -> natural config near home, not a folded branch

    for _ in range(400):                        # converge to the goal config
        solver.solve(True)
        r.update_kinematics()
    q_goal = np.asarray(r.state.q[nq - 6:], dtype=float)
    reached = float(np.linalg.norm(np.array(r.get_T_world_frame(EE))[:3, 3] - T[:3, 3]))
    print(f"[policy] gripper -> 25cm ahead of home -> goal joints {np.round(q_goal, 2)} "
          f"(IK residual {reached:.3f} m)", flush=True)

    # hold the goal; SafeRobot ramps to it under the velocity cap, kill switch cuts it
    while not stop():
        robot.command(q_goal)
        time.sleep(0.02)
