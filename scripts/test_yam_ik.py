"""Headless correctness test for YAM teleop IK (no Quest). Run in `xr` env from ~/blupe-evals.

  A. placo's EE frame == the MuJoCo sim's EE pose (the consistency invariant).
  B. HOLD: goal == current pose => the arm must not move (no jump on clutch).
  C. TRACK: moving the goal moves the EE there.
"""

import mujoco
import numpy as np
from meshcat import transformations as tf
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

EE = "link6"
cfg = {"right_hand": {"link_name": EE, "pose_source": "right_controller",
                      "control_trigger": "right_grip", "vis_target": "right_target",
                      "control_mode": "pose"}}
c = MujocoTeleopController(xml_path="assets/yam/scene.xml", robot_urdf_path="assets/yam/yam.urdf",
                          manipulator_config=cfg, scale_factor=1.0, visualize_placo=False)
jt = c.solver.add_joints_task()
jt.set_joints({j: 0.0 for j in c.placo_robot.joint_names()})
jt.configure("reg", "soft", 1e-4)
mujoco.mj_resetDataKeyframe(c.mj_model, c.mj_data, c.mj_model.key("home").id)
mujoco.mj_forward(c.mj_model, c.mj_data)
c._update_robot_state()


def ori_deg(a, b):
    return float(np.degrees(2 * np.arccos(min(1.0, abs(np.dot(a, b))))))


xyz, quat = c._get_link_pose(EE)
Rmj = tf.quaternion_matrix(quat)[:3, :3]
Rpl = np.array(c.placo_robot.get_T_world_frame(EE))[:3, :3]
a = np.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(Rmj.T @ Rpl) - 1) / 2))))
print(f"[A] consistency: {a:.3f} deg -> {'PASS' if a < 0.5 else 'FAIL'}")

T = tf.quaternion_matrix(quat); T[:3, 3] = xyz
task = c.effector_task["right_hand"]; task.T_world_frame = T
for _ in range(800):
    task.T_world_frame = T
    c._update_ik()
    c._send_command()
    mujoco.mj_step(c.mj_model, c.mj_data)
x1, q1 = c._get_link_pose(EE)
pdrift, odrift = float(np.linalg.norm(np.array(x1) - np.array(xyz))), ori_deg(np.array(quat), np.array(q1))
print(f"[B] HOLD goal=current: EE pos {pdrift:.4f} m, ori {odrift:.2f} deg "
      f"-> {'PASS' if pdrift < 5e-3 and odrift < 1.0 else 'FAIL'}")

goal = np.array(x1) + np.array([0.0, 0.0, 0.08]); T2 = T.copy(); T2[:3, 3] = goal
for _ in range(1500):
    task.T_world_frame = T2
    c._update_ik()
    c._send_command()
    mujoco.mj_step(c.mj_model, c.mj_data)
x2, _ = c._get_link_pose(EE)
terr = float(np.linalg.norm(goal - np.array(x2)))
print(f"[C] TRACK +8cm z: EE error {terr:.4f} m -> {'PASS' if terr < 0.02 else 'FAIL'}")
