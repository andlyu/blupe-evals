"""Reproduce 'arm straight, rotate wrist' and check IK behavior + whether placo can weight
position over orientation. Run in `xr` env from ~/blupe-evals."""

import mujoco
import numpy as np
import placo
from meshcat import transformations as tf
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

print("FrameTask.configure doc:", placo.FrameTask.configure.__doc__)

EE = "link6"
cfg = {"right_hand": {"link_name": EE, "pose_source": "right_controller",
                      "control_trigger": "right_grip", "vis_target": "right_target",
                      "control_mode": "pose"}}
c = MujocoTeleopController(xml_path="assets/yam/scene.xml", robot_urdf_path="assets/yam/yam.urdf",
                          manipulator_config=cfg, scale_factor=1.0, visualize_placo=False)
jt = c.solver.add_joints_task()
jt.set_joints({j: 0.0 for j in c.placo_robot.joint_names()})
jt.configure("reg", "soft", 1e-4)
task = c.effector_task["right_hand"]


def ori_deg(a, b):
    return float(np.degrees(2 * np.arccos(min(1.0, abs(np.dot(a, b))))))


def trial(qstart, axis, ang, label):
    c.mj_data.qpos[:6] = qstart
    mujoco.mj_forward(c.mj_model, c.mj_data)
    c._update_robot_state()
    x0, q0 = c._get_link_pose(EE)
    T = tf.quaternion_matrix(q0) @ tf.rotation_matrix(ang, axis)
    T[:3, 3] = x0
    qt = tf.quaternion_from_matrix(T)
    for _ in range(1500):
        task.T_world_frame = T
        c._update_ik()
        c._send_command()
        mujoco.mj_step(c.mj_model, c.mj_data)
    x1, q1 = c._get_link_pose(EE)
    print(f"  {label}: ori_err {ori_deg(qt, q1):.1f} deg, POS DRIFT {np.linalg.norm(np.array(x1)-np.array(x0)):.3f} m")


for name, q in [("HOME   ", [0, 1.0, 1.0, 0, 0, 0]),
                ("STRAIGHT", [0, 1.57, 0.0, 0, 0, 0]),
                ("EXTENDED", [0, 0.3, 0.3, 0, 0, 0])]:
    print(name)
    trial(q, [0, 0, 1], 0.6, "roll  ")
    trial(q, [1, 0, 0], 0.6, "yaw   ")
