"""Test whether the IK can rotate the EE (gripper) in place about each axis, from the home
pose. Position held; only the target orientation changes. Reports orientation error reached
and which joint moved most. Run in `xr` env from ~/blupe-evals.
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
hid = c.mj_model.key("home").id
task = c.effector_task["right_hand"]


def ori_deg(a, b):
    return float(np.degrees(2 * np.arccos(min(1.0, abs(np.dot(a, b))))))


def run(label, axis, ang=0.6):
    mujoco.mj_resetDataKeyframe(c.mj_model, c.mj_data, hid)
    mujoco.mj_forward(c.mj_model, c.mj_data)
    c._update_robot_state()
    x0, q0 = c._get_link_pose(EE)
    R0 = tf.quaternion_matrix(q0)
    q_start = c.mj_data.qpos.copy()
    Rd = tf.rotation_matrix(ang, axis)            # rotation in the EE's own frame
    T = R0 @ Rd
    T[:3, 3] = x0
    q_tgt = tf.quaternion_from_matrix(T)
    for _ in range(1500):
        task.T_world_frame = T
        c._update_ik()
        c._send_command()
        mujoco.mj_step(c.mj_model, c.mj_data)
    x1, q1 = c._get_link_pose(EE)
    oe = ori_deg(np.array(q_tgt), np.array(q1))
    pe = float(np.linalg.norm(np.array(x1) - np.array(x0)))
    dq = c.mj_data.qpos - q_start
    moved = ", ".join(f"j{i+1}{dq[i]:+.2f}" for i in range(6) if abs(dq[i]) > 0.02)
    print(f"{label} ({np.degrees(ang):.0f}deg about EE-{['x','y','z'][int(np.argmax(np.abs(axis)))]}): "
          f"ori err {oe:.1f} deg, pos err {pe:.3f} m -> {'PASS' if oe < 5 else 'FAIL'} | moved: {moved}")


run("ROLL  (gripper spin)", [0, 0, 1])
run("PITCH", [0, 1, 0])
run("YAW",   [1, 0, 0])
