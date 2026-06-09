"""Same YAM teleop, but logs controller pose + gripper pose every frame to ~/yam_teleop.log
so we can see what actually happens during a real (headset-on) session. Diagnostic only.
"""

import numpy as np
import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

EE = "link6"


def main(scale_factor: float = 1.0):
    cfg = {"right_hand": {"link_name": EE, "pose_source": "right_controller",
                          "control_trigger": "right_grip", "vis_target": "right_target",
                          "control_mode": "pose"}}
    c = MujocoTeleopController(xml_path="assets/yam/scene.xml",
                              robot_urdf_path="assets/yam/yam.urdf",
                              manipulator_config=cfg, scale_factor=scale_factor, visualize_placo=False)
    jt = c.solver.add_joints_task()
    jt.set_joints({j: 0.0 for j in c.placo_robot.joint_names()})
    jt.configure("reg", "soft", 1e-4)

    log = open("/home/andrew/yam_teleop.log", "w")
    log.write("grip,cx,cy,cz,cqx,cqy,cqz,cqw,eqw,eqx,eqy,eqz\n")
    _orig = c._send_command

    def _send():
        p = c.xr_client.get_pose_by_name("right_controller")  # [x,y,z, qx,qy,qz,qw]
        g = c.xr_client.get_key_value_by_name("right_grip")
        eq = c.mj_data.body(EE).xquat  # [w,x,y,z]
        log.write(",".join(f"{v:.3f}" for v in
                  [g, p[0], p[1], p[2], p[3], p[4], p[5], p[6], eq[0], eq[1], eq[2], eq[3]]) + "\n")
        log.flush()
        _orig()

    c._send_command = _send
    c.run()


if __name__ == "__main__":
    tyro.cli(main)
