"""Teleop a single YAM arm in MuJoCo with an XRoboToolkit Quest controller.

Quest controller -> XRoboToolkit PC Service -> xrobotoolkit_sdk -> placo IK -> MuJoCo.

Follows the stock framework process (teleop_details.md): the MuJoCo MJCF (assets/yam/yam.xml,
the i2rt ground-truth model) and the placo URDF (assets/yam/yam.urdf) describe the SAME robot
-- the URDF is generated from the MJCF by scripts/gen_yam_urdf.py, so link/joint names and
kinematics match exactly. Same `link_name` ("link6") works for both. Stock full-pose control.

Prereqs (see docs/ORIN-SETUP.md):
  - PC Service running, Quest connected (Controller + Send ON)
  - `python scripts/gen_yam_urdf.py` has produced a consistent assets/yam/yam.urdf

Run (on the Orin desktop, DISPLAY=:0):
  python scripts/teleop_yam_mujoco.py
"""

import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)


def main(
    xml_path: str = "assets/yam/scene.xml",
    robot_urdf_path: str = "assets/yam/yam.urdf",  # consistent with yam.xml (generated from it)
    scale_factor: float = 1.0,
    visualize_placo: bool = False,
):
    config = {
        "right_hand": {
            "link_name": "grasp",  # EE = i2rt grasp_site tool frame (Z = gripper approach)
            "pose_source": "right_controller",
            "control_trigger": "right_grip",  # grip = clutch (engage >0.9)
            "vis_target": "right_target",
            "control_mode": "pose",  # full 6-DOF: EE tracks controller position + orientation
        }
    }

    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
    )

    # Soft joint regularization keeps the IK well-behaved (mirrors the UR5e / Flexiv examples).
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
