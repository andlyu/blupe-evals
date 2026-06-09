"""Teleop a single YAM arm in MuJoCo with an XRoboToolkit Quest controller.

Quest controller -> XRoboToolkit PC Service -> xrobotoolkit_sdk -> placo IK -> MuJoCo.

Prereqs (see README):
  - XRoboToolkit PC Service running on this host, Quest connected (Controller + Send ON)
  - `xrobotoolkit_sdk` and `xrobotoolkit_teleop` installed (Python 3.10)

Run:
  python scripts/teleop_yam_mujoco.py
"""

import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)


def main(
    xml_path: str = "assets/yam/scene.xml",
    robot_urdf_path: str = "assets/yam/yam.urdf",
    scale_factor: float = 1.0,
    visualize_placo: bool = True,
):
    # Single right arm. NOTE on names (verified against the vendored assets):
    #   - placo IK runs on the URDF, whose EE link is "link_6" (underscore).
    #   - the MuJoCo body for the same link is "link6" (no underscore).
    # `link_name` here feeds placo IK, so it must be the URDF name "link_6".
    config = {
        "right_hand": {
            "link_name": "link_6",  # YAM EE link in yam.urdf (placo target)
            "pose_source": "right_controller",
            "control_trigger": "right_grip",  # grip = clutch (engage >0.5)
            "vis_target": "right_target",
        }
    }

    MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
    ).run()


if __name__ == "__main__":
    tyro.cli(main)
