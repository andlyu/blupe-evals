"""Teleop the REAL YAM arm via i2rt (runs on the Jetson Orin).

This is the hardware counterpart to teleop_yam_mujoco.py. The plan (from the
handoff) is a thin bridge:

    Quest controller -> xrobotoolkit_sdk -> placo IK joint targets
                     -> i2rt `command_joint_pos`   (state via `get_joint_pos`)

Model this on XRoboToolkit's own hardware example:
    XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_dual_ur5e_hardware.py

Hardware facts (see docs/XROBOTOOLKIT-YAM-HANDOFF.md for the full list):
  - Orin: andrew@192.168.0.185, Ubuntu 22.04 arm64, i2rt venv at ~/i2rt/.venv
  - YAM = 6 arm joints (DM4340x3, DM4310x3) + gripper (DM4310), motor ids 1-7
  - Damiao MIT mode, ~400 ms watchdog -> must stream commands or the arm goes limp
  - CAN bus name swaps can0/can1 across reboots -> ping motors to find it
  - EE: sim body "link6" / placo URDF link "link_6" / real i2rt site "grasp_site"

Safe torque-off sequence (do this on shutdown):
    robot._stop_event.set(); robot._server_thread.join()   # stop control thread FIRST
    for i in range(1, 8): robot.motor_chain.motor_interface.motor_off(i)
    robot.close()
  (A naive disable throws a harmless fd=-1 traceback but does disable.)

TODO: implement. Left as a stub until sim teleop works end-to-end.
"""

import tyro


def main(
    robot_urdf_path: str = "assets/yam/yam.urdf",
    can_channel: str = "can0",  # may be can1 after a reboot — ping motors to confirm
    scale_factor: float = 1.0,
):
    raise NotImplementedError(
        "Hardware teleop not implemented yet. Get teleop_yam_mujoco.py working first, "
        "then port the placo IK loop here and drive i2rt's command_joint_pos. "
        "See the module docstring and the XRoboToolkit dual_ur5e_hardware example."
    )


if __name__ == "__main__":
    tyro.cli(main)
