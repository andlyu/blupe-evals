"""Turn the YAM OFF — cut ALL motor torque so the arm goes fully limp, then exit.

Killing the serve does NOT turn the arm off: i2rt's Damiao motors keep holding their last
commanded position with torque on. The real limp is `disable_motorchain` (stop the control thread,
then motor_off each motor, then close the bus). We build the robot with a gripper_limits_override
so it SKIPS gripper calibration — no gripper movement on the way to off.

    ~/i2rt/.venv/bin/python YAM_control/turn_off.py --channel can0

Stop the serve / gripper_cycle first (one owner of the CAN bus at a time).
"""

import argparse

import numpy as np

from yam_real_serve import disable_motorchain   # the proven full-limp sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    args = ap.parse_args()

    from i2rt.robots.get_robot import get_yam_robot
    print(f"[off] connecting to YAM on {args.channel} (skipping gripper calibration)...", flush=True)
    robot = get_yam_robot(args.channel, gripper_limits_override=np.array([0.0, 1.0]))
    disable_motorchain(robot)
    print("[off] done.", flush=True)


if __name__ == "__main__":
    main()
