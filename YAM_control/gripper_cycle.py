"""Standalone gripper exerciser: cycle the gripper OPEN <-> CLOSE.

THE GRIPPER IS NORMALIZED [0,1] -- this was the whole confusion. i2rt remaps the gripper
command/observation to [0,1] via its limits: 0.0 = fully CLOSED, 1.0 = fully OPEN. You never touch
raw motor radians. (Commanding a raw number like 0.164 is read as 16% open -> the old "2 cm" bug.)

Open/close the way TRI's `raiden` stack does (raiden/robot/controller.py): walk the target a small
BOUNDED step ahead of the ACTUAL position toward the goal (0 or 1). When the jaws are blocked the
actual position stalls, so the command can never get more than `lead` ahead -> bounded force ->
no grind, no crush. When free it walks to the stop.

    ~/i2rt/.venv/bin/python scripts/orin/gripper_cycle.py --channel can0

Stop = Ctrl-C (or `pkill -9 -f gripper_cycle`) -> motors torque-off. can0 must be free.
"""

import argparse
import time

import numpy as np

from yam_real_serve import disable_motorchain   # reuse the serve's torque-off

N = 6   # arm joints (gripper is index N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--period", type=float, default=4.0, help="seconds per open/close phase")
    ap.add_argument("--lead", type=float, default=0.9 * 6 / 71,
                    help="max amount (in [0,1]) the target may lead the ACTUAL position "
                         "(raiden uses 0.9*6/71 ~= 0.076). Lower = gentler / weaker.")
    args = ap.parse_args()

    from i2rt.robots.get_robot import get_yam_robot
    print(f"[cycle] connecting real YAM on {args.channel} ...", flush=True)
    robot = get_yam_robot(args.channel)        # default; i2rt's limits set the [0,1] normalization
    info = robot.get_robot_info()
    gidx = info.get("gripper_index", N)
    arm = np.asarray(robot.get_joint_pos(), dtype=float)[:N].copy()    # hold the arm here
    g0 = float(robot.get_joint_pos()[gidx])
    print(f"[cycle] gripper idx={gidx} normalized 0=closed 1=open; start pos={g0:.3f}; "
          f"lead={args.lead:.3f}. Ctrl-C = stop.", flush=True)

    phases = [("OPEN", 1.0), ("CLOSE", 0.0)]
    last_print = 0.0
    try:
        while True:
            for label, goal in phases:
                t_end = time.monotonic() + args.period
                while time.monotonic() < t_end:
                    g = float(robot.get_joint_pos()[gidx])             # normalized [0,1]
                    if goal > g:
                        target = min(goal, g + args.lead)              # opening -> toward 1.0
                    else:
                        target = max(goal, g - args.lead)              # closing -> toward 0.0
                    cmd = np.asarray(robot.get_joint_pos(), dtype=float).copy()
                    cmd[:N] = arm                                       # keep the arm still
                    cmd[gidx] = target
                    robot.command_joint_pos(cmd)
                    now = time.monotonic()
                    if now - last_print > 0.3:
                        print(f"[cycle] {label:5s} pos={g:.3f} -> tgt={target:.3f}", flush=True)
                        last_print = now
                    time.sleep(0.02)
    except KeyboardInterrupt:
        print("[cycle] stopped by user", flush=True)
    finally:
        disable_motorchain(robot)


if __name__ == "__main__":
    main()
