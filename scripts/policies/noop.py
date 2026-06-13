"""No-op policy for testing robot-side policy execution.

It proves the relay can start a policy on the robot computer without commanding motion.
"""

import time


def run(robot, stop):
    obs = robot.read()
    print(f"[policy] noop start joints={obs.joint_pos.tolist()} gripper={obs.gripper:.4f}", flush=True)
    start = time.monotonic()
    while not stop() and time.monotonic() - start < 5.0:
        time.sleep(0.05)
    print("[policy] noop done", flush=True)
