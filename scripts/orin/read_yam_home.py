"""Read the real YAM's J1-J6 (radians) and print them as a home qpos.

Enables each arm motor at ZERO torque (limp / back-drivable), medians a few encoder samples,
then disables. Hold the arm in the desired rest pose while this runs.

  PYTHONPATH=$HOME/i2rt $HOME/yam-venv/bin/python scripts/orin/read_yam_home.py [can0]
"""

import statistics
import sys
import time

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

MOTORS = [(1, MotorType.DM4340), (2, MotorType.DM4340), (3, MotorType.DM4340),
          (4, MotorType.DM4310), (5, MotorType.DM4310), (6, MotorType.DM4310)]
channel = sys.argv[1] if len(sys.argv) > 1 else "can0"

iface = DMSingleMotorCanInterface(channel=channel, bustype="socketcan", control_mode=ControlMode.MIT)
for mid, mt in MOTORS:
    try:
        iface.motor_on(mid, mt)
    except Exception as e:
        print(f"motor {mid} enable failed: {e}", file=sys.stderr)

samples = {mid: [] for mid, _ in MOTORS}
for _ in range(15):
    for mid, mt in MOTORS:
        try:
            fb = iface.set_control(mid, mt, 0, 0, 0, 0, 0)   # zero torque -> just read
            samples[mid].append(fb.position)
        except Exception:
            pass
    time.sleep(0.04)

for mid, _ in MOTORS:
    try:
        iface.motor_off(mid)
    except Exception:
        pass
iface.close()

vals = [statistics.median(samples[mid]) if samples[mid] else float("nan") for mid, _ in MOTORS]
for mid, v in zip([m for m, _ in MOTORS], vals):
    print(f"  J{mid}: {v:+.5f} rad  ({v*57.2958:+.1f} deg)  n={len(samples[mid])}", file=sys.stderr)
print("HOME_QPOS: " + " ".join(f"{v:.5f}" for v in vals))
