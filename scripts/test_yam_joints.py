"""Per-joint actuator test (no Quest, no IK): command each joint to a target and check it
reaches it. Isolates a stuck/broken joint actuator from IK behavior.
Run in `xr` env from ~/blupe-evals:  python scripts/test_yam_joints.py
"""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("assets/yam/scene.xml")
d = mujoco.MjData(m)
hid = m.key("home").id
print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
print("njnt:", m.njnt, "nu:", m.nu)

for i in range(6):
    mujoco.mj_resetDataKeyframe(m, d, hid)
    mujoco.mj_forward(m, d)
    home_ctrl = d.ctrl.copy()
    lo, hi = m.jnt_range[i]
    cur = d.qpos[i]
    tgt = cur + 0.6 if (cur + 0.6) < hi - 0.05 else cur - 0.6
    d.ctrl[:] = home_ctrl
    d.ctrl[i] = tgt
    for _ in range(3000):
        mujoco.mj_step(m, d)
    err = abs(tgt - d.qpos[i])
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"{jn}: range[{lo:+.2f},{hi:+.2f}] cmd {tgt:+.2f} reached {d.qpos[i]:+.2f} "
          f"err {err:.3f} -> {'OK' if err < 0.05 else 'STUCK'}")
