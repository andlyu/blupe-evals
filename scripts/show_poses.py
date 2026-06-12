"""Debug viewer (Mac): toggle the YAM between HOME and the gripper-forward POLICY pose.

Shows why the policy looks wrong — the +25 cm world-X target is reached via a folded-up config.
Run on the laptop:  .venv/bin/mjpython scripts/show_poses.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path("assets/yam/scene.xml")
d = mujoco.MjData(m)
HOME = m.key("home").qpos[:6].copy()
POLICY = np.array([-0.007, 1.779, 0.703, 0.971, -0.034, -0.225])  # IK for +25cm in front; grasp->[0.489,..]
POSES = [("HOME", HOME), ("POLICY (gripper +25cm in front)", POLICY)]

with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.azimuth, v.cam.elevation, v.cam.distance = 120, -20, 1.6
    v.cam.lookat[:] = [0.2, 0.0, 0.2]
    i = 0
    while v.is_running():
        name, q = POSES[i % 2]
        cur = d.qpos[:6].copy()
        for s in np.linspace(0.0, 1.0, 60):              # ease into the pose
            if not v.is_running():
                break
            d.qpos[:6] = cur + s * (q - cur)
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(0.02)
        print(f"showing {name}: qpos = {np.round(q, 2)}", flush=True)
        for _ in range(120):                             # hold ~2.4 s
            if not v.is_running():
                break
            v.sync()
            time.sleep(0.02)
        i += 1
