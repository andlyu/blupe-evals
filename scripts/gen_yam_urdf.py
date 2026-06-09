"""Generate assets/yam/yam.urdf as a kinematics URDF IDENTICAL to assets/yam/yam.xml.

The XRoboToolkit framework needs a consistent MJCF (MuJoCo sim) + URDF (placo IK) pair with
matching link/joint names (teleop_details.md). yam.xml is the i2rt ground-truth model, so we
derive the URDF directly from it -- they cannot disagree. Link names == MJCF body names
(link1..link6); joint names == joint1..joint6. Run in the `xr` env from ~/blupe-evals.
"""

import mujoco
import numpy as np
from meshcat import transformations as tf

SRC = "assets/yam/yam.xml"
DST = "assets/yam/yam.urdf"
m = mujoco.MjModel.from_xml_path(SRC)


def rpy_from_quat(wxyz):
    R = tf.quaternion_matrix(wxyz)[:3, :3]  # URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)
    return (np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])),
            np.arctan2(R[1, 0], R[0, 0]))


def bn(b):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)


out = ['<?xml version="1.0"?>', '<robot name="yam">', '  <link name="base_link"/>']
for b in range(1, m.nbody):
    name, parent = bn(b), bn(m.body_parentid[b])
    parent = "base_link" if parent == "world" else parent
    pos = m.body_pos[b]
    r, p, y = rpy_from_quat(m.body_quat[b])
    out += [f'  <link name="{name}">',
            '    <inertial><mass value="0.5"/><origin xyz="0 0 0"/>'
            '<inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>',
            '  </link>']
    jids = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
    if jids:
        j = jids[0]
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        ax, (lo, hi) = m.jnt_axis[j], m.jnt_range[j]
        assert np.allclose(m.jnt_pos[j], 0), f"{jn} not at body origin"
        out += [f'  <joint name="{jn}" type="revolute">',
                f'    <parent link="{parent}"/>', f'    <child link="{name}"/>',
                f'    <origin xyz="{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}" rpy="{r:.9f} {p:.9f} {y:.9f}"/>',
                f'    <axis xyz="{ax[0]:.6f} {ax[1]:.6f} {ax[2]:.6f}"/>',
                f'    <limit lower="{lo:.6f}" upper="{hi:.6f}" effort="100" velocity="3.14159"/>',
                '  </joint>']
    else:
        out += [f'  <joint name="fixed_{name}" type="fixed">',
                f'    <parent link="{parent}"/>', f'    <child link="{name}"/>',
                f'    <origin xyz="{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}" rpy="{r:.9f} {p:.9f} {y:.9f}"/>',
                '  </joint>']
out.append("</robot>")
open(DST, "w").write("\n".join(out) + "\n")
print(f"wrote {DST} ({m.nbody - 1} links, EE link = {bn(m.nbody - 1)})")

# ---- verify: consistency with the sim + placo can solve ----
import placo  # noqa: E402

EE = bn(m.nbody - 1)
d = mujoco.MjData(m)
r = placo.RobotWrapper(DST)
nq = len(r.state.q)


def consistent_at(q6):
    d.qpos[:6] = q6
    mujoco.mj_forward(m, d)
    qq = r.state.q.copy()
    if nq > 6:
        qq[:7] = [0, 0, 0, 0, 0, 0, 1]
    qq[nq - 6:] = q6
    r.state.q = qq
    r.update_kinematics()
    T = np.array(r.get_T_world_frame(EE))
    ang = np.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(d.body(EE).xmat.reshape(3, 3).T @ T[:3, :3]) - 1) / 2))))
    return ang, float(np.linalg.norm(d.body(EE).xpos - T[:3, 3]))


a1, p1 = consistent_at(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
a2, p2 = consistent_at(np.array([0.5, 1.2, 0.8, -0.3, 0.4, 0.6]))
print(f"consistency vs sim: cfg1 {a1:.3f}deg/{p1:.4f}m  cfg2 {a2:.3f}deg/{p2:.4f}m  "
      f"-> {'PASS' if max(a1, a2) < 0.5 and max(p1, p2) < 1e-3 else 'FAIL'}")

s = placo.KinematicsSolver(r)
s.dt = 0.01
if nq > 6:
    s.mask_fbase(True)
r.update_kinematics()
s.add_frame_task(EE, np.array(r.get_T_world_frame(EE))).configure("ee", "soft", 1.0)
try:
    s.solve(True)
    print("placo solve: OK")
except Exception as e:
    print("placo solve: FAIL ->", repr(e)[:80])
