"""MJCF -> kinematics-URDF converter shared by the asset generators.

The XRoboToolkit framework needs an MJCF (MuJoCo sim) + URDF (placo IK) pair with matching
link/joint names. We always DERIVE the URDF from the MJCF body tree so they cannot disagree
(recipe proven in gen_yam_urdf.py). Links are massless frames — the URDF is for kinematics
only. Requires every joint at its body origin (asserts).
"""

import mujoco
import numpy as np
from meshcat import transformations as tf


def _rpy(wxyz):
    """URDF rpy (fixed-axis XYZ) from a wxyz quaternion. Uses euler_from_matrix, which
    handles gimbal lock (pitch = +-90 deg) — a naive three-atan2 extraction silently
    returns a WRONG roll/yaw pair there (cost: 300 mm of FK error on the SO-101, whose
    upper_arm frame sits exactly at pitch -90)."""
    return tf.euler_from_matrix(tf.quaternion_matrix(wxyz), "sxyz")


def write_urdf(m, dst, robot_name):
    """Walk the MJCF body tree -> URDF with identical names. Returns the EE-less path."""
    def bn(b):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)

    body_names = {bn(b) for b in range(m.nbody)}      # joints may not shadow a link name:
                                                      # placo frames would become ambiguous
    out = ['<?xml version="1.0"?>', f'<robot name="{robot_name}">', '  <link name="base_link"/>']
    for b in range(1, m.nbody):
        name, parent = bn(b), bn(m.body_parentid[b])
        parent = "base_link" if parent == "world" else parent
        pos = m.body_pos[b]
        r, p, y = _rpy(m.body_quat[b])
        out += [f'  <link name="{name}">',
                '    <inertial><mass value="0.5"/><origin xyz="0 0 0"/>'
                '<inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>',
                '  </link>']
        jids = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        if jids:
            j = jids[0]
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jn in body_names:
                # Renaming EITHER side breaks something downstream (joint rename breaks the
                # framework's by-name placo<->mujoco joint mapping; ambiguous names break
                # placo frame lookup). The fix is renaming the BODY in the vendored MJCF.
                raise SystemExit(f"joint {jn!r} shadows a body name — rename the MJCF body "
                                 f"(e.g. {jn}_link) and regenerate")
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
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[mjcf2urdf] wrote {dst} ({m.nbody - 1} links)")


def fk_check(m, urdf, ee_bodies, q=None):
    """FK consistency sim vs URDF at a test config; PASS = same EE position < 1 mm."""
    import placo
    d = mujoco.MjData(m)
    r = placo.RobotWrapper(urdf)
    nq = len(r.state.q)
    n = m.nq
    q = np.full(n, 0.3) if q is None else np.asarray(q, dtype=float)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    q = np.clip(q, lo + 0.05, hi - 0.05)               # stay inside every joint's range
    d.qpos[:n] = q
    mujoco.mj_forward(m, d)
    qq = r.state.q.copy()
    if nq > n:
        qq[:7] = [0, 0, 0, 0, 0, 0, 1]
    qq[nq - n:] = q
    r.state.q = qq
    r.update_kinematics()
    ok = True
    for ee in ee_bodies:
        err = float(np.linalg.norm(d.body(ee).xpos - np.array(r.get_T_world_frame(ee))[:3, 3]))
        good = err < 1e-3
        ok = ok and good
        print(f"[mjcf2urdf] FK {ee}: {err * 1000:.2f} mm -> {'PASS' if good else 'FAIL'}")
    return ok
