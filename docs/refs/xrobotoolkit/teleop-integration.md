# XRoboToolkit — how to add a robot (vendored guidance)

**Source:** https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python — `teleop_details.md`
and `README.md`, `main` branch.
**Fetched:** 2026-06-08. **License:** MIT.
**Note:** the framework is NOT installed on this Mac (it runs on the Orin). This is the
authoritative integration spec; re-check against the repo for the version you actually run.

---

## The required files (this is the rule YAM broke)

Quoting `teleop_details.md` → *Mujoco simulation → Robot definition files*:

> "both `.xml` and `.urdf` files are required and they should be **consistent with each other
> (same link names and joint names)**. The `.xml` file is for mujoco simulation, and the
> `.urdf` is for placo. Optionally, there should be 1 additional free floating body per end
> effector defined in the `.xml` file for visualization of commanded teleop targets in mujoco."

| Use case | Files required |
|---|---|
| **MuJoCo simulation teleop** | **both** `.xml` (MJCF, for MuJoCo) **and** `.urdf` (for placo), **consistent — same link & joint names**; + 1 free-floating body per EE for `vis_target` |
| **Hardware teleop** (UR5, ARX R5) | **only `.urdf`** |

The single most important line: **same link names and joint names across `.xml` and `.urdf`.**
`link_name` in the config must be that shared name.

## Config dict (the teleop manipulator_config)

Per `teleop_details.md`:
- `link_name` — EE link name, **as defined in the mujoco `.xml` & `.urdf`** (identical in both).
- `pose_source` — `"left_controller"` / `"right_controller"` (or hand/head/tracker source).
- `control_trigger` — key that gates whether this arm is active (e.g. `right_grip`).
- `control_mode` — optional: `"pose"` (default, full 6DOF) or `"position"` (3DOF position-only).
- `vis_target` — optional, mujoco-only: name of the free-floating body that shows the target.
- `motion_tracker` — optional: `{serial, link_target}` to drive an extra link (not for 6DOF arms).
- `gripper_config` — optional parallel-gripper control:
  ```python
  "gripper_config": {
      "type": "parallel",
      "gripper_trigger": "right_trigger",
      "joint_names": ["right_gripper_finger_joint1"],  # the ONE actuated joint
      "open_pos": [0.05],
      "close_pos": [0.0],
  }
  ```
  - The placo `.urdf` **does not need the gripper DOF**.
  - A multi-joint gripper keeps **one** actuated joint in the `.xml`; the others are driven by a
    MuJoCo `<equality>` constraint, e.g.
    `<joint name="..._constraint" joint1="...finger1" joint2="...finger2" polycoef="0 -1 0 0 0"/>`.

Canonical examples to model new code on (upstream `scripts/`):
- `scripts/simulation/teleop_dual_ur5e_mujoco.py` — sim, dual arm.
- `scripts/hardware/teleop_dual_ur5e_hardware.py` — hardware, URDF-only.
- `scripts/hardware/teleop_dual_arx_r5_hardware.py` — `link_name="right_link6"`, gripper as joint7.

## How YAM violated this (root cause of the tedium)

The guidance above existed the whole time. YAM broke it three ways:
1. **`.xml` and `.urdf` were NOT consistent** — exported separately, differ ~6 cm + ~180° in the
   EE frame, different joint limits/inertias. The doc's headline requirement, violated at the
   source. (Our fix: stop using `yam.urdf`; load placo from the MJCF via `Flags.mjcf` so there's
   one model — see `scripts/orin/patch_framework.py`.)
2. **`link_name` not identical across files** — `link6` (xml) vs `link_6` (urdf); only the
   `blupe-evals` copy was hand-renamed to line up.
3. **Gripper not done the documented way** — should be a `gripper_config` with one actuated joint;
   YAM injects it at runtime via `combine_arm_and_gripper_xml()` instead.

The correct long-term fix is to make the two files consistent (or generate both from one source)
so `link_name` and joints match, exactly as `teleop_details.md` requires.

## See also
- [../robot-models/INDEX.md](../robot-models/INDEX.md) — our URDF / MJCF / scene.xml format conventions.
