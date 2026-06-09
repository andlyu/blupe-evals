---
name: adding-a-new-arm
description: Add a robot arm to XRoboToolkit MuJoCo teleop in blupe-evals — the recipe plus the EE tool-frame gotcha that cost us a day on YAM. Use when integrating a new arm or debugging "gizmo/wrist orientation is wrong" teleop.
---

# Adding a new arm (XRoboToolkit MuJoCo teleop)

## What the framework needs (`teleop_details.md`)
- A MuJoCo **MJCF** (`.xml`, sim) and a placo **URDF** (`.urdf`, IK) for the **same** robot:
  identical link & joint names, identical kinematics. **Generate the URDF from the MJCF**
  so they can't drift (`scripts/gen_yam_urdf.py`).
- One free-floating **mocap body per EE** in `scene.xml` (`vis_target`) — the goal gizmo.
- A config dict: `link_name`, `pose_source`, `control_trigger`, `vis_target`, `control_mode="pose"`.
- Hardware teleop needs **only the URDF**.

## Recipe
1. Drop the arm's MJCF + meshes in `assets/<arm>/`. **Ground truth = the MJCF the vendor's
   stack actually drives the real arm with.** (i2rt loads only the `.xml`; its `.urdf` is stale.)
2. Complete the MJCF for sim: position actuators per joint, a `home` keyframe, real mass on any
   massless EE link (else NaN), `armature`/`damping`, and visual-only geoms
   (`contype=0 conaffinity=0`) so the home pose can't self-collide and jam a joint.
3. **Define the EE as a proper TOOL frame** (see THE GOTCHA) — a `grasp`/`tool0` body whose
   **+Z = the gripper approach axis**.
4. Generate the URDF from the MJCF — same names; EE link = the tool frame.
5. Add the `vis_target` mocap to `scene.xml`.
6. Wire the config (`link_name=<tool frame>`, `control_mode="pose"`).
7. **Verify before trusting it** (`scripts/test_yam_ik.py`): consistency (placo EE == MuJoCo
   EE ≈ 0), hold-test (goal=current ⇒ no motion = no jump on clutch), track-test.

## THE GOTCHA — EE tool frame (cost us a day on YAM)

**Symptom:** teleop works for translation but the goal gizmo's **blue (Z) axis points the wrong
way** (e.g. down) while the gripper points forward, and **wrist rotation feels wrong / flails**.

**Cause:** we pointed the IK at the raw last body `link6` from the *standalone* `yam.xml`. That
link is a **placeholder** (`pos="0 0 0" quat="1 0 0 0"`, joint6 `axis="0 0 1"`). i2rt never uses
it — `combine_arm_and_gripper_xml` **overrides link6's pos/quat/joint-axis from the gripper mount
config** when attaching the gripper. The real link6 is
`pos="0 -0.042 0.0405" quat="0.5 -0.5 -0.5 -0.5"`, joint6 `axis="0 0 -1"`. So the EE frame's +Z
pointed **down**, not along the gripper.

**Why it was sneaky:** the URDF was generated from the same bad MJCF, so sim and IK were perfectly
**self-consistent** — every consistency/hold/track test passed. It was *"consistent with a wrong
model."* Self-consistency ≠ correct; the model must match the **real arm**.

**Fix:**
1. Take the real `link6` (pos/quat/joint6 axis) from i2rt's **combined** model:
   `combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)`.
2. Add a `grasp` EE body matching i2rt's `grasp_site` — rel link6 `pos="0 0 -0.1347" quat="0 1 0 0"`,
   +Z = approach, 13.5 cm out.
3. `link_name="grasp"`, regenerate URDF.

Result: blue ∥ gripper, natural 6-DOF wrist, and the model matches the real arm (same URDF works
on hardware).

**General rule:** the EE/IK frame must be the **tool/grasp frame from the COMBINED arm+gripper
model**, never the raw last link of a standalone arm xml. UR5e worked out of the box because
mujoco_menagerie ships a proper `tool0` frame; YAM's vendored xml did not.

## Getting a tool frame's transform from i2rt
```python
from i2rt.robots.utils import combine_arm_and_gripper_xml, ArmType, GripperType
xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)  # writes /tmp/i2rt_*.xml
# load in mujoco, read grasp_site site_xmat/site_xpos, express relative to link6:
#   R_rel = R_link6.T @ R_site ;  p_rel = R_link6.T @ (p_site - p_link6)
```
Needs `python-can click pydantic rich tabulate pyyaml` in the env.

## Reference files (YAM, the worked example)
- `assets/yam/yam.xml` — MJCF, real link6 + `grasp` tool frame, actuators, home keyframe.
- `assets/yam/yam.urdf` — generated from the MJCF (`scripts/gen_yam_urdf.py`).
- `assets/yam/scene.xml` — floor + `right_target` mocap gizmo.
- `scripts/teleop_yam_mujoco.py` — stock full-pose config, `link_name="grasp"`.
- `scripts/test_yam_ik.py` — consistency / hold / track checks.
