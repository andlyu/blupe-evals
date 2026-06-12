# OpenArm (Enactic) — vendored reference (researched 2026-06-12)

Open-source 7-DOF human-scale arm + parallel gripper, Damiao CAN motors (same vendor as
YAM). Bimanual is first-class ($6,500 complete system). Canonical org: github.com/enactic
(continuation of reazon-research). Do NOT confuse with berkeleyopenarms ("Blue").
Current version: v2.0 / "OpenArm 02". Docs: https://docs.openarm.dev

## Hardware
- 7 DOF/arm + gripper (8th motor). Payload 4.1 kg nominal / 6.0 peak. QDD backdrivable.
- Motors: DM8009 (j1,j2), DM4340 (j3,j4), DM4310 (j5-j7, gripper). IDs: send 0x0N /
  recv 0x1N for joint N (1..8) — different convention from i2rt's YAM map.
- Comms: **CAN-FD** over socketcan (1M/5M; `--no-fd` classic fallback). One CAN interface
  per arm (bimanual = can0+can1). USB-CAN adapter is **Linux-only**.

## Software
- Repos: openarm_can (C++ core, Python bindings UNSTABLE), openarm_driver, openarm_control,
  openarm_ros2, openarm_description (URDF/xacro: `openarmv2.urdf`), openarm_mujoco (MJCF,
  pip `openarm-mujoco`, default branch **master**), openarm_teleop (bilateral leader-
  follower), dora-rs nodes incl. dora-openarm-vr (Quest 3).
- **Easiest integration surface: lerobot** (`[damiao]` extra): `openarm_follower`,
  degrees-based `joint_N.pos` + `gripper.pos`, per-joint position_kp/kd (MIT mode),
  `max_relative_target` clamp, `side="left"/"right"` mirrored limits.
  https://huggingface.co/docs/lerobot/en/openarm

## Models (what we vendor for sim)
- MJCF: enactic/openarm_mujoco `v2/openarm_bimanual.xml` (+ v1 single `v1/openarm.xml`,
  `v1/scene.xml`). Joints `openarm_{left,right}_joint1..7` + `finger_joint1/2`
  (mimic-coupled). Left limits (rad): j1[-3.49,1.40] j2[-3.32,0.175] j3 ±1.571
  j4[0,2.443] j5 ±1.571 j6 ±0.785 j7 ±1.571; right mirrored. Actuator kp by motor class
  (DM8009 230, DM4340 190, DM4310 30); torque 40/27/7 N·m.
- URDF: enactic/openarm_description → `openarmv2.urdf` (for placo IK).
- Gripper: fingers 0–0.7854 rad, one position actuator per arm (right uses [-0.7854,0]).
  Our [0,1] → finger angle mapping is a thin linear layer.

## Safety
- Damiao watchdog timeout is FIRMWARE-CONFIGURABLE (Damiao Debugging Tool); lerobot warns
  timeout=0 ⇒ motors ignore commands. The ~400 ms damping behavior we rely on with YAM is
  NOT guaranteed — verify per motor at commissioning. E-stop required (kit includes one).
  Safety guide: https://docs.openarm.dev/getting-started/safety-guide

## Gotchas
- CAN-FD bring-up differs from YAM: `ip link set can0 type can bitrate 1000000
  dbitrate 5000000 fd on`.
- v1 vs v2 models differ (j7 motor class, gripper geometry) — stay on v2 consistently.
- Units by layer: lerobot=degrees, MJCF/URDF=radians.
