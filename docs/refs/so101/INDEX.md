# SO-101 (SO-ARM101) — vendored reference (researched 2026-06-12)

LeRobot/TheRobotStudio 6-motor arm: 5 arm joints + 1 gripper, Feetech STS3215 serial servos.

## Hardware
- 5 DOF body + 1 DOF gripper; Feetech STS3215 bus servos (12-bit encoder, 4096/rev).
  Follower 1/345 gearing (7.4 V, 19.5 kg·cm; 12 V variant 30 kg·cm). Leader variant has
  lighter mixed gearing (back-drivable) + trigger handle.
- Comms: motors daisy-chained on half-duplex TTL serial → Waveshare bus-servo board →
  USB-C (`/dev/ttyACM*`), 1 Mbps. Board jumpers MUST be on channel B.
- Sources: https://github.com/TheRobotStudio/SO-ARM100 ·
  https://huggingface.co/docs/lerobot/so101

## Software (lerobot)
- Driver: `lerobot` `[feetech]` extra; robot type `so101_follower`
  (`src/lerobot/robots/so_follower/so_follower.py` — module recently renamed from
  `so101_follower/`; import via the registered type string, pin the version).
- Motor IDs fixed: shoulder_pan=1, shoulder_lift=2, elbow_flex=3, wrist_flex=4,
  wrist_roll=5, gripper=6 (all "sts3215").
- Commands: `send_action({"<joint>.pos": float})` → `sync_write("Goal_Position")`.
  Units: `use_degrees=True` (default) → degrees; else [-100,100] of calibrated range.
  **Gripper always [0,100], 0=closed 100=open** (→ our [0,1] = /100).
- 50 Hz position streaming is comfortably in budget (driver sets Return_Delay_Time=0).
- Calibration: `lerobot-calibrate` (mid-range homing + ROM sweep) → JSON in
  `~/.cache/huggingface/lerobot/calibration/robots/<name>/<id>.json` + servo EEPROM.

## Models (the ones we vendor)
- **Use `Simulation/SO101/so101_new_calib.{xml,urdf}`** from TheRobotStudio/SO-ARM100
  (zero = mid-range, matches current lerobot calibration). `scene.xml` includes the robot.
  The OLD-calib variants and the mujoco_menagerie `trs_so_arm100` (SO-100, different joint
  names) are traps — don't mix.
- Joints (rad): shoulder_pan ±1.920, shoulder_lift ±1.745, elbow_flex ±1.69,
  wrist_flex ±1.658, wrist_roll −2.744..2.841, gripper −0.1745..1.745.
- Source: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

## Safety
- No watchdog: unclean disconnect leaves servos holding last goal WITH torque.
  (`disable_torque_on_disconnect=True` only covers clean disconnects.) Robot-side serve
  must own this for our stack.
- Gripper burnout caps written by configure(): Max_Torque_Limit=500/Protection_Current=250.
- `max_relative_target` = optional per-step jump clamp (costs an extra sync_read).

## Gotchas
- First-time servo ID setup mandatory: `lerobot-setup-motors`, one motor at a time.
- Two calibration conventions (old horizontal-zero vs new mid-range-zero) — mismatched
  model↔calibration shows up as ~90° shoulder/elbow offsets.
- `use_degrees` flips units between datasets/policies — pick one and pin it.
