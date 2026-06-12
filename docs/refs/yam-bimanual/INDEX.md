# Bimanual YAM (i2rt) — vendored reference (researched 2026-06-12)

## Topology (official i2rt pattern)
- **One SocketCAN channel per arm; one `get_yam_robot(channel)` per arm.** Motor IDs are
  identical on every YAM (joints 0x01–0x06, gripper 0x07) so two arms can NEVER share a
  bus — bimanual follower = 2 USB-CAN adapters (4 with leaders).
- Official example: `examples/bimanual_lead_follower/bimanual_lead_follower.py` — four
  subprocesses of `minimum_gello.py`, channels `can_follower_r/can_leader_r/
  can_follower_l/can_leader_l`, ZMQ 1234 (right) / 1235 (left). Process-per-arm, not
  one process two buses (though two `get_yam_robot()` in one process also works).
- Docs: https://doc.i2rt.com/examples/bimanual-teleoperation (YAM Cell = 2+2).

## Channel naming (critical for reboot-stable bimanual)
- udev rules `/etc/udev/rules.d/90-can.rules` keyed on each CANable's `ATTRS{serial}`:
  `SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<serial>", NAME="can_left"`.
  Names must start with `can`, ≤13 chars. Map adapters one-at-a-time first.
  Guide: i2rt `docs/guides/set-persistent-can-ids.md`. Then per interface:
  `ip link set up <name> type can bitrate 1000000`.

## Models
- i2rt ships SINGLE-arm models only (`i2rt/robot_models/arm/yam/yam.{xml,urdf}` +
  gripper MJCFs attached programmatically). **No official two-YAM scene — we compose our
  own** (reference: github.com/uynitsuj/robots_realtime does bimanual YAM MuJoCo scenes,
  config `configs/yam/yam_bimanual_yam_leader.yaml`).
- Our approach: generate left/right name-prefixed copies of our `assets/yam/yam.xml` into
  one scene (MuJoCo can't <include> the same file twice — names collide).

## Teleop conventions
- No canonical VR mapping exists in i2rt (their bimanual is teaching-handle leader-
  follower; TRI raiden = leaders/SpaceMouse). left controller → left arm / right → right
  is ours to define; xrobotoolkit_teleop's manipulator_config already supports a
  `left_hand` entry alongside `right_hand`.

## Gripper per arm
- Command space [0,1]; limits are NOT constants — i2rt `detect_gripper_limits()` drives
  the gripper at boot (it MOVES) unless `gripper_limits_override=[closed, open]` is
  passed. For a bimanual standard: calibrate once per physical arm, persist both values
  per side, pass overrides at boot. Gripper kp/kd defaults 20.0/0.5.

## Budget sanity
- Per-bus load (7 motors @ 50–100 Hz) is fine at 1 Mbps; arms don't share a bus so
  bimanual doesn't change CAN math. Power: one supply per arm (YAM Cell pattern).
- Damiao 400 ms watchdog per arm — each arm's command stream must stay alive
  independently.
