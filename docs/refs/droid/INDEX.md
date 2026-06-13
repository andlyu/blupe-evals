# DROID rig — vendored reference (2026-06-12)

DROID (droid-dataset.org) standardizes on a **Franka Panda 7-DOF** + **Robotiq 2F-85**
gripper (+ Zed cameras, Oculus-based teleop — kindred spirit to our stack).

## Sim assets (ours)
- Composed by `scripts/gen_droid_scene.py` from MuJoCo Menagerie models vendored in
  `assets/droid/` (`franka_emika_panda/panda_nohand.xml` + `robotiq_2f85/2f85.xml`,
  meshes merged into one assets dir). Robotiq attaches at the panda's `attachment_site`
  with its FULL mechanism (8 coupled finger joints + tendon + `grip_fingers_actuator`),
  so the trigger opens/closes it in sim. nq=15, nu=8 — the gripper actuator lands at
  index 7, AFTER the 7 arm actuators, so the eval's `d.ctrl[:7]` arm slicing stays clean.
- IK URDF is **ARM-ONLY** (generated straight from `panda_nohand.xml`): the Robotiq's
  closed loop has joints off their body origins (can't be a URDF tree) and the fingers are
  downstream of the EE frame, so placo never needs them. Panda mounts at the scene origin
  so arm FK matches.
- EE frame: `attachment`. Gripper: `arms.py` sets `gripper_joint="grip_fingers_actuator"`
  and `gripper_open_high=False` (Robotiq ctrl 0=open, 255=closed — INVERTED vs SO-101's
  hinge where high=open). The trigger TOGGLES (press = open↔close), same as YAM.
- Gotchas hit and encoded in the generator:
  - MjSpec attach does NOT propagate child `<option>`s — panda needs
    `integrator="implicitfast"` (without it: 0.7 rad/s oscillation at rest).
  - attach leaves the panda's anonymous root `<default>` unnamed → re-parses standalone
    but fails via `<include>` ("empty class name"); generator names it.

## Real hardware (later)
- Franka control: DROID uses polymetis (franka-interface); modern alternatives: franky /
  libfranka direct. 1 kHz torque interface; our serve would speak position @ 50 Hz on top.
- Robotiq 2F-85: Modbus RTU over USB/serial; [0,255] position — our [0,1] maps linearly.
- DROID's own teleop is Quest-controller based (VRPolicy in their repo) — conventions
  worth comparing when wiring the real thing: https://github.com/droid-dataset/droid
