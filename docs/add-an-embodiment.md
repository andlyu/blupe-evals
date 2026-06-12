# Add a new embodiment (a new arm MODEL)

This is the advanced path: teaching the stack an arm type it has never seen. If your arm
model already exists in `scripts/arms.py` (YAM, SO-101, ...), you don't need this page —
go to [integrate-your-hardware.md](integrate-your-hardware.md) to connect your unit.

Two phases, in order. **Phase A (sim)** makes the embodiment drivable in MuJoCo — no
hardware at risk; the full eval loop (trials, reports) works on the sim twin. **Phase B**
writes the reference serve for the embodiment's motor driver. Do not skip A: every Phase B
problem is easier to see when the sim twin already works.

## Phase A — sim-ready

Your embodiment = one `ArmSpec` entry in `scripts/arms.py` + a model folder in
`assets/<arm>/`. Copy the closest existing arm (`yam` = 6-DOF single, `so101` = 5-DOF with
a modeled gripper) and work through:

1. **Get the model files: MJCF + URDF, consistent with each other.** Same link names, same
   joint names in both — the framework's headline rule and the single most common
   integration failure (see `docs/refs/xrobotoolkit/teleop-integration.md`, including how
   the YAM violated it and what it cost). If you only have one format, GENERATE the other
   from it (one source of truth) — never export both from CAD separately.
   Mesh paths must be relative and resolvable; every link needs its mesh.
2. **Build the scene** (`assets/<arm>/scene.xml`): include the arm MJCF, position actuators
   for the arm joints, a **`home` keyframe** (safe, natural pose — it anchors GO_HOME,
   policy waypoints, and IK regularization), one free-floating **mocap body per controlled
   hand** named `right_target` (and `left_target` if bimanual) for the teleop target
   visual, and `<visual><global offwidth="1920" offheight="720"/>` (the headset canvas
   renders double-wide).
3. **Fill the `ArmSpec`**: `dof` = arm joints only (gripper EXCLUDED — gripper travels as a
   separate 0..1 value end to end); `ee_link` = the URDF frame the IK tracks; `ee_body` =
   the MJCF body for FK/HUD (often the same name; differs on SO-101); `gripper_joint` if
   the sim models one; `max_vel` = the joint-speed cap bounding teleop, policies, homing.
4. **Check it** (until the `--check` validator ships, by hand): placo loads the URDF;
   MuJoCo loads the scene; the `home` keyframe applies; IK reaches a probe pose near home.
5. **Live sim teleop**: run the eval with `--arm <name> --cameras none`; drive it from the
   headset or the browser console (`:8810`). Clutch, move, GO_HOME, run a policy trial.
   ✅ sim-ready.

## Phase B — the reference serve for this embodiment

The stack never talks to motor drivers. It talks to a **serve**: a small TCP server that
owns the hardware and the safety. Reference implementation:
`YAM_control/yam_real_serve.py` (~200 lines over i2rt/CAN). Write the equivalent over the
new embodiment's driver (SO-101: lerobot `so101_follower`; its gripper is 0..100 → /100).

### The serve wire protocol

Newline-delimited JSON over TCP (default `:5599`):

```
server -> client, once on connect:   {"start_joints": [q1..qN]}     # no-jump seed
client -> server, ~50 Hz:            {"q": [q1..qN], "g": 0..1}     # joint targets + gripper
client -> server, optional field:    {"q": ..., "t": <any>}         # if present, echo after APPLY:
server -> client:                    {"ack": <t>}                   #   -> sender-side RTT probe
client -> server, on quit:           {"shutdown": true}             # -> cut motor torque
```

### The safety contract (non-negotiable, lives in the serve)

Robot-side on purpose — no network or operator bug can bypass it:
1. **Velocity-clamp every command** (rad/s cap) — the backstop even if the operator stack
   misbehaves.
2. **Hold the last pose on disconnect** — a dropped link freezes the arm, never drops it.
3. **Torque off on `shutdown` and on your own exit** (Ctrl-C, crash handlers).
4. **`start_joints` reports the arm's REAL current joints** so the client ramps from where
   the arm actually is — this is what prevents the first-command jump.

Once the reference serve exists, every physical unit of this embodiment onboards through
[integrate-your-hardware.md](integrate-your-hardware.md) — fleet token, agent one-liner,
Turn ON.

## What you DON'T have to touch

Headset transport, the eval state machine, recording/judging/reports, fleet management,
operator UX — all arm-agnostic. They see joint vectors, a 0..1 gripper, and cameras.
