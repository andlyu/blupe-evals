# Add your arm

Two phases, in order. **Phase A (sim)** gets you teleoperating your arm in MuJoCo from the
headset or browser — no hardware at risk, usually in an afternoon. **Phase B (hardware)**
connects the real arm by implementing one small JSON protocol over your motor driver.
Do not skip A: every Phase B problem is easier to see when the sim twin already works.

## Phase A — sim-ready

Your arm = one `ArmSpec` entry in `scripts/arms.py` + a model folder in `assets/<arm>/`.
Copy the closest existing arm (`yam` = 6-DOF single, `so101` = 5-DOF + modeled gripper)
and work through:

1. **Get the model files: MJCF + URDF, consistent with each other.** Same link names, same
   joint names in both — this is the framework's headline rule and the single most common
   integration failure (see `docs/refs/xrobotoolkit/teleop-integration.md`, including the
   war story of how the YAM violated it). If you only have one format, GENERATE the other
   from it (one source of truth) rather than exporting both from CAD separately.
   Mesh paths must be relative and resolvable; every link needs its mesh.
2. **Build the scene** (`assets/<arm>/scene.xml`): include your arm MJCF, add position
   actuators for the arm joints, a **`home` keyframe** (a safe, natural pose — it is the
   anchor for everything: GO_HOME, policy waypoints, IK regularization), one free-floating
   **mocap body per controlled hand** named `right_target` (and `left_target` if bimanual)
   for the teleop target visual, and `<visual><global offwidth="1920" offheight="720"/>`
   (the headset canvas renders double-wide).
3. **Fill the `ArmSpec`**: `dof` = arm joints only (gripper EXCLUDED — gripper travels as a
   separate 0..1 value end to end); `ee_link` = the URDF frame the IK tracks; `ee_body` =
   the MJCF body for FK/HUD (often the same name); `gripper_joint` if your sim models one;
   `max_vel` = the joint-speed cap that bounds teleop, policies, and homing alike.
4. **Check it** (until the `--check` validator ships, by hand):
   - placo loads the URDF: `placo.RobotWrapper("assets/<arm>/<arm>.urdf")`
   - MuJoCo loads the scene and the `home` keyframe applies
   - the IK reaches a pose near home (run `scripts/policies/gripper_forward.py`-style probe)
5. **Live sim teleop**: run the eval with `--arm <name> --cameras none`. Drive it from the
   headset, or hardware-free from the browser console (`scripts/preview_server.py`,
   `http://<host>:8810/` — arrows/Enter drive the menu). Clutch, move, GO_HOME. ✅ sim-ready.
   You can already run the full eval loop here: policies, trials, verdicts, reports.

## Phase B — hardware-ready

The stack never talks to your motor driver. It talks to a **serve**: a small TCP server
YOU run next to the arm, which owns the hardware and the safety. Reference implementation:
`YAM_control/yam_real_serve.py` (~200 lines, i2rt/CAN). Implement the same protocol over
your driver (for SO-101: the lerobot `so101_follower` driver; gripper 0..100 → divide by 100).

### The wire protocol (newline-delimited JSON over TCP, default :5599)

```
server -> client, once on connect:   {"start_joints": [q1..qN]}     # no-jump seed
client -> server, ~50 Hz:            {"q": [q1..qN], "g": 0..1}     # joint targets + gripper
client -> server, optional field:    {"q": ..., "t": <any>}         # if present, echo after APPLY:
server -> client:                    {"ack": <t>}                   #   -> sender-side RTT probe
client -> server, on quit:           {"shutdown": true}             # -> cut motor torque
```

### The safety contract (non-negotiable, lives in YOUR serve)

These are robot-side on purpose — no network or operator bug can bypass them:
1. **Velocity-clamp every command** (rad/s cap) — the backstop even if the operator stack
   misbehaves.
2. **Hold the last pose on disconnect** — a dropped link must freeze the arm, not drop it.
3. **Torque off on `shutdown` and on your own exit** (Ctrl-C, crash handlers).
4. Use `start_joints` honestly: report the arm's REAL current joints so the client ramps
   from where the arm actually is (this is what prevents the first-command jump).

### Wire it into the fleet

1. In the fleet UI (ask us, or your admin): **Add arm** → you get a token + an install
   one-liner. Run it on the arm's computer: that's the relay agent (`relay/relay.py robot`),
   stdlib-only, dials OUT (no ports to open). Your arm's card appears and flips online.
2. Point the agent's `--serve-cmd` at your serve and `--camera-cmd` at
   `YAM_control/camera_relay.py --devices ...` (any V4L2/UVC cameras). Now the UI's
   **Turn ON / Turn OFF / Check** and camera view work for your arm, and **Check** runs the
   preflight that tells you exactly which motor/camera/bus is unhappy.
3. Operator side connects with `--serve-host/--serve-port` pointed at the relay's local
   mapping (or LAN-direct when you're next to the robot — lower latency).
4. First contact: arm powered, **Turn ON** from the UI, headset **CONNECT** — the HUD shows
   `ROBOT OFF` until the `start_joints` handshake arrives, so you always know whether you're
   commanding metal. Keep a hand on the e-stop for the first session.

## What you DON'T have to touch

The headset transport, the eval state machine, recording/judging/reports, fleet
management, and all the operator UX are arm-agnostic — they see only joint vectors, a 0..1
gripper, and your cameras. If Phase A works in sim and your serve honors the contract,
everything else (trials, verdict modal, report.html, link/unlink) works unchanged.
