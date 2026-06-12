# Integrate your hardware

You have a **supported arm** (see the embodiment registry in `scripts/arms.py` — e.g. YAM,
SO-101) and want to drive YOUR physical unit. If your arm model isn't in the registry yet,
start at [add-an-embodiment.md](add-an-embodiment.md) instead — that's the bigger job of
teaching the stack a new arm type; this page is about connecting a unit of a known type.

**What you need:** the arm + a computer physically attached to it (any Linux box; we use a
Jetson Orin), one or two USB cameras pointed at the workspace, an operator computer, and a
Quest with the XRoboToolkit client app.

## 1. Prove the stack in sim first (15 min, no hardware)

On the operator computer: install (`pip install -r requirements.txt`, build
`docker/xr-bridge`), then run the eval for your embodiment with `--cameras none`.
Drive the sim twin from the headset (or the browser console at `:8810` — no Quest needed).
If sim teleop works, every later problem is hardware-side by construction. Don't skip this:
it converts "nothing works" into "only the robot link can be wrong".

## 2. Start the serve on the robot computer

The **serve** is the small TCP server that owns your motors and the safety rules
(velocity clamp, hold-on-disconnect, torque-off). Supported embodiments ship a reference
serve — YAM: `YAM_control/yam_real_serve.py` (i2rt/CAN). Run it next to the arm and
verify the handshake: connecting to `:5599` must immediately yield one line,
`{"start_joints": [...]}`, with the arm's REAL current joints. No handshake = the serve
can't talk to the motors yet — fix that before anything else (power, bus, driver).

If your unit needs a custom serve (different driver/firmware), the wire protocol and the
non-negotiable safety contract are specified in
[add-an-embodiment.md](add-an-embodiment.md#the-serve-wire-protocol) — it's ~200 lines.

## 3. Join the fleet (one click + one paste)

In the fleet UI, an admin clicks **Add arm** → you receive an install one-liner.
Run it on the robot computer: that's the relay agent (`relay/relay.py robot`) — stdlib-only
Python, dials OUT to the relay (no inbound ports, no VPN). Point its `--serve-cmd` at your
serve and `--camera-cmd` at `YAM_control/camera_relay.py --devices <your cams>`.
Your arm's card appears in the UI (offline → online), and access is granted/revoked by
the admin **link/unlink**ing your account to the arm.

## 4. First contact

1. Arm powered, clear workspace, hand near the e-stop.
2. Fleet UI → **Check**: runs the preflight (bus, each motor, cameras) and names exactly
   what's unhappy. Then **Turn ON**.
3. Headset → **CONNECT**. The HUD shows `ROBOT OFF` until the serve's handshake arrives —
   you always know whether you're commanding metal. The arm ramps gently from its true
   pose (that's what `start_joints` is for); it never jumps.
4. Teleop. Then run the eval loop for real: MARK waypoints, POLICY trials, verdicts,
   `eval_report.py render`.

## The exact interface contract

Your machines never call our cloud API. You run our **agent** (`relay/relay.py robot`,
stdlib Python) on the robot computer; it makes ONE outbound TCP connection to the relay
(`<relay-host>:8443`) and everything else flows through it, both directions. Nothing on
your side is exposed to the internet.

### What YOU provide (local services on the robot computer, localhost-only)

| Service | Where | Interface |
|---|---|---|
| **Serve** (owns motors + safety) | TCP `:5599` | newline-JSON: sends `{"start_joints":[...]}` once on connect; accepts `{"q":[...], "g":0..1}` at ~50 Hz and `{"shutdown":true}`; echoes `{"ack":t}` if a command carries `"t"`. Ships for supported drivers; custom = ~200 lines ([protocol + safety contract](add-an-embodiment.md#the-serve-wire-protocol)). |
| **Cameras** | HTTP `:8089` | `GET /<idx>` → multipart MJPEG. Just run `YAM_control/camera_relay.py --devices 0 2` for any UVC cameras; custom cameras only need to mimic that one GET. |
| **Lifecycle commands** | agent flags | three shell commands the fleet buttons run on your box: `--serve-cmd` (Turn ON), `--turnoff-cmd` (Turn OFF / guaranteed torque-off), `--camera-cmd`. |

### What flows OUT from your site (over the agent's single outbound connection)

- registration: `{"role":"robot","robot":"<id>","token":"<your arm token>"}`
- results of fleet commands (preflight / on / off) as JSON
- raw bytes of any open channels: serve replies, camera MJPEG

### What the app sends BACK down that connection

- channel-open requests (`{"open": <port>, "conn": id}`) — only for ports you allowed
- fleet commands (`{"cmd": "preflight"|"arm_on"|"arm_off", "req": id}`)
- the operator's live joint/gripper stream, spliced into your `:5599`

### What WE hand you (the onboarding packet)

1. an **arm token** + the agent install one-liner (from the fleet UI "Add arm")
2. a **customer token** → your fleet UI URL `http://<relay-host>:8080/?token=<yours>`,
   showing only your linked arms: status, Turn ON/OFF, Check (per-motor preflight),
   live cameras

Note: transport is currently plain TCP — treat tokens like passwords; TLS is on the
roadmap before broader rollout.

## When something is off

Check `.claude/skills/small-errors/SKILL.md` (known papercuts: dead stick, doze, stale
IPs) before debugging, then the infra playbooks. Staleness questions are answered by the
timestamp burned into every camera frame — read the clock in the image.
