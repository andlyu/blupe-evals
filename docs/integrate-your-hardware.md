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
serve — YAM: `YAM_control/yam_real_serve.py` (i2rt/CAN). Start it, start the cameras,
then **verify your whole side with one command** (stdlib-only, safe — reads the handshake,
never moves the arm):

```bash
python3 scripts/check_robot_setup.py --relay 35.203.190.87:8443
```

PASS/FAIL lines — serve handshake, camera frames, and outbound relay reachability — each
FAIL printing its exact fix, in dependency order. ALL GREEN means the agent one-liner is
the only thing left. You can still pass `--robot <id> --token <customer-token>` to verify
that an already-linked customer token can reach the arm. (You can also verify end to end from anywhere:
`curl "http://35.203.190.87/api/status?token=<your-customer-token>"` returns your
arms' live state as JSON.)

If your unit needs a custom serve (different driver/firmware), the complete contract —
wire messages, units, joint order, timing tolerances, the safety rules, and a skeleton —
is **[serve-protocol.md](serve-protocol.md)**; implementations run ~200 lines.

## 3. Join the fleet (one click + one paste)

In the fleet UI, an admin clicks **Add arm** → you receive an install one-liner.
Run it on the robot computer: that's the relay agent (`relay/relay.py robot`) — stdlib-only
Python, dials OUT to the relay with just the arm id (no robot token, no inbound ports, no
VPN). Point its `--serve-cmd` at your
serve and `--camera-cmd` at `YAM_control/camera_relay.py --devices <your cams>`.
Your arm's card appears in the UI (offline → online), and access is granted/revoked by
the admin **link/unlink**ing your account to the arm.

Use `env VAR=value command`, not `VAR=value command`, because the relay agent executes
lifecycle commands as `exec <cmd>`.

Correct:

```bash
--serve-cmd "env PYTHONPATH=/home/andrew/i2rt:/home/andrew/lerobot-robot-yam \
  /home/andrew/i2rt/.venv/bin/python -m lerobot_robot_yam.yam_serve --channel can0"
```

Incorrect:

```bash
--serve-cmd "PYTHONPATH=/home/andrew/i2rt:/home/andrew/lerobot-robot-yam \
  /home/andrew/i2rt/.venv/bin/python -m lerobot_robot_yam.yam_serve --channel can0"
```

### Lifecycle sanity check

Before you put a person in the headset, prove the relay lifecycle:

1. Fleet UI → **Turn ON**. Expect `{"result":"ON"}` or `{"result":"already on"}` and a
   `start_joints` array. On the robot computer, `:5599` should be listening.
2. Fleet UI → **Turn OFF**. Expect the serve to receive `{"shutdown":true}` and run its own
   torque-off path. On the robot computer, `:5599` must stop listening while the relay agent
   remains online.
3. Fleet UI → **Turn ON** again. Expect a fresh `start_joints` handshake and `:5599`
   listening again.

For a quick robot-side check after each button press:

```bash
ss -ltnp | grep ':5599' || echo 'serve is off'
tail -50 /tmp/relay_agent.log
tail -50 /tmp/serve_managed.log
```

- ON: `:5599` listening, `start_joints` returned.
- OFF: serve receives `{"shutdown":true}`, torque-off path runs, `:5599` stops listening.
- Camera relay may remain up on `:8089`.
- Robot agent must remain online after OFF.
- ON/OFF commands must be serialized; OFF must not race with another ON.

## 4. Configure policy execution

Before POLICY can run, decide where the policy lives and how it is invoked. The default
customer path is robot-side execution: the relay sends a `run_policy` command to the robot
agent, and the robot computer runs the policy locally against its own serve on
`127.0.0.1:5599`.

Ask the user for:

1. **Policy entrypoint**
   - Python callable: `scripts/policies/pick_place.py:run`
   - Python module: `my_pkg.my_policy:run`
   - Shell command: `python run_policy.py --checkpoint ...`

2. **Current run command**
   - exact command they use today
   - working directory
   - Python env / conda env / venv / container

3. **Policy interface**
   - preferred: `run(robot, stop)`
   - otherwise: describe inputs/outputs so we can wrap it

4. **Runtime dependencies**
   - checkpoints, configs, assets
   - camera requirements
   - network/API requirements
   - max runtime

5. **Stop behavior**
   - how to interrupt cleanly
   - what command or signal stops it
   - whether it returns a verdict/result

### Robot-side policy model

The relay UI does not stream policy actions itself. It sends a lifecycle command:

```json
{"cmd": "run_policy", "policy": "<policy-id>", "req": 123}
```

The UI only sends a policy id. It does not accept an arbitrary shell command from the
browser. Each policy id must be configured when the robot agent starts:

```bash
--policy <policy-id>='<exact shell command>'
```

The command should `cd` to the repo/workspace first, then invoke the right Python/env. If
the command needs environment variables, put them inside the command with `env VAR=value`.
Keep the whole command in quotes so spaces and `&&` remain part of that single policy
mapping.

The robot agent:

1. verifies the YAM serve is ON
2. starts the configured policy runner on the robot computer
3. passes a local robot adapter connected to `127.0.0.1:5599`
4. streams/logs status back to the relay
5. supports stop/cancel by calling the policy `stop()` hook or killing the process

Example YAM policy flags:

```bash
--policy noop='cd /home/andrew/blupe-evals && \
  /home/andrew/miniforge3/envs/xr/bin/python scripts/run_policy.py \
  scripts/policies/noop.py:run \
  --serve-host 127.0.0.1 --serve-port 5599' \
--policy pick_place='cd /home/andrew/blupe-evals && \
  /home/andrew/miniforge3/envs/xr/bin/python scripts/run_policy.py \
  scripts/policies/pick_place.py:run \
  --serve-host 127.0.0.1 --serve-port 5599' \
--default-policy noop
```

Start with `noop` from the fleet UI. It exercises the full robot-side `run_policy` path
without commanding motion. Switch the default to a task policy only after the no-op path
starts, logs, and stops correctly.

Full agent shape:

```bash
python /home/andrew/blupe-evals/relay/relay.py robot \
  --relay 35.203.190.87:8443 \
  --robot yam-1 \
  --serve-cmd "env PYTHONPATH=/home/andrew/i2rt:/home/andrew/lerobot-robot-yam \
    /home/andrew/i2rt/.venv/bin/python -m lerobot_robot_yam.yam_serve \
    --channel can0 --host 127.0.0.1" \
  --policy noop='cd /home/andrew/blupe-evals && \
    /home/andrew/miniforge3/envs/xr/bin/python scripts/run_policy.py \
    scripts/policies/noop.py:run \
    --serve-host 127.0.0.1 --serve-port 5599' \
  --policy pick_place='cd /home/andrew/blupe-evals && \
    /home/andrew/miniforge3/envs/xr/bin/python scripts/run_policy.py \
    scripts/policies/pick_place.py:run \
    --serve-host 127.0.0.1 --serve-port 5599' \
  --default-policy noop \
  --policy-log /tmp/policy_managed.log
```

Validation order:

1. Fleet UI → **Turn ON**.
2. Fleet UI → **Run policy** with `noop`.
3. On the robot computer, verify `/tmp/policy_managed.log` shows the runner connected,
   printed joints, and completed.
4. Fleet UI → **Stop policy** should return `no policy running` after `noop` completes, or
   stop the active process if a longer policy is running.
5. Only then run a motion policy such as `pick_place`.

### Operator-side eval policy path

The older eval path runs policy inside `mac_quest_bridge.py` on the operator machine:

```bash
--policy scripts/policies/pick_place.py:run
```

That is useful for sim, headset trials, and reports, but it is not the default fleet
policy execution path. For fleet POLICY, prefer robot-side execution through `run_policy`.

## 5. First contact

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
(`35.203.190.87:8443`) and everything else flows through it, both directions. Nothing on
your side is exposed to the internet.

### What YOU provide (local services on the robot computer, localhost-only)

| Service | Where | Interface |
|---|---|---|
| **Serve** (owns motors + safety) | TCP `:5599` | newline-JSON: sends `{"start_joints":[...]}` once on connect; accepts `{"q":[...], "g":0..1}` at ~50 Hz and `{"shutdown":true}`; echoes `{"ack":t}` if a command carries `"t"`. Ships for supported drivers; custom = ~200 lines (full spec: [serve-protocol.md](serve-protocol.md)). |
| **Cameras** | HTTP `:8089` | `GET /<idx>` → multipart MJPEG. Just run `YAM_control/camera_relay.py --devices 0 2` for any UVC cameras; custom cameras only need to mimic that one GET. |
| **Lifecycle commands** | agent flags | three shell commands the fleet buttons run on your box: `--serve-cmd` (Turn ON), `--turnoff-cmd` (Turn OFF / guaranteed torque-off), `--camera-cmd`. |

### What flows OUT from your site (over the agent's single outbound connection)

- registration: `{"role":"robot","robot":"<id>"}`
- results of fleet commands (preflight / on / off) as JSON
- raw bytes of any open channels: serve replies, camera MJPEG

### What the app sends BACK down that connection

- channel-open requests (`{"open": <port>, "conn": id}`) — only for ports you allowed
- fleet commands (`{"cmd": "preflight"|"arm_on"|"arm_off", "req": id}`)
- the operator's live joint/gripper stream, spliced into your `:5599`

### What WE hand you (the onboarding packet)

1. the agent install one-liner (from the fleet UI "Add arm")
2. a **customer token** → your fleet UI URL `http://35.203.190.87/?token=<yours>`,
   showing only your linked arms: status, Turn ON/OFF, Check (per-motor preflight),
   live cameras

Note: transport is currently plain TCP — treat customer/admin tokens like passwords; TLS is
on the roadmap before broader rollout.

## When something is off

Check `.claude/skills/small-errors/SKILL.md` (known papercuts: dead stick, doze, stale
IPs) before debugging, then the infra playbooks. Staleness questions are answered by the
timestamp burned into every camera frame — read the clock in the image.
