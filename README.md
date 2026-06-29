# blupe-evals

**Teleoperate robot arms from a Quest headset — anywhere — and turn every policy run into
a judged, shareable evaluation report.**

One operator computer (the headset talks to it), one robot computer (owns the arm + safety),
optionally a cloud relay between them so neither site needs a VPN or open ports. Sim twin
first, real hardware second; the whole eval loop (trials, verdicts, success-rate reports)
works on both.

```
OPERATOR (laptop + Quest, same Wi-Fi)         CLOUD (optional)        ROBOT SITE
Quest ──input──► xr-bridge (docker)           relay + fleet UI        agent (dials OUT)
Quest ◄─video─── mac_quest_bridge.py  ◄─ cameras ──── relay channels ────► camera_relay
eval ──joints──► serve (robot-side safety: clamp · hold · torque-off) ─► your arm
```

## I want to…

| …do this | read |
|---|---|
| **Connect my arm** (a supported model: YAM, SO-101) | [docs/integrate-your-hardware.md](docs/integrate-your-hardware.md) |
| **Run the current SO101 + MolmoAct + SAM eval stack** | [docs/SO101-EVAL-STACK-OVERVIEW.md](docs/SO101-EVAL-STACK-OVERVIEW.md) |
| **Add a new arm model** (a new embodiment, sim assets + driver) | [docs/add-an-embodiment.md](docs/add-an-embodiment.md) |
| Run evals and get a report | below: *The eval loop* |
| Operate the fleet / onboard a customer | fleet UI (`relay.py serve`): Add arm, Add customer, link/unlink |
| Fix a known papercut fast | `.claude/skills/small-errors/SKILL.md` |
| Understand the live deployment | [docs/SESSION-HANDOFF.md](docs/SESSION-HANDOFF.md) |

Current SO101 eval stack:
`scripts/start_so101_eval_stack.sh` starts UI/cameras/GPU tunnels/policy/SAM,
`scripts/check_so101_eval_stack.sh` verifies the endpoints, and
`scripts/stop_so101_eval_stack.sh` stops them.

## Quickstart (operator computer, sim — no robot needed)

```bash
git clone https://github.com/andlyu/blupe-evals && cd blupe-evals
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
docker build -t xr-bridge docker/xr-bridge          # the XR input appliance
docker run -d --rm --name xr-bridge -p 63901:63901 -p 8765:8765 xr-bridge
.venv/bin/python scripts/xrtk_announce.py &          # headset discovers you: no IP typing
XR_INPUT=bridge .venv/bin/python scripts/mac_quest_bridge.py --quest-ip <quest-ip> --cameras none
```

Headset (XRoboToolkit Quest app, sideloaded once): **Network panel → tap the popped-up IP →
Controller + Send ON**; **Camera panel → ZEDMINI → Listen → this computer's IP → Confirm**.
You're teleoperating the sim arm. No headset handy? `scripts/preview_server.py` mirrors the
exact operator view to a browser at `:8810` with keyboard control.

## Hardware setup (let Claude do it)

On the **robot computer** (e.g. the Jetson/Orin wired to the YAM), open Claude Code and paste:

> SSH into the Jetson at andrew@192.168.0.185. Clone git@github.com:andlyu/blupe-evals.git
> into /home/andrew/blupe-evals. Read docs/integrate-your-hardware.md first. Set up the
> relay for YAM robot yam-1 using the VM relay service at 35.203.190.87:8443.

[docs/integrate-your-hardware.md](docs/integrate-your-hardware.md) is the integration flow
Claude follows — bring up the CAN bus, start a serve that speaks the
[serve protocol](docs/serve-protocol.md), run `scripts/check_robot_setup.py` (PASS/FAIL
doctor), and join the fleet through `35.203.190.87:8443`. The prompt above points the serve at
the **lerobot** YAM driver; the doc's own reference serve is `YAM_control/yam_real_serve.py`
(i2rt/CAN) — either satisfies the same contract. Orin specifics:
[docs/ORIN-SETUP.md](docs/ORIN-SETUP.md).

## The eval loop

Start the eval with a task: `--task red-plate-pickup --stages reach grasp lift place
--policy scripts/policies/pick_place.py:run`. Then, from inside the headset:

1. **TELEOP** to the pick spot → **MARK A**; drop spot → **MARK B** (one-time per scene).
2. **POLICY** (X) — each run auto-records a trial (video + event timeline) and ends with a
   SUCCESS/FAIL modal.
3. Afterwards: `eval_report.py serve` to score failures (stage + 0–1 progress),
   `eval_report.py render` → a self-contained `report.html` with success rate, mean
   progress, failure-by-stage histogram, and every trial's video.

## Pieces

```
scripts/mac_quest_bridge.py     the operator process: state machine, headset video, trials, HUD
scripts/arms.py            embodiment registry (ArmSpec: assets, EE frames, dof, caps)
scripts/eval_report.py     judge UI + report renderer       scripts/policies/   scripted policies
scripts/stereo_sender.py   headset video transport          scripts/xrtk_announce.py  IP popup
relay/relay.py             cloud relay + fleet UI + robot agent + operator client (stdlib-only)
YAM_control/               YAM reference serve (safety contract) + camera relay + CAN tools
docker/xr-bridge/          XR input appliance (vendored PC service + sdk bindings)
assets/<arm>/              per-embodiment models (MJCF + URDF, consistent names)
docs/refs/                 vendored upstream docs (read before integrating!)
```

Python 3.10 · macOS arm64 + Linux (robot side) · safety lives robot-side in the serve:
velocity clamp, hold-on-disconnect, torque-off — transport changes can never touch it.
