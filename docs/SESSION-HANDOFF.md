# Session handoff — remote teleop via cloud relay (2026-06-11)

Everything below was built and verified in one session (commits `96d24f1..e681504`). The system
went from "everything on the Orin" to a **three-site production-shaped topology** with a cloud
relay and fleet UI. One issue is open (headset video — see "Open issues #1").

## Architecture as deployed (live right now)

```
OPERATOR (this Mac + Quest, same Wi-Fi)        CLOUD                ROBOT SITE (Orin + YAM)
Quest ──input :63901──► Docker xr-bridge       GCP e2-micro         agent (dials OUT)
Quest ◄─video :12345─── eval_yam_vr.py         35.185.232.107       ├─ yam_real_serve :5599
eval ──► localhost:15599 (joints) ──┐          :8443 relay          │  (vel clamp·hold·torque-off)
eval ──► localhost:18089 (cameras) ─┴─ operator client ──► relay ◄──┴─ camera_relay :8089 (MJPEG)
                                               :8080 fleet UI
```

- **Quest never leaves the operator's Wi-Fi** (its video port is inbound-only; that's why).
- **Both sites dial OUT to the relay** — no VPN, no port-forwards; customer-shaped.
- **Safety is robot-side** in the serve: 0.6 rad/s clamp, hold-on-disconnect, torque-off on
  shutdown; Damiao ~400 ms watchdog as last resort. Transport changes never touch this.

## Endpoints & credentials

| Thing | Where |
|---|---|
| Relay VM | GCP project `blupe-relay-13546`, instance `blupe-relay`, `us-west1-a` (free e2-micro). `gcloud compute ssh blupe-relay --zone=us-west1-a` |
| Relay data port | `35.185.232.107:8443` (plain TCP; TLS = open item) |
| Fleet UI | `http://35.185.232.107:8080/?token=<RELAY_ADMIN_TOKEN>` |
| Tokens | Mac: `/tmp/relay_token` (robot `yam-1`), `/tmp/relay_admin_token` (UI). Canonical copy: `Environment=` line in `/etc/systemd/system/blupe-relay.service` on the VM. `/tmp` dies on reboot — recover from the VM or regenerate everywhere. |
| Quest | `192.168.0.30` (Wi-Fi; can change). Mac `192.168.0.190`. Orin `andrew@192.168.0.185`. |

## How to start each piece

**Relay VM** — systemd `blupe-relay`, auto-restarts; nothing to do. Redeploy:
`gcloud compute scp relay/relay.py blupe-relay:/tmp/relay.py --zone=us-west1-a` then
`sudo mv + systemctl restart blupe-relay`.

**Mac (run from repo root):**
```bash
docker start xr-bridge || docker run -d --rm --name xr-bridge -p 63901:63901 -p 8765:8765 xr-bridge
.venv/bin/python relay/relay.py operator --relay 35.185.232.107:8443 --robot yam-1 --token $(cat /tmp/relay_token) &
XR_INPUT=bridge .venv/bin/python scripts/eval_yam_vr.py --quest-ip 192.168.0.30 \
  --serve-host 127.0.0.1 --serve-port 15599 \
  --cameras http://127.0.0.1:18089/0 http://127.0.0.1:18089/2     # logs: /tmp/eval_live.log
```

**Orin** (agent is the only must; serve/cameras can be started from the fleet UI):
```bash
ssh andrew@192.168.0.185
setsid nohup ~/miniforge3/envs/xr/bin/python ~/blupe-evals/relay/relay.py robot \
  --relay 35.185.232.107:8443 --robot yam-1 --token <robot token> \
  > /tmp/relay_agent.log 2>&1 < /dev/null &
# CAN after every reboot/replug (sudo): bash ~/blupe-evals/YAM_control/setup_can.sh
```

**Headset:** Network panel → `192.168.0.190` → Controller+Send ON. Remote Vision → source
`192.168.0.190` → LISTEN. Menu: stick L/R + click; A=TELEOP X=POLICY B=GO_HOME Y=QUIT;
CONNECT toggles real-arm follow; VIEW toggles cameras↔sim; trigger = gripper.

**Operate-the-arm sequence:** power arm → fleet UI **Turn ON** (runs preflight: CAN, motors
via `start_joints` handshake, cameras; starts what's down) → headset LISTEN + CONNECT → drive.
**Turn OFF** kills serve + guaranteed torque-off (`turn_off.py`).

## What's verified end-to-end ✅

- Quest poses → Docker PC Service/SDK → bridge → native eval (~44 Hz loop) — the input seam
  (`scripts/xrobotoolkit_sdk.py`, `XR_INPUT=sdk|bridge|stub`; stub = full headless session test).
- Sim teleop from the headset with Mac-side processing; menu/HUD; VIEW toggle.
- Real YAM driven from the headset (joints crossed Mac→Orin; serve clamped; hold-on-drop seen live).
- Cloud relay path: joints handshake + ~26 fps MJPEG through `35.185.232.107` (≈ LAN fps).
- Fleet UI: status, **Check** (preflight on real robot), **Turn ON** (headset-ready semantics),
  **Turn OFF** (step-wise honest reporting, survives unpowered arm).

## Open issues / next subtasks

1. **Headset video black (ACTIVE).** Sender connects to Quest:12345 and streams (verified), port
   open, frames flowing, no VIEW mix-up; app restart did NOT fix. Ruled out: relay (26 fps to
   Mac), camera relay, encoder running. Next steps: (a) check Remote Vision **Video Source
   dropdown** (resets; must match what worked before); (b) build the **fleet-UI camera view**
   (below) to prove robot→browser path; (c) add frame/keyframe send logging to `VideoStreamer`;
   (d) try the known-good standalone `scripts/orin/sim_video_sender.py` FROM THE MAC at the
   Quest to isolate eval-encoder vs app; (e) headset reboot. Suspect list: app's decoder/panel
   state, source-IP filtering in the app, encoder SPS timing after many reconnects.
2. **Fleet UI camera view** ("Turn ON lets us view cameras"). Design ready: relay gains
   `Relay.open_channel(robot_id, port)` (reuse pending/open machinery), HTTP route
   `/cam/<robot>/<idx>?token=` that opens a channel to robot :8089, writes a raw
   `GET /<idx> HTTP/1.1` and splices the upstream response verbatim to the browser
   (multipart MJPEG renders in an `<img>`). UI: per-arm "Show cameras" button injecting the
   `<img>`; IMPORTANT: the 5 s `refresh()` re-renders innerHTML — restructure so camera `<img>`
   elements are NOT re-created each refresh (update text nodes only).
3. **Hardening:** systemd units on the Orin (agent + camera relay; serve stays manual on
   purpose), TLS on the relay, tokens out of `/tmp` into proper config, Orin on Ethernet.
4. **Task #7 remainder:** full real-arm run with Turn ON on a powered arm; M2 polish.

## Hard-won gotchas (this session — don't relearn)

- **`pkill -f` over ssh kills the ssh session itself** when the same compound command also
  *mentions* the pattern (the remote shell's cmdline matches). Kill and start in SEPARATE ssh
  sessions. Likewise `pgrep -f name` self-matches; use `pgrep -f "name[.]py"`.
- **Remote background processes**: `ssh host 'setsid nohup cmd > log 2>&1 < /dev/null &'` —
  without setsid + stdin redirect the child dies with the ssh session (exit 255, empty log).
- **The Quest kills ALL its sockets on sleep** (take headset off → video LISTEN + service conn
  die). Re-enter IP / re-LISTEN after every doze; disable Auto-Sleep (Settings→Power) or tape
  the proximity sensor for long sessions.
- **i2rt swallows SIGINT** — a serve can't be stopped with INT remotely; `kill -9` + run
  `turn_off.py` (its own bus connection) for guaranteed torque-off. `turn_off` TIMES OUT if the
  arm is unpowered — that's fine, report it (the fleet agent does).
- **cv2 is in conda `xr`, NOT in `~/i2rt/.venv`** on the Orin — camera scripts use
  `~/miniforge3/envs/xr/bin/python`; serve uses `~/i2rt/.venv/bin/python`.
- `pip`-installed pybind11 in Docker needs `CMAKE_PREFIX_PATH=$(python3 -m pybind11 --cmakedir)`.
- `docker build | tail` eats the exit code — never pipe the build.
- macOS arm64: placo/mujoco/av wheels all fine; MuJoCo offscreen needs NO `MUJOCO_GL`; the
  cask `google-cloud-sdk` symlinks gcloud into `/opt/homebrew/bin`.
- `patch_framework.py` never existed (stale ORIN-SETUP.md): superseded by generating a
  consistent `yam.urdf` from the MJCF (`scripts/gen_yam_urdf.py`). Stock framework works.
- Background subagents get auto-denied permissions — run privileged work inline.

## Commits this session (oldest first)

`96d24f1` topology decision + diagrams + bridge protocol · `cbe1b8d` YAM_control commit ·
`1b89b38` input seam + xr-bridge container + --serve-host · `2c74fb1` camera relay + transport
decision · `50f66ac` relay v1 · `c20a41b` fleet UI + VIEW toggle · `e681504` headset-ready
Turn ON. Diagrams: `docs/remote-topology.png`, `docs/what-happens-where.png`.
Protocol: `docs/XR-INPUT-BRIDGE.md`. Plan: `docs/PLAN.md` ("Remote topology", "Customer
transport" sections).
