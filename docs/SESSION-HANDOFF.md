# Session handoff — remote teleop via cloud relay (2026-06-11)

Everything below was built and verified in one session (commits `96d24f1..e681504`). The system
went from "everything on the Orin" to a **three-site production-shaped topology** with a cloud
relay and fleet UI. One issue is open (headset video — see "Open issues #1").

## ⚡ Session pass-off — 2026-06-11 end of day (read this first if you're a new session)

**Where things stand:** BACK HOME. Mac `192.168.0.231` (NOTE: the router re-leased — it used
to be `.190`; get a DHCP reservation), Quest `192.168.0.30`, Orin `192.168.0.185`. Everything
LAN-direct (no cloud hop at home — infra ISSUE-004): eval cameras/serve point straight at the
Orin. Running on the Mac: `mac_quest_bridge.py` (task `red-plate-pickup`, log `/tmp/eval_live.log`),
`xrtk_announce.py`, relay operator, docker `xr-bridge`, `eval_report.py serve` (:7799).
**The eval pipeline ran its first REAL session today: 7 trials in `runs/2026-06-11_red-plate-
pickup/`, all judged in-VR, 43% success.** Failures still need stage+score via the judge UI.

**OPEN — hardware:** the arm's **motor 3 is silent on CAN** (serve exits rc=1: "fail to
communicate with the motor 3"); bus itself is clean (ERROR-ACTIVE, 0 errors), motors 1-2
answer. Next: power-cycle the arm → fleet UI Check; if still dead, reseat joint-3
connectors (CAN daisy-chain + power). Until fixed, CONNECT shows "ROBOT OFF".

**OPEN — process:** NOTHING IS COMMITTED (working tree spans two threads' work since
`1d0a536`). Other known gaps: stub timeline never enters POLICY (verdict modal untested
headlessly), mono video path lacks every robustness fix (legacy), TLS/tokens/systemd
hardening backlog, true stereo-3D needs a calibrated camera pair + per-eye HUD (transport
supports it; docs/refs/xrobotoolkit/stereo-vision.md), camera-channel drop root cause behind
the grabber's auto-reopen never chased (watch `[camera] ... reopening` frequency).

**The headset ritual (after killing the app / doze):** app open → Network panel → tap popup
IP → Controller+Send ON (controller lives here; a red NO-CONTROLLER-INPUT banner in the HUD
means redo this) → Camera panel → ZEDMINI → Listen → Confirm (video lives here; IP is saved
per network). Trials: just run POLICY — recording is automatic, verdict modal after each run.

**Skills grown this session — check them BEFORE debugging:** `small-errors` (papercut
symptom→fix: dead stick, doze, stale IPs, no popup, B collision), `infra` (issue log now
ISSUE-001..004 + "probe trap" rule: fresh connections lie about long-lived stream staleness;
test sustained), `logging` ([lat] conventions; from the parallel thread).

**Shipped this session** (all UNCOMMITTED — see inventory below):
- **Onboarding doc suite + repo is now CLONE-READY (2026-06-12, all pushed):** README
  rewritten as the product front door ("I want to…" table, sim quickstart), pinned
  `requirements.txt` (incl. the previously-invisible editable `xrobotoolkit_teleop` →
  git-pinned @79e5cb8), `docs/integrate-your-hardware.md` (connect YOUR unit: interface
  contract table, onboarding packet), `docs/add-an-embodiment.md` (new arm MODEL),
  `docs/serve-protocol.md` (complete serve spec: messages/units/timing/safety/skeleton),
  `scripts/check_robot_setup.py` (stdlib robot-side doctor: serve/cameras/relay/token,
  each FAIL prints its fix — tested against prod relay). Everything committed+pushed
  through `ad264d9`. Next product steps: SO-101 reference serve (lerobot), single-command
  "blupe-node" packaging, TLS before less-trusted customers.
- **Fleet management / customer onboarding (2026-06-12, DEPLOYED to the relay VM):** the
  relay now keeps a persisted fleet registry (`/opt/fleet.json` on the VM: robots + customers
  + links; seeded from the old RELAY_TOKENS env). Fleet UI admin card: **Add arm** (returns an
  install one-liner with no robot token; card appears offline → flips online when the agent dials
  in), **Add customer** (mints a token + scoped UI URL), per-arm **link/unlink chips**.
  Customer tokens see/control ONLY their linked arms (UI + `/api/cmd` + `/cam/` + the
  operator DATA PLANE: `auth_operator` accepts a user token only while linked — unlink
  revokes new connections instantly). Report card + `/api/fleet` are admin-only. All
  mutations live, no restarts. e2e-tested locally (grant/revoke/persistence assertions).
  WART: yam-1 and mac-1 still share one legacy token — mint fresh ones. TLS still pending
  (tokens travel plaintext) — now customer-facing, so hardening priority went UP.
- **Stereo video transport is the default** (`--video stereo`): Quest dials Mac `:13579`
  (ZEDMINI flow), double-wide canvas = both cameras side by side + ONE HUD bar, wide sim
  render on VIEW toggle. Protocol: `docs/refs/xrobotoolkit/stereo-vision.md`.
- **Robustness fixes, each root-caused live**: newest-connection-wins preemption (green-screen
  zombie, infra ISSUE-003), send-each-frame-once (3 s Quest decode backlog), SO_SNDBUF+drop
  (doze backlog), camera auto-reopen after 3 s dead (NO SIGNAL self-heals), announce
  broadcast on the interface's REAL broadcast (not /24 guess).
- **No more IP typing**: `scripts/xrtk_announce.py` replays the PC-service discovery packet
  Docker was eating (infra ISSUE-002, `docs/refs/xrobotoolkit/discovery-announce.md`).
- **Per-stage latency tracking** (lag diagnosis, e2e-verified headless): every tier prints a
  `[lat]` line with `stage=avg/max ms` over the window. Eval (every 2 s, next to `[dbg]`):
  `input_age` (XR tick age at use; bridge mode only), `ik`, `loop_busy` (>20 ms avg = can't
  hold 50 Hz), `cmd_queue` (set_target→socket write; ~8–10 ms by design, decoupled 50 Hz
  sender), `cmd_write`, `arm_rtt` — Mac-clock round trip: command write → serve APPLIES it →
  `{"ack": t}` echo (protocol grew optional `"t"` in commands; old serve = no acks, harmless),
  `cam_age` = freshest camera frame age at composite. Stereo sender (every 5 s):
  `vid_queue` (submit→encode), `vid_encode`, `vid_send`, `vid_skip`/`vid_drop` counters.
  Serve, Orin-side (every 5 s): `gap` (inter-command = network+sender jitter), `apply`.
  Clocks are NOT synced across machines — every number is single-clock (RTT trick for
  cross-machine). Helper: `LatencyStats` in `stereo_sender.py`; `xrobotoolkit_sdk.get_input_age_s()`.
  Full stage map, log-file locations, and conventions: **`.claude/skills/logging/SKILL.md`**.
- **Browser operator console + report sessions (2026-06-12):** `preview_server.py` serves a
  live mirror of the EXACT headset canvas at `http://<mac>:8810/` (MJPEG; a watcher alone
  makes the eval render — no Quest needed) with full keyboard control (arrows/Enter = menu,
  a/x/b/y = shortcuts; `/key` endpoint). Report sessions: fleet UI (`:8080`) gained
  **New report / Finish report / Status** buttons → `/api/report` → proxied over a relay
  channel to the Mac, which registers as node `mac-1` (`relay.py robot --robot mac-1
  --allow 8810`; token added to VM RELAY_TOKENS; UI hides `mac-*` from arm cards).
  New report = rotate to fresh `runs/<date>_<task>_<HHMMSS>/` + record the WHOLE operator
  view to `session.mp4` (SessionTape, wall-clock pts). Finish = stop tape + render
  `report.html`. Gotcha: relay HTTP proxies must read by Content-Length, NOT to EOF — the
  agent-side splice holds the channel open until both directions close. `setsid` does not
  exist on macOS (Orin-only recipe). Demo capture: `scripts/record_mirror.py` (standalone
  mirror→mp4), ffmpeg now installed on the Mac for side-by-side composition.
- **Multi-arm standards onboarded, SIM-FIRST (2026-06-12):** `scripts/arms.py` registry +
  `--arm` flag on the eval (`yam` default | `so101` | `yam-bimanual` | `openarm`). All four
  verified headlessly: stub session reaches TELEOP, sim renders through the browser console.
  Research vendored: `docs/refs/{so101,openarm,yam-bimanual}/INDEX.md` (drivers, models,
  gotchas, sources). Assets: `assets/so101/` (TheRobotStudio new-calib + our scene wrapper —
  EE must exist in BOTH MJCF and URDF: use body `gripper`, not `gripper_frame_link`);
  `assets/yam_bimanual/` GENERATED by `scripts/gen_bimanual_scene.py` (MjSpec prefix-attach
  of two yam.xml, left_/right_ targets, combined home key, URDF from the shared MJCF-walk
  `scripts/mjcf2urdf.py`, FK-verified < 1 mm); `assets/openarm/` from enactic/openarm_mujoco
  v2 via `scripts/gen_openarm_scene.py` — fingers WELDED (upstream nq=18/nu=16 breaks the
  eval's contiguous ctrl slicing; fingerless = 14/14 aligned, same as YAM's no-sim-gripper
  convention). Bimanual teleop config = left_hand+right_hand manipulators (left ctrl→left
  arm). REAL-hardware drivers/serves NOT wired yet (per-arm notes in arms.py + refs);
  policies still assume YAM (warned at startup). OpenArm bases sit at floor level —
  upstream pedestal.xml/cell.xml exist if we want a mounted look.
  `XR_INPUT=stub mac_quest_bridge.py --cameras none --serve-port 5599` (stub PRESSES buttons —
  fake serve only!). MuJoCo `offwidth` raised to 1920 in `assets/yam/scene.xml`.

**Eval report system v1 — BUILT & verified with real in-VR trials (this thread):**
`mac_quest_bridge.py --task <name> --stages reach grasp lift place` auto-records **one trial per
POLICY run** (no button: entering POLICY starts video+meta, the post-run SUCCESS/FAIL verdict
modal — stick l/r + click — saves the result and closes the trial; re-entering POLICY with a
verdict pending saves the old trial unjudged). Operator-view canvas →
`runs/<date>_<task>/trial_NNN/video.mp4` + `meta.json` with timestamped
state/gripper/connect/policy events; REC badge in HUD. Recordings are full-frame even when
the headset view is letterboxed (`--screen-scale`, default 0.85). HUD menu = 2x4 centered
grid; stick up/down jumps rows.
Judge: `eval_report.py serve` → http://127.0.0.1:7799 (play video, success/fail, failed
stage + 0-1 score, notes → saved into meta.json; mp4 served with Range for Safari).
Report: `eval_report.py render` → self-contained `report.html` (success rate, mean progress,
failure-by-stage histogram, per-trial videos). Scoring: success=1.0 else
(completed_stages + score)/n. NOTE: mean-progress only counts scored trials — binary in-VR
fails contribute nothing until staged+scored in the judge. Verified headlessly AND with a
real 7-trial in-VR session (videos, verdicts, event timelines all correct).
**Operator UX shipped alongside:** 2x4 centered menu grid (stick up/down = rows), `--screen-
scale 0.85` letterbox ("move the screen back"; recordings stay full-frame), red NO-CONTROLLER-
INPUT banner (liveness = head-pose jitter, bridge mode), policy-verdict modal owns the stick
while up (A/X/B/Y safety shortcuts stay live). Camera relay (Orin) got only-new-frames +
SO_SNDBUF+drop (ISSUE-004) — slow consumers get FEWER frames, never OLDER frames.
**File ownership to avoid parallel-edit conflicts:** this thread owns `scripts/mac_quest_bridge.py`,
`scripts/stereo_sender.py`, `scripts/eval_report.py` (new), `docs/SESSION-HANDOFF.md`.
Parallel threads: don't edit those; everything else is fair game.

**Uncommitted working tree** (nothing committed since `1d0a536`; spans BOTH threads' work —
coordinate before committing): modified — `mac_quest_bridge.py`, `YAM_control/camera_relay.py`
(deployed to Orin), `YAM_control/yam_real_serve.py` (ack protocol), `xrobotoolkit_sdk.py`,
`scene.xml`, `SESSION-HANDOFF.md`, `refs/xrobotoolkit/INDEX.md`,
`.claude/skills/infra/SKILL.md`, `policies/gripper_forward.py`;
new — `stereo_sender.py`, `eval_report.py`, `xrtk_announce.py`, `fake_quest_stereo.py`, `show_poses.py`,
`refs/xrobotoolkit/stereo-vision.md`, `refs/xrobotoolkit/discovery-announce.md`,
`.claude/skills/small-errors/`, `.claude/skills/logging/`, `runs/2026-06-11_red-plate-pickup/`
(7 real trial videos + metas — decide whether runs/ belongs in git or .gitignore).

## Architecture as deployed (live right now)

```
OPERATOR (this Mac + Quest, same Wi-Fi)        CLOUD                ROBOT SITE (Orin + YAM)
Quest ──input :63901──► Docker xr-bridge       GCP e2-micro         agent (dials OUT)
Quest ◄─video :12345─── mac_quest_bridge.py         35.185.232.107       ├─ yam_real_serve :5599
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
.venv/bin/python scripts/xrtk_announce.py --unicast 192.168.0.30 &   # headset IP popup, no typing
XR_INPUT=bridge .venv/bin/python scripts/mac_quest_bridge.py --quest-ip 192.168.0.30 \
  --serve-host 127.0.0.1 --serve-port 15599 \
  --cameras http://127.0.0.1:18089/0 http://127.0.0.1:18089/2     # logs: /tmp/eval_live.log
# ^ REMOTE (cloud-relay) endpoints. At home go LAN-direct (no GCP hop; infra ISSUE-004):
#   --serve-host 192.168.0.185 --serve-port 5599 \
#   --cameras http://192.168.0.185:8089/0 http://192.168.0.185:8089/2
```

**Orin** (agent is the only must; serve/cameras can be started from the fleet UI):
```bash
ssh andrew@192.168.0.185
setsid nohup ~/miniforge3/envs/xr/bin/python ~/blupe-evals/relay/relay.py robot \
  --relay 35.203.190.87:8443 --robot yam-1 \
  > /tmp/relay_agent.log 2>&1 < /dev/null &
# CAN after every reboot/replug (sudo): bash ~/blupe-evals/YAM_control/setup_can.sh
```

**Headset (no typing):** with `xrtk_announce.py` running on the Mac, launching the app pops
an IP-select dialog — tap `192.168.0.190` (that IS the Network-panel connect) → Controller+
Send ON. Video (stereo, default): Camera panel → source **ZEDMINI** (default) → Listen →
IP pre-fills after the FIRST ever entry (PlayerPrefs; exit the app cleanly once) → Confirm.
Shows both cameras side by side, ONE HUD bar across; keep the app in FLAT mode (B toggles
flat↔3D AND collides with B=GO_HOME — avoid B). Video (`--video mono` legacy): Remote
Vision → source `192.168.0.190` → LISTEN. Menu: stick L/R + click; A=TELEOP X=POLICY
B=GO_HOME Y=QUIT; CONNECT toggles real-arm follow; VIEW toggles cameras↔sim; trigger = gripper.

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

1. **Headset video "black" — SOLVED (root cause: decode backlog, not a broken stream).**
   When the Quest sleeps, its app freezes but its TCP stack keeps buffering; the sender pumps
   ~4 Mbit/s into kernel buffers on both ends. On wake the decoder plays the backlog IN ORDER →
   first "black", later **old frames**, never catching up. The stream was time-shifted, never
   broken (confirmed by user seeing stale camera frames). Workaround: re-press LISTEN (fresh
   socket = current frames). Structural fix for the video transport rework: **small SO_SNDBUF
   (~128 KB) + drop frames on backpressure** so the sender can never queue more than a fraction
   of a second; the stereo flow (Quest dials us, fresh session per panel-open) also resets this
   inherently. Measured: camera composite encodes at ~4 Mbit/s, 62 KB keyframes (content is NOT
   the issue; cv2 URL reads verified at 24–31 fps with real content).
   **Transport rework SHIPPED (2026-06-11):** stereo (ZEDMINI) flow is now the eval's default
   (`--video stereo`; `--video mono` = legacy). Quest dials us on :13579 → we stream a
   double-wide H.264 canvas back; VIEW toggle works over it (cameras: both side by side at
   full size; sim: one wide render; ONE HUD bar across the whole frame — view in the app's
   FLAT mode). Both structural fixes built into `stereo_sender.py`: SO_SNDBUF 128 KB + drop-
   frames-on-backpressure (no doze backlog), fresh session per panel-open. Headless e2e PASSED
   (`scripts/fake_quest_stereo.py` plays the Quest side; run it against
   `XR_INPUT=stub … mac_quest_bridge.py --cameras none`). **Real-headset test pending.** Protocol
   doc: `docs/refs/xrobotoolkit/stereo-vision.md`. KNOWN COLLISION: in stereo view the Quest
   app uses **B = flat↔3D toggle** — same button as our B=GO_HOME shortcut.
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
- **`XR_INPUT=stub` PRESSES BUTTONS** (it simulates a full session, including CONNECT): a
  headless test eval will dial whatever serve it's pointed at and drive it. Point test evals
  at a `yam_real_serve.py --fake` instance, NEVER at a real arm's serve.
- **Docker breaks the PC-service LAN announce** (UDP :29888 → headset's one-click IP popup):
  the container broadcasts onto its own bridge subnet with the container IP. That's the ONLY
  reason we ever typed IPs into the Network panel. `scripts/xrtk_announce.py` replays the
  packet natively (details: `docs/refs/xrobotoolkit/discovery-announce.md`).
- **MuJoCo offscreen render width is capped by `<global offwidth=…>`** in the scene XML
  (default 640!) — `mujoco.Renderer(width=…)` above it raises. `assets/yam/scene.xml` is now
  1920 for the stereo double-wide canvas.

## Commits this session (oldest first)

`96d24f1` topology decision + diagrams + bridge protocol · `cbe1b8d` YAM_control commit ·
`1b89b38` input seam + xr-bridge container + --serve-host · `2c74fb1` camera relay + transport
decision · `50f66ac` relay v1 · `c20a41b` fleet UI + VIEW toggle · `e681504` headset-ready
Turn ON. Diagrams: `docs/remote-topology.png`, `docs/what-happens-where.png`.
Protocol: `docs/XR-INPUT-BRIDGE.md`. Plan: `docs/PLAN.md` ("Remote topology", "Customer
transport" sections).
