# Eval VR — status & handoff

Snapshot for kicking off **Step 2 (state buttons)**, with Step 1 (VR viewer) state preserved.

## Where we are
- **YAM sim teleop works** (full 6-DOF, correct EE frame) on the Orin via XRoboToolkit.
- Repo pushed: `github.com/andlyu/blupe-evals` (private), branch `main`.
- Goal now (from `docs/PLAN.md`): an **eval** where the operator is remote — **see the robot
  in the headset** (camera-like) + **buttons to switch teleop / policy / go-home**.

## Architecture decision (settled)
- **XRoboToolkit can't render to the headset** — its SDK is input only (controller/headset
  poses + A/B/X/Y buttons) + `send_bytes_to_device`. No display surface.
- **But** the Quest **Unity-Client has "Remote Vision"** (Listen → display an H.264 stream),
  fed by the **`OrinVideoSender`** pipeline. So **one Quest app = controller teleop + video
  view**. This is also the real-arm camera pipeline (real cam → headset), so it's right for M2.
- We do NOT need Vuer/WebXR.

## Step 1 — VR viewer (IN PROGRESS)
- Sender: `scripts/orin/sim_video_sender.py` — `mujoco.Renderer` (offscreen EGL) → **PyAV
  libx264** → TCP. Validated on Orin: render `(540,960,3)` OK, H.264 encode OK.
- Quest Remote Vision panel shows: **State**, source dropdown **ZEDMINI / PICO4u**, **LISTEN**.
  Listen = Quest opens a socket; **sender connects to the Quest** (`--quest-ip`, port `12345`).
- **Wire format** (from `XRoboToolkit-Orin-Video-Sender`, `main_zed_tcp.cpp`):
  - The Quest first sends a **config handshake** `CameraRequestData`:
    magic `0xCA 0xFE`, version `1`, then 7×int32 = `width,height,fps,bitrate,enableMvHevc,
    renderMode,port`, then 2 compact strings `camera,ip`.
  - Then the sender streams, per frame: **`[4-byte big-endian length][H.264 Annex-B]`**.
  - **ZED = 2560×720 side-by-side stereo**; PICO4u = headset format. `enableMvHevc`/`renderMode`
    select mono vs stereo-3D layout.
- **TODO to finish viewer:** (1) read the config handshake on connect; (2) render at the
  requested resolution, side-by-side stereo when stereo is requested; (3) test with the Quest
  in Listen mode (need the Quest's IP). Current sender streams raw framed H.264 but does NOT
  yet read the handshake — that's the next edit.

## Step 2 — STATE BUTTONS (kick off here)
Goal: a **Console gate** — one driver at a time — switched by Quest controller buttons.

- **SDK button getters** (already available, env below):
  `get_A_button, get_B_button, get_X_button, get_Y_button,
   get_left_menu_button, get_right_menu_button, get_left_axis_click, get_right_axis_click,
   get_left_trigger, get_right_trigger, get_left_grip, get_right_grip`.
- **Proposed mapping** (tune later):
  - **A → teleop** (current grip-clutch IK teleop, `scripts/teleop_yam_mujoco.py`).
  - **B → go-home** (ease to the `home` keyframe under the speed clamp).
  - **X → policy** (run a stub `run(robot, stop)` — see PLAN Part 2/3).
  - **Y / menu → STOP/kill** (disarm + hold, no-jump).
- **Design (from `docs/PLAN.md`):** one driver at a time; teleop is ours (smooth + speed-clamp);
  policy is the user's (we monitor + halt, don't rewrite); `stop()` disarms instantly and holds.
- **Port the core from `blupe-eval-console`** (reference, not a dep):
  - `interface.py` — `Robot` / `Observation` / `run(robot, stop)` seam.
  - `console.py` — `Console` gate + `SafeRobot` (velocity cap, workspace box, kill).
  - `arms.py` — `ArmProfile` / `ARMS` (YAM profile: ee frame, home, limits, speed cap).
  - Note: console teleop is **position-only, fixed orientation** (`real.py` holds `R_home`); our
    XRoboToolkit teleop is full 6-DOF — keep ours for the teleop state.
- **First milestone:** a loop that reads buttons → prints/holds state transitions (teleop ↔
  go-home ↔ stop) driving the **sim** YAM, before policy. Sim-first, then real (M2).

## Env / how to run
- Orin: `andrew@192.168.0.185`, conda env **`xr`** (py3.10). PC Service must be up.
- `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$APP:$APP/lib:$APP/SDK/arm64"` where
  `APP=~/roboticsservice/opt/apps/roboticsservice` (needed for `xrobotoolkit_sdk`).
- Teleop: `cd ~/blupe-evals && DISPLAY=:0 python scripts/teleop_yam_mujoco.py`.
- Sync from Mac: `rsync -az --exclude .venv --exclude .git ~/Projects/Blupe/blupe-evals/ andrew@192.168.0.185:~/blupe-evals/`.

## Key facts / gotchas
- **EE frame = `grasp`** (i2rt `grasp_site`, linear_4310 gripper; +Z = approach). The standalone
  `yam.xml` `link6` is a placeholder i2rt overrides — see `skills/adding-a-new-arm/SKILL.md`.
- **Ground truth = `yam.xml`** (i2rt loads only the `.xml`); `yam.urdf` is generated from it by
  `scripts/gen_yam_urdf.py`.
- Verify model: `scripts/test_yam_ik.py` (consistency / hold / track).
- Diagnostics (can be cleaned up): `scripts/test_yam_{joints,orient,wrist}.py`,
  `scripts/orin/teleop_yam_logged.py`.
