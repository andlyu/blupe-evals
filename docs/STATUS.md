# blupe-evals — status & plan

_Last updated: 2026-06-08_

## Goal

Teleop a single **YAM** arm with [XRoboToolkit](https://xr-robotics.github.io/):
Quest controller → XRoboToolkit → YAM, **sim first, then the real arm**. This repo is
the teleop / data-collection front end that later grows into the evals harness.

## Architecture (as actually built)

```
Quest (XRoboToolkit-Quest app)
   │  Wi-Fi
   ▼
PC Service  (RoboticsServiceProcess, headless)         ── on the Orin
   │  localhost gRPC (127.0.0.1:60061)
   ▼
xrobotoolkit_sdk  (Python, reads XR state)             ── conda env `xr`
   │
   ▼
xrobotoolkit_teleop  (placo IK)  →  MuJoCo sim  →  viewer on the Orin's monitor
```

**Everything runs on the Jetson Orin** (`andrew@192.168.0.185`, Ubuntu 22.04 aarch64).
The Mac cannot run the stack (no macOS build of the PC Service / SDK — precompiled
Linux/Windows only).

## What works (verified end-to-end) ✅

- **Full stack on the Orin**: PC Service + `xrobotoolkit_sdk` + `xrobotoolkit_teleop` +
  `placo` + `mujoco` (conda env `xr`, Python 3.10).
- **Quest app**: was *not* installed (only KinovaBot / another teleop app were). Sideloaded
  `XRoboToolkit-Quest-1.0.1.apk` via adb. Connects to the Orin and streams.
- **Live input confirmed**: controller + head pose track movement (23/24 non-zero samples),
  FPS ticker active.
- **UR5e example teleops in sim** on the Orin's monitor (fullscreen): squeeze grip = clutch,
  move controller = arm EE follows via IK. **← M1 done.**

## Current blocker — YAM in sim 🚧

Built a teleop-ready YAM scene (`assets/yam/`): added **actuators**, a **`home` keyframe**,
and a **`right_target` mocap**. The **MuJoCo side is validated** (constructs: nq=6, nu=6,
mocap found, home pose applied).

**placo fails to load the URDF:**
```
ValueError: Mesh package://assets/link_6_collision.stl could not be found.
```
`yam.urdf` references meshes as `package://assets/...`, which pinocchio/placo can't resolve.
Fix candidates (pick one):
1. `export ROS_PACKAGE_PATH=~/blupe-evals/assets/yam` so `package://assets/X` →
   `assets/yam/assets/X` (non-invasive; need to confirm placo honors it).
2. Rewrite the mesh paths in `yam.urdf` to resolvable paths.
3. Load placo ignoring collision meshes (IK only needs kinematics) — but the framework calls
   `placo.RobotWrapper(urdf)` with no flags, so this needs a wrapper.

## Key facts & decisions

- **Box**: the Orin is the only machine that can run XRoboToolkit (and later reach the arm's CAN).
- **PC Service**: prebuilt **headless arm64 `.deb`** extracted to `~/roboticsservice` (no sudo;
  `sudo` needs a password on the Orin, so we avoid it). One missing lib (`libdouble-conversion`)
  dropped in via `apt-get download`.
- **Env gotchas** (baked into `scripts/orin/run_teleop.sh`): conda `libstdc++` first (meshcat),
  deb dirs for the SDK, `DISPLAY=:0` + `XAUTHORITY` for the viewer.
- **EE link name**: placo/URDF = **`link_6`** (underscore); MuJoCo body = `link6`.
- **Display**: the Orin's only screen is a small **1024×600** panel (DP-0), X11/GNOME. Viewing
  experience is limited — open question below.

## Open questions (for planning)

1. **YAM mesh fix** — which of the three approaches above.
2. **Viewing** — stick with the small Orin panel, attach a bigger monitor, render on the Mac, or
   (later) stream into the Quest headset? (Asked once; currently on the small panel.)
3. **Sim scope** — add manipulable objects (a cube), task setups, data logging?
4. **Real arm (M2)** — when to move to the physical YAM: placo IK → i2rt `command_joint_pos` on
   `can0`, with the ~400 ms watchdog + safe torque-off.
5. **Evals** — when does the actual eval harness (the point of "blupe-evals") get built vs. more
   teleop polish.

## Proposed next steps

1. Unblock YAM: fix the URDF mesh resolution → YAM teleops in sim (finish current task).
2. Decide the viewing setup (affects how pleasant everything after this is).
3. Add task objects (cube) for manipulation in sim.
4. M2: real YAM on the Orin via i2rt, with safety.

## Where things live

| What | Path |
| --- | --- |
| This repo (Mac) | `~/Projects/Blupe/blupe-evals` |
| This repo (Orin) | `~/blupe-evals` (rsynced) |
| XRoboToolkit repos (Orin) | `~/XRoboToolkit/` |
| PC Service (Orin) | `~/roboticsservice/opt/apps/roboticsservice` |
| conda env | `xr` (miniforge3) |
| Launcher | `scripts/orin/run_teleop.sh` |
| Setup notes | `docs/ORIN-SETUP.md` |
