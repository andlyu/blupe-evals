# Orin setup (what's actually installed, and how to run)

This records the real, working setup for sim teleop on the Jetson Orin
(`andrew@192.168.0.185`, Ubuntu 22.04 aarch64). It deviates from the original
handoff in a few important ways — see "Deviations" below.

## TL;DR — run it

From a terminal **on the Orin's desktop** (so `DISPLAY=:0` works), with the Quest on:

```bash
bash ~/blupe-evals/scripts/orin/run_teleop.sh        # dual UR5e in MuJoCo
# preflight (no viewer, just check the stack imports):
CHECK=1 bash ~/blupe-evals/scripts/orin/run_teleop.sh
```

Then on the Quest: **Network → enter the Orin IP (`192.168.0.185`) → Controller + Send ON**.
Hold a grip to clutch; move the controller to drive the arm.

## YAM teleop — the model-consistency fix (important)

YAM's `assets/yam/yam.xml` (MuJoCo sim) and the legacy `yam.urdf` are **inconsistent** — they
were exported separately and describe `link_6` differently (~6 cm + a non-constant ~180°).
The framework does IK with placo (a URDF model) and the sim with MuJoCo, so the two must
match. They don't, which made the arm swing/jump on every grip and orientation attempt.
(i2rt never hit this: i2rt's IK is *mink*, MuJoCo-native, and never loads the URDF — so the
URDF was vestigial/unvalidated, and we inherited it.)

**Fix: make placo load the SAME MuJoCo model the sim uses.** Two parts:

1. `scripts/teleop_yam_mujoco.py` points placo at the **MJCF**:
   `robot_urdf_path="assets/yam/yam.xml"`, `control_mode="pose"` (stock full-pose). The
   inconsistent `yam.urdf` is **no longer used**, and all the orientation workarounds are gone.
2. Run **`python scripts/orin/patch_framework.py`** once (re-run after any framework
   reinstall). It makes 3 small, guarded edits so placo can load an MJCF (fixed-base) — all
   no-ops for URDF arms (UR5e/Flexiv), so they're unaffected:
   - load via `placo.Flags.mjcf` for `.xml` paths;
   - skip the floating-base `q[:7]` init when the model has no free-flyer (in `_placo_setup`
     and in `mujoco_utils.calc_pin_q_from_mujoco_qpos`).

Verified: with the MJCF, placo's `link_6` matches the sim to **0.00° / 0.0 m**.

Run YAM teleop (from a terminal on the Orin desktop, Quest connected):
```bash
python ~/blupe-evals/scripts/orin/patch_framework.py            # once
DISPLAY=:0 python ~/blupe-evals/scripts/teleop_yam_mujoco.py --no-visualize-placo
```
Controls: hold right grip = clutch (move + rotate → EE follows in full 6-DOF); release =
freeze; **B** = home; **thumbstick click** = save current pose as home.

## What's installed on the Orin

| Component | Where / how |
| --- | --- |
| **PC Service** (the bridge the Quest talks to) | Prebuilt **headless arm64 `.deb`** from the [v1.0.0 release](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases), extracted (no `sudo`) to `~/roboticsservice`. Run via `RoboticsServiceProcess` (headless, `QT_QPA_PLATFORM=offscreen`). Listens on `*:63901` (Quest) + `127.0.0.1:60061` (local SDK). |
| **`xrobotoolkit_sdk`** (Python XR state) | Built in conda env `xr` (Py3.10) by compiling only the pybind wrapper against the **prebuilt `libPXREARobotSDK.so`** from the deb (skips the from-source SDK build). |
| **`xrobotoolkit_teleop`** + `placo`, `mujoco`, `tyro`, `meshcat` | `pip install -e . --no-deps` + minimal deps in env `xr`. (Skipped the hardware/camera deps: `ur_rtde`, `pyrealsense2`, `dex_retargeting`, `torch`, etc.) |
| Repos | `~/XRoboToolkit/{XRoboToolkit-PC-Service (orin branch), -PC-Service-Pybind, -Teleop-Sample-Python}` |
| This repo | `~/blupe-evals` (rsynced from the Mac) |

## Deviations from the original handoff

1. **No Mac in the loop.** The PC Service + `xrobotoolkit_sdk` are Windows/Linux only
   (precompiled `.so`/`.dll`, no macOS `.dylib`). The whole XR/sim stack runs on the Orin.
2. **No Qt build.** The handoff feared a from-source Qt6 build of the service. Unnecessary —
   the **headless arm64 `.deb`** ships `RoboticsServiceProcess` + bundled Qt6/gRPC.
   (One missing system lib, `libdouble-conversion.so.3`, was dropped into the app's `lib/`
   without sudo via `apt-get download`.)
3. **Example arm first.** We run the ready-made **UR5e** example (it already has actuators,
   a `home` keyframe, and a `*_target` mocap body) to validate the pipeline. Porting YAM is
   the next step — our vendored `assets/yam` is kinematics-only (no actuators/keyframe/mocap).
4. **EE link name.** For YAM later: placo/URDF EE link is **`link_6`** (underscore), MuJoCo
   body is `link6`. (Handoff guessed `link6` for both.)

## Gotchas baked into `run_teleop.sh`

- `LD_LIBRARY_PATH` must put **conda's `libstdc++` first** (so `meshcat`/`icu` find
  `CXXABI_1.3.15`) *and* include the deb's dirs (so the SDK's `libPXREARobotSDK.so` resolves).
- The MuJoCo viewer needs a real GL display → `DISPLAY=:0` on the Orin's monitor. Run from a
  local desktop terminal, not plain SSH.
- Orin Wi-Fi drops intermittently; if SSH/the Quest link stalls, nudge it and retry.

## Next: port YAM

Add actuators + a `home` keyframe + a `right_target` mocap body to `assets/yam` (mirror
`assets/universal_robots_ur5e/{ur5e.xml,scene.xml}` from the teleop repo), then run with our
`scripts/teleop_yam_mujoco.py` (`link_name="link_6"`).
