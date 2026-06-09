# Handoff: teleop a single YAM with XRoboToolkit (fresh start)

## The decision (the pivot)
We built `blupe-eval-console` (a cross-arm eval console + our own Quest VR teleop). **We are SCRAPPING the custom VR path and starting a fresh, minimal repo that uses [XRoboToolkit](https://xr-robotics.github.io/) directly to teleop ONE YAM arm.** Ignore `blupe-eval-console` for this; the only thing to reuse from it is the **vendored YAM MuJoCo model** (see Assets).

Goal of the new repo: Quest controller → XRoboToolkit → YAM in **MuJoCo sim** first (no hardware), then the **real YAM** via i2rt.

## What XRoboToolkit is
Published, MIT/Apache framework (Best Paper SII 2026, ByteDance) at `github.com/XR-Robotics`. OpenXR XR-teleop, **placo** optimization IK, controller/hand/head/body tracking, **MuJoCo sim + many real robots**, in-headset stereo video. Built for VLA data collection. **It supports Jetson Orin/arm64.**

## The repos we use (+ licenses)
- **XRoboToolkit-Teleop-Sample-Python** (MIT) — the `xrobotoolkit_teleop` Python framework. Config-driven per-robot controllers (e.g. `MujocoTeleopController`), placo IK, MuJoCo sim + hardware, data logging. Tested on **Ubuntu 22.04 / Orin**. This is our main dependency.
- **XRoboToolkit-PC-Service** (Apache-2.0) — the host **bridge service** (`RoboticsService`, gRPC). **Windows = one-click installer; Linux/Orin = build from source** (less documented).
- **XRoboToolkit-PC-Service-Pybind** (MIT) — `xrobotoolkit_sdk` Python bindings to read XR state. `setup_orin.sh` supports Jetson Orin.
- **XRoboToolkit-Unity-Client-Quest** (MIT) — the Quest headset app (Unity, full source).
- **XRoboToolkit-Orin-Video-Sender** (MIT) — robot camera → headset stereo feedback; runs on Orin (future nice-to-have).
- **AVOID** `XRoboToolkit-RobotVision-PC` (GPLv3).

## Pose/input API (`xrobotoolkit_sdk`)
```python
import xrobotoolkit_sdk as xrt
xrt.init()
xrt.get_right_controller_pose()   # [x,y,z, qx,qy,qz,qw]
xrt.get_right_grip()              # analog 0..1  (use as clutch, >0.5)
xrt.get_right_trigger()           # analog 0..1  (use as gripper)
xrt.get_right_axis_click()        # bool (joystick press)
xrt.get_A_button()                # bool
xrt.close()
```

## Adding YAM = config (sim is ~25 lines)
Their `MujocoTeleopController` is config-driven (xml + urdf + a hand→link map). The whole **sim** experiment:
```python
# teleop_yam_mujoco.py
import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController

def main(xml_path="assets/yam/scene.xml", robot_urdf_path="assets/yam/yam.urdf",
         scale_factor=1.0, visualize_placo=True):
    config = {"right_hand": {                      # single arm
        "link_name": "link6",                      # YAM EE link — VERIFY exact name in yam.urdf
        "pose_source": "right_controller",
        "control_trigger": "right_grip",           # grip = clutch
        "vis_target": "right_target",
    }}
    MujocoTeleopController(xml_path=xml_path, robot_urdf_path=robot_urdf_path,
                          manipulator_config=config, scale_factor=scale_factor,
                          visualize_placo=visualize_placo).run()

if __name__ == "__main__":
    tyro.cli(main)
```
Reference their `scripts/simulation/teleop_dual_ur5e_mujoco.py` (the dual-arm version) and `scripts/hardware/teleop_dual_ur5e_hardware.py` (for the real pattern).

**Real YAM** = a small bridge: take placo IK joint targets → `i2rt` `command_joint_pos`; read state via `get_joint_pos`. Model it on their hardware controller.

## How the Quest connects (network, not USB during use)
```
Quest app ──Wi-Fi──► RoboticsService (PC Service, gRPC) ──local──► xrobotoolkit_sdk ──► your Python
```
- One-time: Quest **Developer Mode** → sideload the XRoboToolkit Quest **APK** (adb/SideQuest).
- Per session: Quest + host on same Wi-Fi → run the **PC Service** on the host → in the Quest app **Network → enter the host IP**, toggle **Controller** + **Send** ON → Status connected.
- **Quest is already connected** (as of this handoff). Open question: **which box runs the PC Service it joined** — a laptop (→ do sim there) or the Jetson Orin (→ needed for real YAM).

## YAM / hardware facts
- Real YAM on a **Jetson Orin**: `andrew@192.168.0.185`, Ubuntu 22.04 **arm64**, passwordless SSH. **Wi-Fi drops intermittently** (SSH times out; recurring — may need a power/Wi-Fi nudge).
- `i2rt` at `~/i2rt`, installed in the **uv venv `~/i2rt/.venv`** (has i2rt + mujoco 3.9.0 + mink). Arm on CAN **`can0`** (name swaps `can0`/`can1` across reboots — ping motors to find it).
- YAM = 6 arm joints (DM4340×3, DM4310×3) + gripper (DM4310), **motor ids 1–7**, Damiao MIT-mode, **~400 ms watchdog** (no command → limp).
- **Torque OFF (limp):** stop the control thread FIRST (`robot._stop_event.set(); robot._server_thread.join()`), then `robot.motor_chain.motor_interface.motor_off(id)` for id 1..7, then close. `get_yam_robot(channel, gripper_limits_override=np.array([0.0,1.0]))` skips gripper calibration. (A naive disable throws a harmless `fd=-1` traceback but does disable.)
- **YAM model assets to reuse:** `blupe-eval-console/src/blupe_eval_console/assets/yam/` → `scene.xml`, `yam.urdf`, `assets/*.stl`. EE: sim body `link6`, real i2rt site `grasp_site`. Copy these into the new repo's `assets/yam/`.

## Where it runs
- **Sim YAM** (first try): PC Service + `xrobotoolkit_teleop` + MuJoCo-YAM on **one box** (where the Quest connected). Easiest on a Windows/Linux PC.
- **Real YAM**: everything on the **Jetson Orin** (only box reaching `can0`) → PC Service built from source on Orin.

## First steps in the new conversation
1. Confirm **which box runs the PC Service** the Quest joined.
2. `git init` a new repo (e.g. `yam-xr-teleop`); copy `assets/yam/` from `blupe-eval-console`.
3. Install: PC Service (Windows installer or Orin source build) + `xrobotoolkit_sdk` (`setup_orin.sh` on Orin) + `xrobotoolkit_teleop` (their `setup_conda.sh`, Python 3.10).
4. Write `teleop_yam_mujoco.py` (above); **verify the YAM EE link name** in `yam.urdf`; run it → Quest drives YAM in MuJoCo.
5. Then `teleop_yam_hardware.py` (placo IK → i2rt) on the Orin for the real arm.
6. Confirm the exact **Orin PC-Service launch command** (their docs are Windows-first — dig into `RoboticsService/`).

## Browsing note
This machine's global rule: use the gstack **`/browse`** skill for web (binary at `~/.claude/skills/gstack/browse/dist/browse`). The XRoboToolkit GitHub blob views are JS-heavy — read source via `raw.githubusercontent.com/...` URLs.
