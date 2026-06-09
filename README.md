# blupe-evals

Teleop a single **YAM** arm with [XRoboToolkit](https://xr-robotics.github.io/):
Quest controller → XRoboToolkit → YAM in **MuJoCo sim** first (no hardware), then
the **real YAM** on a Jetson Orin via `i2rt`.

This is a fresh, minimal repo. We deliberately do **not** reuse the custom Quest VR
path from `blupe-eval-console` — the only thing carried over is the vendored YAM
MuJoCo model in [`assets/yam/`](assets/yam/). Full background and hardware notes:
[`docs/XROBOTOOLKIT-YAM-HANDOFF.md`](docs/XROBOTOOLKIT-YAM-HANDOFF.md).

## Layout

```
assets/yam/            vendored YAM model (scene.xml, yam.xml, yam.urdf, assets/*.stl)
scripts/
  teleop_yam_mujoco.py    sim teleop (start here)
  teleop_yam_hardware.py  real-arm teleop via i2rt (stub — Orin only)
docs/                  handoff + notes
```

## Install (Python 3.10)

Three external pieces, none cleanly on PyPI:

1. **XRoboToolkit PC Service** — the host bridge service (gRPC) the Quest connects to.
   - Windows: one-click installer.
   - Linux / Jetson Orin: build from source (`RoboticsService/`, less documented).
2. **`xrobotoolkit_sdk`** — Python bindings to read XR state, from
   [XRoboToolkit-PC-Service-Pybind](https://github.com/XR-Robotics) (`setup_orin.sh` on Orin).
3. **`xrobotoolkit_teleop`** — the framework, from
   [XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics) (`setup_conda.sh`, Python 3.10).

Then this repo's small deps:

```bash
pip install -e .   # tyro, numpy
```

## Connect the Quest

```
Quest app ──Wi-Fi──► PC Service (gRPC) ──local──► xrobotoolkit_sdk ──► this code
```

- One-time: Quest **Developer Mode** → sideload the XRoboToolkit Quest APK (adb/SideQuest).
- Per session: Quest + host on the **same Wi-Fi** → run the **PC Service** on the host →
  in the Quest app **Network → enter the host IP**, toggle **Controller** + **Send** ON →
  Status: connected.

## Run (sim)

```bash
python scripts/teleop_yam_mujoco.py
# options: --xml-path --robot-urdf-path --scale-factor --visualize-placo / --no-visualize-placo
```

Hold the **right grip** (clutch, >0.5) to engage; move the controller to drive the arm.

## Model name gotchas (verified against the vendored assets)

The same link has two different names depending on the file — easy to mix up:

| Thing            | Name in file        | Used by                         |
| ---------------- | ------------------- | ------------------------------- |
| EE link (URDF)   | `link_6`            | placo IK (`link_name` in config)|
| EE body (MuJoCo) | `link6`             | MuJoCo sim / visualization      |
| EE site (i2rt)   | `grasp_site`        | real hardware                   |
| Joints (URDF)    | `joint1`…`joint6`   | —                               |

The handoff originally guessed `link6` for the placo target; the correct URDF link is
**`link_6`**, which is what `scripts/teleop_yam_mujoco.py` uses.

## Next: real YAM

Everything runs on the **Jetson Orin** (`andrew@192.168.0.185`, arm64) — the only box
reaching the arm on CAN. Implement `scripts/teleop_yam_hardware.py` (placo IK targets →
i2rt `command_joint_pos`), modeled on XRoboToolkit's `teleop_dual_ur5e_hardware.py`.
Mind the **~400 ms motor watchdog** and the safe **torque-off** sequence — see the
hardware script's docstring and the handoff.
