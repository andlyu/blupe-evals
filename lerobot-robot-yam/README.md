# lerobot-robot-yam

Drive an **i2rt YAM** arm as a **LeRobot follower** (`--robot.type=yam_follower`), so YAM
works with `lerobot-record`, `-replay`, `-teleoperate`, and policy training/eval.

Modeled on the Seeed reBot plugin (`lerobot-robot-seeed-b601`) — same plugin shape, same
`Robot` interface — with one deliberate difference: **the motor layer is a socket, not a
motor SDK.**

## Why a socket instead of importing i2rt

YAM's driver (i2rt) pins `numpy==2.2.6` and pulls `python-can`, `mujoco`, `mink`, `ruckig`,
… A typical LeRobot environment is built on `numpy 1.x` (and on a Jetson, the NVIDIA-built
torch is ABI-locked to it). Forcing i2rt into that env risks breaking the whole rig.

So `YamFollower` never imports i2rt. It talks over TCP to **`yam_serve.py`**, which runs
in i2rt's own venv and already owns the arm and all the safety (velocity clamp,
hold-on-disconnect, torque-off). LeRobot and i2rt stay in separate interpreters — even
separate Python versions — and exchange only joint vectors.

```
LeRobot env (numpy 1.x, torch)              i2rt venv (numpy 2.x, python-can)
  YamFollower(Robot) ──TCP :5599──►  yam_serve.py  ──CAN──►  YAM
  --robot.type=yam_follower          (clamp · hold · torque-off)
```

## Install

```bash
pip install -e .            # in your lerobot environment; only dep is lerobot>=0.4
```

## Use

1. On the robot computer, start the serve (i2rt venv):
   ```bash
   cd ~/lerobot-robot-yam
   PYTHONPATH=$HOME/i2rt:$HOME/lerobot-robot-yam \
     $HOME/i2rt/.venv/bin/python -m lerobot_robot_yam.yam_serve --channel can0
   ```
2. Point a lerobot command at the follower:
   ```bash
   lerobot-record \
     --robot.type=yam_follower \
     --robot.serve_host=127.0.0.1 --robot.serve_port=5599 \
     --robot.cameras='{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
     ...
   ```
   Cameras can point at local devices or at `camera_relay.py`'s MJPEG URLs.

## Hardware Check

When the YAM is physically set up and `can0` is free, start the serve above in one terminal.
Then run this from the LeRobot environment:

```bash
cd ~/lerobot-robot-yam
PYTHONPATH=$HOME/lerobot-robot-yam \
  $HOME/miniforge3/envs/blapa-jetson-env/bin/python -m lerobot_robot_yam.examples.read_and_cycle \
  --read-seconds 5 \
  --low 0.5 --high 0.9 --cycles 5
```

This connects through `YamFollower`, prints `joint_1.pos` through `joint_6.pos` plus
`gripper.pos` for five seconds, then holds the current arm pose while cycling the gripper
between `0.5` and `0.9` five times. By default it disconnects with `{"shutdown": true}`,
so the serve turns motor torque off at the end. Add `--keep-serve` if you want the serve
to keep holding the last pose.

## Units (read this before trusting a dataset)

Arm joints are **radians**, gripper is **normalized 0..1** (0=closed, 1=open) — passed to
and from the serve verbatim, no conversion. reBot exposes degrees; we keep YAM's native
i2rt units to avoid a silent conversion bug. Feature names: `joint_1.pos … joint_6.pos`,
`gripper.pos`. Confirm the joint order matches your arm before recording.

## Status

Built; registration + protocol tested headless against the serve. A full moving-arm test
needs healthy hardware.
