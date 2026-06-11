# YAM_control

Everything the real YAM arm needs to run, plus helper scripts. Runs on the **robot host**
(Orin, `andrew@192.168.0.185`) in the **i2rt venv** (`~/i2rt/.venv`, which has full i2rt + deps).
Deploy with: `rsync -az YAM_control/ andrew@192.168.0.185:~/blupe-evals/YAM_control/`

## Setup the YAM requires

1. **CAN bus up** — the gs_usb CANable is `can0`. After a reboot or a CANable replug it comes back
   DOWN; bring it up (needs your sudo password):
   ```bash
   bash YAM_control/setup_can.sh           # can0 @ 1 Mbps (Damiao motors)
   ```
2. **Environment** — use the i2rt venv (`~/i2rt/.venv/bin/python`); it has dm_env and the full i2rt
   stack. (`yam-venv` is CAN-only and is missing deps.)
3. **One owner of the bus at a time** — only one script may hold `can0`. Stop the serve / exerciser
   before starting another.

## Key facts (hard-won)

- **Gripper is normalized `[0,1]`** at the i2rt API: `0.0 = closed, 1.0 = open`. Never command raw
  motor radians (commanding the raw `0.164` is read as 16% open → the old "2 cm" bug).
- **Open/close = walk a bounded lead** toward the target (TRI raiden pattern): the command never
  leads the actual position by more than `GRIPPER_LEAD ≈ 0.076`, so a blocked jaw can't grind/crush.
- **Turning off ≠ killing the process.** The motors hold their last pose with torque on. A real limp
  must stop the control thread, then `motor_off` each motor (`turn_off.py` / `disable_motorchain`).

## Scripts

| Script | What it does | Run |
|---|---|---|
| `setup_can.sh` | Bring up `can0` (needs sudo) | `bash YAM_control/setup_can.sh` |
| `yam_real_serve.py` | The teleop/CONNECT consumer: arm follows the eval's streamed joints + gripper | `~/i2rt/.venv/bin/python YAM_control/yam_real_serve.py --channel can0` |
| `turn_off.py` | Cut all motor torque → arm limp | `~/i2rt/.venv/bin/python YAM_control/turn_off.py --channel can0` |
| `check_serve.py` | Health check: is the robot being served? (exit 0 = yes) | `~/i2rt/.venv/bin/python YAM_control/check_serve.py` |
| `gripper_cycle.py` | Standalone gripper exerciser (open↔close test) | `~/i2rt/.venv/bin/python YAM_control/gripper_cycle.py --channel can0` |
| `camera_sender.py` | Stream a webcam to the Quest (H.264) | `~/i2rt/.venv/bin/python YAM_control/camera_sender.py` |

## Typical session

```bash
bash YAM_control/setup_can.sh                                        # 1. CAN up (after reboot/replug)
~/i2rt/.venv/bin/python YAM_control/yam_real_serve.py --channel can0 # 2. serve (teleop drives the arm)
#    ... do teleop / eval from the headset (CONNECT) ...
# Ctrl-C the serve, then:
~/i2rt/.venv/bin/python YAM_control/turn_off.py --channel can0       # 3. turn the arm OFF (limp)
```
