# Eval VR — status & handoff

Single-operator VR eval for the YAM arm: **see the arm in the headset + drive it + switch
teleop/policy/home**, sim-first, graduating to the real arm (M2).

## What works now ✅
- **YAM sim teleop** — full 6-DOF, correct EE frame (`grasp` = i2rt grasp_site), on the Orin.
- **Live view in the headset** — the MuJoCo sim is rendered offscreen and streamed to the Quest's
  **Remote Vision** over H.264; the headset shows the *live arm you're driving* (fixed camera).
- **Coexistence confirmed** — Remote Vision video + controller tracking run at the same time
  (logged 722 frames streaming while the controller moved + gripped).
- **State machine + HUD** — `TELEOP / POLICY / GO_HOME / QUIT / CONNECT`, one driver at a time
  (Console + SafeRobot gate), with an on-screen **menu bar overlaid on the video**.
- **Demo policy** — `scripts/policies/gripper_forward.py` moves the gripper ~25 cm forward (placo
  IK), commanded through SafeRobot (velocity-clamped + killable).
- **Teleop velocity cap** — rate-independent rad/s clamp so teleop is smooth (no jumps); go-home
  eases at a rad/s cap too (PLAN Part 1 #6).

## One process: `scripts/eval_yam_vr.py`
Runs the control loop **headless** + streams the live sim. Reuses `eval_yam_states.py` for the
gate (SimRobot / SafeRobot / Console / policy loader) and the menu constants.
- **Controls:** right thumbstick L/R = move menu cursor, **click stick = select**; shortcuts
  A=TELEOP, X=POLICY, B=GO_HOME, Y=QUIT. (Nav moved L→R stick at user request.)
- **Loop:** read buttons → state → IK/policy/ease-home → step sim (real-time) → render @30fps →
  HUD overlay → stream.

## Video pipeline (verified against XRoboToolkit source, both ends)
- **Protocol:** sender connects to the Quest (Quest LISTENs via `MediaDecoder.startServer`); per
  frame send **`[4-byte big-endian length][H.264 Annex-B]`**. **No config handshake** for this
  direction (the `CameraRequestData` struct is only for the `--listen` flow).
- **Encoder:** PyAV libx264, `ultrafast / zerolatency / baseline / yuv420p`, 960×540 (proven).
  Resilient: reconnects on drop, fresh encoder per connection (leads with SPS/PPS + IDR).
- **Quest side:** Remote Vision panel → set **camera-source IP = the Orin (192.168.0.185)** →
  **LISTEN** (opens port **12345**). Quest IP seen as `192.168.0.30` (blocks ping; randomized MAC).
- **GL:** `MUJOCO_GL=glfw` + `DISPLAY=:0` (EGL is flaky on this Jetson).

## Key findings this session
- **EE frame was the root bug:** the standalone `yam.xml` `link6` is a placeholder i2rt overrides
  via the gripper mount; the real EE is i2rt's `grasp_site`. Fixed → `link_name="grasp"`. See
  `skills/adding-a-new-arm/SKILL.md`.
- **Ground truth = `yam.xml`**; `yam.urdf` is generated from it (`scripts/gen_yam_urdf.py`).
- **"half-works = negotiation mismatch"** — verifying the receiver source (not just the sender)
  showed no handshake was needed; first-keyframe-then-drop was a red herring (it was view/coexist).

## Run it (Orin)
```
# PC Service up; Quest connected. On the Quest: Remote Vision -> source IP 192.168.0.185 -> LISTEN
source ~/miniforge3/etc/profile.d/conda.sh && conda activate xr
APP=~/roboticsservice/opt/apps/roboticsservice
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$APP:$APP/lib:$APP/SDK/arm64" DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority MUJOCO_GL=glfw
cd ~/blupe-evals
python scripts/eval_yam_vr.py --quest-ip 192.168.0.30 --policy scripts/policies/gripper_forward.py:run
```

## Open items / next
- **Home pose** — current keyframe is a placeholder `[0,1,1,0,0,0]`; i2rt resets to all-zeros, but
  the *right* home is the real arm's rest pose. Capture it with
  `sim_teleop/scripts/yam_read_positions.py` (motors at zero torque, limp — **support the arm**),
  then bake those angles into the `home` keyframe in `assets/yam/yam.xml`.
- **Nav tuning** — confirm right-stick axis polarity on device; A/X/B/Y shortcuts are optional.
- **Forward direction** — gripper-forward policy uses world +X; flip `FORWARD` if needed.
- **M2 — real arm** — swap `SimRobot` for an i2rt-backed `Robot` (CAN, watchdog, torque-off);
  the same gate + caps graduate over. Reuse i2rt video pipeline with a real camera.
- Commit the eval scripts (currently uncommitted on the working tree).

## Files
- `scripts/eval_yam_vr.py` — integrated eval + live headset stream (the main entry).
- `scripts/eval_yam_states.py` — on-monitor variant + the gate/state classes (reused by vr).
- `scripts/orin/sim_video_sender.py` — standalone stream tester (orbiting arm).
- `scripts/policies/gripper_forward.py` — demo policy.
- `scripts/gen_yam_urdf.py`, `scripts/test_yam_ik.py` — model gen + verify.
- `assets/yam/{yam.xml,yam.urdf,scene.xml}` — consistent model (grasp EE, offscreen framebuffer).
