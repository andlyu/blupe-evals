---
name: infra
description: >
  Operate and debug the distributed teleop infrastructure (Mac+Quest operator node, GCP cloud
  relay, Orin+YAM robot node). Use when: something is down or stale (video, joints, input);
  restarting/deploying any tier; remote-managing the Orin over ssh; or when a new infra issue
  gets root-caused — append it to the Issue log here so it is never re-debugged from scratch.
---

# infra — operating & debugging the teleop stack

Endpoints, tokens, start commands, and the live architecture diagram live in
`docs/SESSION-HANDOFF.md` (keep that current). This skill is the *operating discipline* +
the issue log.

## Operating rules (each one bought with hours)

1. **Kill and start in SEPARATE ssh sessions.** A compound `ssh 'pkill -f X; ... X ...'`
   kills its own remote shell (the cmdline matches the pattern) → exit 255, nothing ran.
   Also use `pgrep -f "name[.]py"` so the probe can't match itself.
2. **Remote daemons:** `ssh host 'setsid nohup CMD > /tmp/x.log 2>&1 < /dev/null &'` —
   without `setsid` + stdin redirect the child dies with the ssh session.
3. **Singletons by port-bind.** One owner per resource: CAN bus, each `/dev/video*`,
   `camera_relay :8089`, serve `:5599`. Two camera relays racing over devices garbles
   capture rates silently (see ISSUE-001). The port bind is the lock — if a start "succeeds"
   but the port was taken, suspect a zombie.
4. **The Quest kills ALL sockets when it sleeps** (headset off → video LISTEN + service
   conn die; on wake nothing auto-resumes except our retry loops). Re-enter IP / re-LISTEN,
   or disable Auto-Sleep / tape the proximity sensor for work sessions.
5. **i2rt swallows SIGINT.** To stop a serve remotely: `kill -9` it, then run
   `YAM_control/turn_off.py` (fresh bus connection) for guaranteed torque-off. `turn_off`
   timing out = arm unpowered (fine, report it).
6. **Measure freshness, not fps.** `camera_relay` re-sends its latest JPEG at 30 Hz whether
   or not it's new — "30 fps received" proves nothing about staleness. The relay burns
   `HH:MM:SS.d #frame` into every frame; **read the clock in the image** vs a wall clock.
   Any staleness question is answered by a screenshot, at any point in the pipeline.
7. **Python envs on the Orin:** serve/turn_off → `~/i2rt/.venv/bin/python` (has dm_env/i2rt,
   NO cv2); camera/agent → `~/miniforge3/envs/xr/bin/python` (has cv2).

## Playbook: "video is stale / black / old"

The pipeline has exactly four buffer points; bisect with the burned-in clock:

```
camera → camera_relay → cloud relay ──► fleet-UI browser <img>   ← checkpoint A
                              └──► ① cv2 reader (eval grabber) → ② grabber tiles
                                   → encoder → ③ sender→Quest TCP → ④ Quest decode queue
```

- **A: open the fleet UI cameras.** Live clock in browser = everything through the cloud is
  good; the problem is Mac-side or headset-side. Stale in browser = robot-side (check for
  duplicate camera_relay processes, dead capture).
- **① reader:** must be one drain-thread per camera (fixed in ISSUE-001; never read two
  network streams sequentially in one loop — TCP queues, staleness grows without bound).
- **② tiles:** dead streams must show "NO SIGNAL", never a frozen last frame.
- **③ sender:** must have a small SO_SNDBUF + drop-on-backpressure (stereo_sender has it;
  legacy mono path does NOT — sleep builds minutes of backlog).
- **④ headset:** fresh LISTEN/panel-open = fresh queue. Clock in headset vs wall clock is
  the final end-to-end measurement.

## Issue log (append; newest on top)

### ISSUE-001 · 2026-06-11 · Headset showed old/black video while browser was live
**Symptom:** headset video "not streaming", later recognized as *old frames*; fleet-UI
browser cameras perfectly live; sender socket connected and streaming the whole time.
**Three stacked causes** (each alone sufficient, which is why every single fix "didn't work"):
1. `CameraGrabber` read two network MJPEG streams **sequentially in one thread** → consumed
   slower than produced → TCP backpressure queued frames server-side → composite drifted
   **32 s stale after 25 s of runtime**, unbounded. *Fix:* one drain-to-latest thread per
   camera + paced compositor.
2. Mac→Quest sender had **unbounded TCP buffering**; headset sleep froze the app but not its
   TCP stack → minutes of video queued → decoder replayed the backlog in order on wake
   ("black", then old frames). *Fix:* `SO_SNDBUF ~128 KB` + drop frames on backpressure
   (built into `stereo_sender.py`; mono path remains legacy-broken).
3. **Two `camera_relay` processes** briefly raced over `/dev/video*` (manual restart vs the
   fleet agent's `arm_on` auto-start) → erratic capture rates/counters. *Fix:* let the port
   bind enforce the singleton; check `pgrep -af camera_relay` before manual starts.
**Instrument that cracked it:** burning capture wall-time + frame counter into every frame
at the source — staleness became readable in any view, including by the operator in the
headset. Keep it.
**Measurement trap found:** received-fps ≠ freshness (relay re-sends latest at 30 Hz).
