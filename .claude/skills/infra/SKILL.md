---
name: infra
description: >
  Operate and debug the distributed teleop infrastructure (Mac+Quest operator node, GCP cloud
  relay, Orin+YAM robot node). Use when: something is down or stale (video, joints, input);
  restarting/deploying any tier; remote-managing the Orin over ssh; or when a new infra issue
  gets root-caused — append it to the Issue log here so it is never re-debugged from scratch.
---

# infra — operating & debugging the teleop stack

**Before debugging anything: check the `small-errors` skill** — known papercuts with
30-second fixes (dead joystick, doze fallout, stale IPs). This skill is for issues that
need actual investigation; confirmed quick fixes graduate to small-errors.

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
- **③ sender:** must have a small SO_SNDBUF + drop-on-backpressure (stereo_sender AND
  camera_relay have it since ISSUE-004; legacy mono path does NOT — sleep builds minutes
  of backlog).
- **Probe trap:** a FRESH connection can show 0.1 s freshness while every LONG-LIVED stream
  is seconds behind (new conns start at the relay's current frame; old conns carry their
  queue). Test sustained: hold one connection ≥15 s, drain continuously, clock the LAST frame.
- **④ headset:** fresh LISTEN/panel-open = fresh queue. Clock in headset vs wall clock is
  the final end-to-end measurement.

## Issue log (append; newest on top)

### ISSUE-004 · 2026-06-11 · Cameras seconds behind on EVERY long-lived stream (headset + viewer)
**Symptom:** camera video "incredibly behind" (multi-second, drifting) in both the headset
and the fleet-UI browser — yet a fresh probe connection measured 0.1 s freshness. That
contradiction IS the diagnosis: new connections start at the current frame; long-lived ones
carry their backlog (see "Probe trap" in the playbook above).
**Root cause (two, both in `camera_relay.py`'s per-client send loop):**
1. Blocking `wfile.write` with a default-size TCP send buffer — any throughput dip (Wi-Fi
   hiccup, cloud-relay congestion) queued frames that were all still delivered in order;
   once behind, a stream stayed behind forever. Same disease as ISSUE-001 cause 2, one hop
   earlier in the pipeline.
2. The loop re-sent the LATEST jpeg at 30 Hz even when no new frame existed — duplicate
   frames doubled bandwidth on the constrained path, making the queueing more likely (and
   it's the known received-fps ≠ freshness trap from ISSUE-001).
**Fix (deployed to the Orin):** only-new-frame sends (identity check on the jpeg buffer) +
`SO_SNDBUF` 128 KB + `select()` writability check that DROPS the frame when the client
stalls. Slow consumers now get FEWER frames, never OLDER frames (choppy beats laggy for
teleop). Verified: last frame of a 15 s sustained stream was 0.1 s old.
**Also:** when operator and robot share a LAN (home), point the eval DIRECTLY at the Orin
(`--cameras http://192.168.0.185:8089/{0,2} --serve-host 192.168.0.185 --serve-port 5599`)
— routing through GCP from ten feet away adds internet RTT and an unneeded choke point.
Relay endpoints (`127.0.0.1:18089/:15599`) are for remote operation.

### ISSUE-003 · 2026-06-11 · GREEN screen in stereo camera view after changing Wi-Fi
**Symptom:** at a cafe (network hop home→Starbucks→venue), the headset's ZEDMINI camera
view showed a solid GREEN screen on Listen, even with the correct new IP typed. Green =
the app's video texture allocated but its decoder never received a single frame.
**Root cause:** when a device changes networks, its established TCP sessions die WITHOUT a
FIN — the peer keeps them ESTABLISHED forever. `StereoVisionServer._serve` handled one
control client at a time with a blocking read, so it sat blocked on the morning's half-dead
session; the Quest's NEW connection completed its handshake in the kernel backlog (netstat:
ESTABLISHED with the 191-byte OPEN_CAMERA sitting in Recv-Q, never read) and was never
accepted. Each further Listen press queued another zombie.
**Diagnostic that cracked it:** `netstat -an | grep 13579` showing TWO established sessions
— the old network's pair still present + a new pair with bytes stuck in Recv-Q. If Recv-Q
is nonzero on a port we serve, the app behind it is not reading — find what it's blocked on.
**Fix:** newest-connection-wins preemption in `stereo_sender.py` — every accept closes the
previous control conn (unblocking its reader thread) and takes over; only the CURRENT
session may stop the video stream. Regression test: `fake_quest_stereo.py` run with a
parked zombie connection on :13579 (see scripts) — must still PASS.
**Lesson:** any of-ours TCP server that serves "the one operator" must treat a new
connection as the operator moving networks, not as an intruder: preempt, never queue.

### ISSUE-002 · 2026-06-11 · Typing IPs into the headset every session
**Symptom:** every headset launch needed the Mac IP pecked into the Network panel on the
in-VR keyboard (and once into the camera panel) — slow, error-prone, every session.
**Root cause:** the Quest app ALREADY auto-discovers the PC service — it listens on UDP
:29888 and pops a one-click IP-select dialog on a valid announce. Our PC service runs inside
Docker on macOS, so its announce (a) broadcasts onto the container's bridge subnet, never
the LAN, and (b) would advertise the container IP anyway. Doubly broken → silent fallback
to manual typing. Compounding: the Network-panel IP field is *never persisted by design*
(discovery is the intended UX); the camera-panel IP IS persisted (PlayerPrefs) after the
first entry + clean app exit.
**Fix:** `scripts/xrtk_announce.py` — replays the exact announce packet natively on the Mac
every 5 s (subnet broadcast + `--unicast <quest-ip>` belt-and-braces, since Android may
filter subnet broadcasts and the client binds 0.0.0.0). Start it with the Mac stack
(runbook in SESSION-HANDOFF.md). Wire format + client parse rules:
`docs/refs/xrobotoolkit/discovery-announce.md`.
**Lesson:** when an upstream tool demands tedious manual config, read its source for the
automation it already has — ours was broken by our own containerization, not missing.

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
