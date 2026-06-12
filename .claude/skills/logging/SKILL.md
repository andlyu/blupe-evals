---
name: logging
description: >
  Logging conventions + latency instrumentation for the teleop stack. Use when: adding log
  lines or latency probes to any tier; diagnosing teleop lag from [lat] lines; hunting for
  the right log file on the Mac/Orin/relay; or when a new logging convention is adopted —
  record it here so every tier stays greppable the same way.
---

# logging — conventions, log map, latency probes

## Where every log lives

| Process | Host | File | How it gets there |
|---|---|---|---|
| `eval_yam_vr.py` (+ `stereo_sender` — same process) | Mac | `/tmp/eval_live.log` | shell redirect in the start command (SESSION-HANDOFF "How to start") |
| relay operator | Mac | none by default | add `> /tmp/relay_operator.log 2>&1` if you need it |
| relay agent | Orin | `/tmp/relay_agent.log` | the setsid/nohup start command |
| `yam_real_serve.py` (fleet-started) | Orin | `/tmp/serve_managed.log` | agent spawns it (`relay.py` ~line 430), **append** mode — restarts accumulate |
| `camera_relay.py` (fleet-started) | Orin | `/tmp/camera_managed.log` | agent spawns it (~line 420), append mode |
| manually-started serve/camera | Orin | wherever YOUR redirect points | convention: `/tmp/<name>.log` |
| relay server | GCP VM | `journalctl -u blupe-relay` | systemd unit |

`/tmp` dies on reboot (both Mac and Orin) — logs are session-scoped by design. Anything
worth keeping past a reboot goes in `runs/` (trial recordings) or a doc, not a log.

## Line conventions (keep greppable)

- Every line starts with a `[tag]` naming the subsystem: `[eval]` `[dbg]` `[lat]` `[serve]`
  `[stereo]` `[camera]` `[connect]` `[stream]` `[xr-bridge]` `[off]` `[fake-quest]`.
  One tag per subsystem, no freeform prefixes — `grep "\[lat\]"` must catch ALL latency
  output on every tier.
- Always `print(..., flush=True)`. These processes run with redirected stdout; without
  flush, Python block-buffers and the log is minutes stale exactly when you need it.
- State CHANGES get a line (connect, disconnect, mode switch, reopen). Steady-state gets a
  periodic summary line, never per-tick spam: eval debug/latency every 2 s, serve/stereo
  every 5 s, serve applied-joints every 25 cmds.
- Numbers carry units in the text (`ms`, `cmd/s`, `Hz`) so a line is readable in isolation.

## Latency instrumentation (`[lat]` lines)

Built 2026-06-11 to localize teleop lag per data-flow stage. Format: `stage=avg/max` in ms
over the window, window drained on report (each line stands alone). Bare `stage=N` (no
`ms`) = pure event counter (e.g. dropped frames).

**Helper:** `LatencyStats` in `scripts/stereo_sender.py` (thread-safe; `note(stage, secs)`,
`count(stage)`, `report()`). The eval has a module-global `LAT`; the stereo server has its
own `self.lat`. To time a new stage anywhere on the Mac path: import/`note()` and it shows
up in the next `[lat]` line automatically — no other wiring.

**Stage map** (what each number means, healthy values from headless e2e):

| Stage | Tier (log) | Meaning | Healthy |
|---|---|---|---|
| `input_age` | eval | XR tick age when used (bridge mode only; absent under sdk/stub) | < ~25 ms (60 Hz bridge) |
| `ik` | eval | IK + targets compute | ~2 ms |
| `loop_busy` | eval | loop body time; > 20 ms avg = can't hold 50 Hz | ~3 ms |
| `cmd_queue` | eval | `set_target` → socket write (decoupled 50 Hz sender) | ~8–10 ms BY DESIGN |
| `cmd_write` | eval | socket write+flush | < 1 ms |
| `arm_rtt` | eval | cmd write → serve APPLIES → `{"ack":t}` back (full network+apply round trip, all on the Mac clock) | ~1 ms LAN; relay path = your network |
| `cam_age` | eval | freshest camera frame age at composite | < ~40 ms |
| `vid_queue` `vid_encode` `vid_send` | stereo (same log) | submit→encode wait, x264 encode, socket send | enc < ~15 ms |
| `vid_skip` / `vid_drop` | stereo | frames superseded before encode / dropped on backpressure | occasional |
| `gap` / `apply` | serve (Orin log) | inter-command spacing seen by the robot / i2rt apply time | gap ≈ 24 ms, apply < 1 ms |

**Reading lag:** big `arm_rtt` + normal `gap` → network to the robot. Big `gap` max +
normal Mac-side numbers → jitter on the path or sender stalls. Big `cam_age` → upstream
camera/relay (then use the infra skill's burned-in-timestamp playbook). Big `vid_*` →
Mac-side encode/send. `input_age` near 500 ms → bridge stale rule about to clutch-release.

**The single-clock rule (why arm_rtt is a round trip):** Mac/Orin/VM clocks are not
synced, and `time.monotonic()` is per-boot anyway. NEVER subtract timestamps from two
machines. Cross-machine latency = echo the sender's timestamp back and difference it on
the sender (`{"q":…, "t":…}` → serve applies → `{"ack": t}`; `t` optional, old peers
just don't ack). Same trick for any future cross-machine probe.

**Verify changes headlessly:** fake serve (`yam_real_serve.py --fake --port 5597`) +
`XR_INPUT=stub eval_yam_vr.py --quest-ip 127.0.0.1 --cameras none --serve-port 5597`,
then `grep "\[lat\]"` both logs. Stereo path: `stereo_sender.py --listen-port 23579` +
`fake_quest_stereo.py --control-port 23579 --video-port 23457` (note: fake quest exits
after 30 packets — shorter than the 5 s report window, so no `[lat]` line is expected
from that test alone).

**Deploy note:** serve-side `[lat]`/acks require the updated `yam_real_serve.py` ON THE
ORIN. Mismatched versions degrade gracefully (no acks → no `arm_rtt`, nothing breaks).
