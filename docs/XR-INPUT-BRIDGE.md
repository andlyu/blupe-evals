# XR input bridge — wire protocol v1

The contract between the **container side** (`docker/xr-bridge/bridge.py`, has the real
`xrobotoolkit_sdk`) and the **host side** (the eval's input seam / `xrobotoolkit_sdk` shim).
Both implementations MUST follow this file. If you need to change the protocol, change this
file first, then both sides.

## Transport
- TCP. The bridge **listens** on `0.0.0.0:8765` inside the container (publish with
  `-p 8765:8765`; host consumers dial `127.0.0.1:8765`).
- Newline-delimited JSON (one object per line, UTF-8), same style as `yam_real_serve.py`.
- The stream is **one-way**: server → client. Clients send nothing. Reconnect = fresh hello.

## Messages
On client connect, the bridge sends one hello line:

```json
{"hello": {"v": 1, "rate_hz": 60}}
```

Then state ticks at `rate_hz`, one per line:

```json
{"t": 12.345,
 "right": {"pose": [0,0,0, 0,0,0,1], "grip": 0.0, "trigger": 0.0,
           "axis": [0.0, 0.0], "axis_click": false, "A": false, "B": false},
 "left":  {"pose": [0,0,0, 0,0,0,1], "grip": 0.0, "trigger": 0.0,
           "axis": [0.0, 0.0], "axis_click": false, "X": false, "Y": false},
 "head":  {"pose": [0,0,0, 0,0,0,1]}}
```

## Field semantics
- `t` — seconds, monotonic clock on the bridge host. Consumers use it only for staleness.
- `pose` — `[x, y, z, qx, qy, qz, qw]`, **exactly as the SDK returns it** (no reframing,
  no unit changes — the consumer reframes). This matches
  `xrt.get_right_controller_pose()` / `get_left_controller_pose()` / `get_headset_pose()`.
- `grip`, `trigger` — floats 0..1, from `get_*_grip()` / `get_*_trigger()`.
- `axis` — thumbstick `[x, y]` floats, from the SDK's axis getter.
- `axis_click`, `A`, `B`, `X`, `Y` — JSON booleans, from `get_*_axis_click()` /
  `get_A_button()` etc. (A/B live on the right controller, X/Y on the left.)
- Any field the SDK can't provide: send `null`. **Consumers must tolerate `null`** (treat as
  "no data": identity pose, 0.0, false).
- Field names mirror the SDK getters; if the bridge exposes additional SDK state, add keys —
  consumers must ignore unknown keys.

## Consumer behavior (host side)
- Keep only the **latest** tick (drop stale lines on slow reads; never queue unboundedly).
- If no tick for > 0.5 s, treat input as lost: report not-tracking (eval already treats "no
  clutch" as freeze, so lost input must read as grip=0.0 — i.e. clutch released, arm freezes).
- The host shim presents the **same Python API as `xrobotoolkit_sdk`** (`init()`,
  `get_right_controller_pose()`, `get_right_grip()`, …) so the teleop framework and eval run
  unmodified. Selection via env: `XR_INPUT=bridge|sdk|stub`, `XR_BRIDGE_HOST`,
  `XR_BRIDGE_PORT` (defaults `127.0.0.1:8765`).
- `stub` = scripted/replayed motion for headless tests (no Quest, no container).
