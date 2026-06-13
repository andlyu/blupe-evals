# The serve — specification

The **serve** is the program YOU run on the robot computer. It is the only process that
touches the motors, and it is where ALL safety lives. The rest of the stack — operator,
relay, cloud — only ever exchanges joint vectors with it over this protocol. If your
embodiment ships a reference serve (YAM: `YAM_control/yam_real_serve.py`), run that and
skip to "Verify". If you're implementing one for your own driver, this page is the
complete contract; the reference implementation is ~200 lines.

## Transport

- TCP server, default port **5599**, bound on the robot computer (localhost/LAN — never
  exposed to the internet; the relay agent carries it outward).
- **Newline-delimited JSON**: one JSON object per line, both directions.
- **One client at a time** (`listen(1)`; serve clients sequentially). A client disconnect
  returns you to accepting; the next client gets a fresh handshake.
- Ignore unknown fields and lines without `"q"` — the protocol grows by optional fields
  (that's how `"t"`/acks were added without breaking old serves).

## Messages

### 1. Handshake — you send, once, immediately on every client connect

```json
{"start_joints": [0.41, 1.52, 0.65, -0.08, 0.0, -0.06]}
```

- The arm's **REAL, current** joint positions — read from the motors, not assumed.
- **Radians**, in the embodiment's joint order (same order as its URDF/ArmSpec).
- This is the no-jump seed: the operator ramps from these values, so lying here (e.g.
  sending zeros) makes the first command a violent jump. If you cannot read the motors,
  do not send a handshake — close/exit nonzero instead; the fleet preflight reports that
  honestly as "arm not ready".

### 2. Command stream — client sends, nominally ~50 Hz

```json
{"q": [0.40, 1.50, 0.66, -0.08, 0.0, -0.06], "g": 0.0, "t": 12345.678}
```

- `q` — **absolute target** joint positions, radians, same order/length as the handshake.
- `g` — optional gripper target, **normalized 0..1, 0 = closed, 1 = open** (never raw
  motor units). Absent = leave the gripper alone.
- `t` — optional opaque value. If present, echo it back (see acks). Never interpret it —
  it's the sender's clock, not yours.
- Tolerate ANY arrival rate: bursts, gaps, stalls. The reference clamps the inter-command
  `dt` to `[0.0001, 0.1] s` before using it in the velocity limit, so a 5-second stall
  can't translate into a 5-second-sized step when traffic resumes.

### 3. Ack — you send, only when `t` was present, only AFTER applying

```json
{"ack": 12345.678}
```

Echo `t` verbatim **after** the command reached the motor driver — the sender measures
write→apply round-trip on its own clock. Ack before applying and the latency numbers lie.

### 4. Shutdown — client sends when quitting cleanly

```json
{"shutdown": true}
```

→ cut motor torque (arm goes limp safely), then exit.

## The safety contract — non-negotiable, enforced HERE

These live in the serve precisely so nothing upstream — operator bug, network fault,
compromised cloud — can bypass them:

1. **Velocity-clamp every applied command.** Per tick: `step = MAX_VEL × dt` (dt bounded
   as above); move each joint at most `±step` toward its target. The reference uses
   `MAX_VEL = 0.6 rad/s`. Whatever arrives, the arm moves gently.
2. **Hold on disconnect.** Socket drops → keep the last applied pose. Do NOT torque off
   (a limp arm drops what it's holding), do NOT keep ramping toward the last target.
3. **Torque off on `shutdown` AND on your own exit paths** (Ctrl-C, exceptions). Mind
   driver quirks: if your driver runs a control thread that re-energizes motors, stop it
   *before* cutting torque (see `disable_motorchain` in the reference — this exact bug
   cost us an evening).
4. **Bounded-force gripper.** Don't slam the gripper to its target: walk it a small
   bounded step per tick toward `g` (reference: lead ≤ ~0.076 of full travel). A blocked
   jaw then stalls with bounded force — no grinding, no crushing — and objects of any
   size are gripped with the same gentle pressure.

## Skeleton (pseudocode of the entire serve)

```python
srv = listen(5599)
while True:
    conn = srv.accept()
    last = read_motor_positions()                 # truth, not assumption
    send(conn, {"start_joints": last})
    for msg in jsonlines(conn):                   # disconnect -> hold `last`, outer loop
        if msg.get("shutdown"): torque_off(); exit()
        if "q" not in msg: continue
        dt   = clamp(now() - prev_time, 1e-4, 0.1); prev_time = now()
        last = last + clip(msg["q"] - last, -MAX_VEL*dt, +MAX_VEL*dt)   # rule 1
        grip = walk_bounded(current_grip, msg.get("g"))                  # rule 4
        apply_to_driver(last, grip)
        if "t" in msg: send(conn, {"ack": msg["t"]})                     # after apply
```

## Verify

```bash
python3 scripts/check_robot_setup.py            # handshake + cameras + relay + token
printf '' | nc 127.0.0.1 5599                   # quick look: must print {"start_joints": ...}
```

The fleet UI's **Check** runs your embodiment's preflight remotely; the headset HUD shows
`ROBOT OFF` until your handshake arrives — operators always know if they're commanding metal.
