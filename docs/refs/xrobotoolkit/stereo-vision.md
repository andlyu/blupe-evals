# Stereo Remote Vision (the "ZEDMINI" flow) — wire protocol

How the Quest app receives side-by-side stereo video. **The direction is REVERSED vs the
mono Remote Vision flow**: mono = we dial the Quest's LISTEN port and push; stereo = the
Quest dials US first, tells us where to send, and then we dial back. That makes the stereo
flow robust to the Quest re-opening its panel (a fresh `OPEN_CAMERA` arrives each time) —
the mono LISTEN port has no such re-handshake, which is why it can wedge into black video.

Verified against both ends' upstream source:
- Sender (the working wire peer): `XRoboToolkit-Orin-Video-Sender` → `main_zed_tcp.cpp`
- Receiver (the headset app): `XRoboToolkit-Unity-Client` →
  `CameraRequestSerializer.cs`, `NetworkDataProtocolSerializer.cs`, `TcpManager.cs`

Our implementations: `scripts/stereo_sender.py` (server + standalone CLI),
`scripts/fake_quest_stereo.py` (plays the Quest's side — headless e2e test),
`scripts/mac_quest_bridge.py --video stereo` (integrated: VIEW toggles cameras ↔ stereo sim render).

## Sequence

1. Robot side LISTENS on TCP **:13579** (hardcoded in the app's `TcpManager.cs`).
2. Quest: Camera panel → video source **ZEDMINI** → **Listen** → enter the robot host's IP.
   The app connects to :13579 and sends `OPEN_CAMERA` carrying `CameraRequestData`:
   its own IP, its video (decoder) port **12345**, and the preset frame size
   (ZEDMINI preset = **2560x720 @ 60**, i.e. 1280x720 per eye).
3. Robot side connects back to `quest_ip:12345` and streams H.264 side-by-side frames.
4. `CLOSE_CAMERA` (or either socket dropping) stops the stream; keep listening for the next
   `OPEN_CAMERA`.

## Control messages (port 13579, Quest → us)

Outer framing: `[4-byte BIG-endian length][body]`.

Body = `NetworkDataProtocol`: `[int32 LE cmdLen][cmd utf-8][int32 LE dataLen][data]`.
Commands seen: `OPEN_CAMERA`, `CLOSE_CAMERA`.

`CameraRequestData` (the `data` of OPEN_CAMERA):

| offset | field |
|---|---|
| 0 | magic `0xCA 0xFE` |
| 2 | protocol version, 1 byte (= 1) |
| 3 | 7 × int32 LE: width, height, fps, bitrate, mv_hevc, render_mode, port |
| 31 | `[1-byte len][camera-type utf-8]` (e.g. `ZEDMINI`) |
| … | `[1-byte len][quest IP utf-8]` |

`width` is the FULL side-by-side width (left eye | right eye). `render_mode=2` = stereo.

## Video stream (we dial quest_ip:12345)

Same wire format as the mono flow: `[4-byte BIG-endian length][H.264 Annex-B]`.
Fresh encoder per connection so the stream leads with SPS/PPS + IDR. Proven config:
libx264, yuv420p, baseline profile, `preset=ultrafast tune=zerolatency g=15`.

## Headset-side behavior / gotchas

- The app renders one half per eye; **left half = left eye**. If depth looks inside-out,
  the eyes are crossed — swap them (`--swap`, or swap the `--cameras` order in the eval).
- **B on the right controller toggles flat ↔ 3D in the app.** This COLLIDES with our eval's
  B = GO_HOME shortcut: pressing B in stereo view also sends the arm home.
- The Quest still kills all sockets on sleep (doze) — but recovery here is automatic-ish:
  re-opening the camera panel re-sends `OPEN_CAMERA`; no IP re-entry needed on our side.
