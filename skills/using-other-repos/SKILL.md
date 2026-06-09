---
name: using-other-repos
description: How to integrate correctly with external/vendored repos (XRoboToolkit, i2rt, ...). Read the actual source on BOTH sides of an interface, enumerate the whole org, and match the reference implementation. Use when wiring our code to someone else's stack, or debugging an integration that "half-works."
---

# Using other repos (integrating with external/vendored code)

**The source is the spec.** READMEs and one-sided reading mislead. Read the real
implementation on **both sides** of any interface, and find the canonical/reference
implementation before writing your own. We learned this the slow way twice (below).

## Checklist
1. **Enumerate the whole org first** — `gh repo list <ORG> --limit 100`. A capability you think
   is missing often lives in a sibling repo.
2. **Read BOTH sides of a protocol.** The **receiver** defines what's actually required; the
   **sender** shows the exact params. Inferring one from the other leaves gaps.
3. **Find the reference/sample impl and match it exactly** — encoder flags, framing, bitrate,
   GOP, byte order. Diff your code against it; don't guess defaults.
4. **Look for a negotiation/handshake** — receivers often send a config request first and expect
   the stream/response to match. Don't just blast data.
5. **Don't trust standalone vendored files — read how the vendor's code USES them** at runtime
   (loaders, "combine"/assembly steps, overrides).

## Diagnostic signals
- **"It half-works"** (one frame decodes then drops; one config works, another doesn't) = a
  **protocol/negotiation mismatch**, not a pure format bug. Stop tweaking format — read the
  receiver + reference for the missing step.
- **"Self-consistent but wrong"** = you validated against your own assumption, not the real
  counterpart. Verify against the external side.

## Read fast without cloning
- `gh repo list <ORG>` — enumerate.
- `gh api "repos/<ORG>/<REPO>/git/trees/<BRANCH>?recursive=1" --jq '.tree[].path'` — list files
  (quote the URL — `?` is a shell glob).
- `gh api "repos/<ORG>/<REPO>/contents/<PATH>" --jq '.content' | base64 -d` — read a file.

## Episode 1 — XRoboToolkit video to the Quest (what we had to read)
Symptom: stream "half-worked" — the first keyframe decoded (arm flashed on the headset) then the
Quest dropped the connection. We'd only read the **sender** and inferred the protocol.
What actually settled it:
- `XRoboToolkit-Orin-Video-Sender/main_zed_tcp.cpp`, `network_helper.hpp` — sender + wire framing.
- `RobotVision-PC/VideoTransferPC/src/CameraDataReceiver.cpp` — receiver: `[4-byte BE len][H.264]`.
- `XRoboToolkit-Native-Video-Viewer/.../MediaDecoderTextureViewTCP.java` — Quest decoder
  (ServerSocket, MediaCodec, big-endian length).
- `XRoboToolkit-Unity-Client/Assets/Scripts/Camera/RemoteCameraWindow.cs` — the LISTEN flow +
  the `StartReceivePcCamera` request (Quest broadcasts the width/height/fps/bitrate it wants).
- `RobotVision-PC/.../CameraDataSender.cpp` + `H264Encoder.cpp` — the **reference sender**: exact
  encoder params (`ultrafast`, `zerolatency`, `baseline`, `annexb=1`, `yuv420p`, explicit
  `bit_rate`, `gop=fps*2`).
Lesson: the *format* was right; we **improvised the negotiation** — we ignored the Quest's
`StartReceivePcCamera` params and direct-connected with guessed resolution/bitrate. Match what the
receiver asks for. (Also: we first wrongly concluded "XRoboToolkit can't render to the headset"
from the SDK alone — the whole video pipeline was in sibling repos we hadn't enumerated.)

## Episode 2 — i2rt YAM model ground truth
The standalone `i2rt/.../yam/yam.xml` `link6` is a **placeholder** that
`combine_arm_and_gripper_xml` overrides at runtime (pose + joint axis from the gripper mount).
Targeting the standalone file gave a wrong EE frame that was self-consistent and passed every
test. Ground truth = how the code combines/loads it. See the `adding-an-arm` skill.
