# SO101 4090 SAM Eval Setup

This is the lightweight eval stack for running SO101 locally while offloading
vision and policy inference to a Vast 4090.

## Services

- MolmoAct2 policy HTTP server: `127.0.0.1:8202`
- SAM3 image prompt server: `127.0.0.1:8213`
- SAM2 tracker: `127.0.0.1:8214`
- Local SO101 eval UI: `http://127.0.0.1:8092/#setup`

## Pipeline Workflow

Start everything:

```bash
scripts/pipeline.sh launch so101-eval
```

Verify everything is reachable:

```bash
scripts/pipeline.sh check so101-eval
```

Stop everything:

```bash
scripts/pipeline.sh stop so101-eval
```

Restart everything:

```bash
scripts/pipeline.sh restart so101-eval
```

The start script verifies or launches the GPU MolmoAct2/SAM3/SAM2 services, opens
local SSH tunnels for `8202`, `8213`, and `8214`, then starts the local camera relay
and eval UI. The stop script first asks the UI to stop recording/eval/policy motion,
then stops the UI, camera relay, tunnels, and remote GPU services.
SAM3 readiness defaults to `/`, because some deployed SAM3 service copies predate
the newer `/health` route.

The launcher also does the two checks that matter for live masks:

- If the remote Hugging Face token is missing and the Mac has
  `~/.cache/huggingface/token`, it copies that token to the remote HF cache without
  printing it. This is required for gated `facebook/sam3`.
- It waits for a real SAM3 `/api/detect_image` request to complete before printing
  that the stack is up. A reachable SAM3 web page is not enough.

The standalone check command runs the same deep SAM3 detection smoke test by
default. Disable that only for low-level debugging:

```bash
SO101_CHECK_SAM3_DETECT=0 scripts/check_so101_eval_stack.sh
```

When the Vast SSH target changes, copy `config/so101_eval_stack.env.example` to
`config/so101_eval_stack.local.env` and update `SO101_GPU_HOST` / `SO101_GPU_PORT`.

The success tracker uses SAM3 to seed the cup/cylinder mask. The ball mask is
refreshed by the configured tracker service behind `/api/track_image`. The
default is the low-latency SAM2 image tracker (`SO101_SAM2_TRACKER=image`), but
live evals can use SAM3 Video tracking by setting
`SO101_SAM2_TRACKER=sam3_video` and pointing `SO101_SAM2_PORT` at the SAM3 Video
tracker port, usually `8216`. The variable name is legacy; the tracker health
response reports `mode: sam3_video` when SAM3 Video is active. Tracker requests
run in the background every
`SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES=2` success-tracker frames, about 5 Hz at
the default 10 Hz success loop, while `/api/success.mjpg` keeps drawing fresh
camera frames with the latest completed mask. Periodic SAM3 ball re-grounding is
off by default with `SO101_SUCCESS_BALL_SAM3_EVERY_N_FRAMES=0`; the operator can
still force it with `Rerun SAM3`. The status payload reports actual mask
latency, inference Hz, and SAM2-vs-SAM3 alignment telemetry. Current defaults
are `black cylinder along with insides` for the cup/cylinder at minimum score
`0.25`, and `light blue object` for the ball at minimum score `0.25`.

## 4090 Setup

On the 4090 instance, use the repo checkout and vision environment:

```bash
cd /workspace/blupe-evals
source /venv/main/bin/activate
```

SAM2 must be installed in that environment. If it is missing:

```bash
git clone https://github.com/facebookresearch/sam2.git /root/sam2
python -m pip install -e /root/sam2
```

SAM3 can run either through the local native SAM3 checkout or the Transformers
`facebook/sam3` backend. The native backend expects the SAM3 assets under
`/root/sam3`; the Transformers backend uses the Hugging Face model path.
`facebook/sam3` is gated, so the remote environment must have a valid Hugging Face
token in `/root/.cache/huggingface/token` or the first `/api/detect_image` request
will fail with a 401 wrapped as HTTP 400.

The SAM3 service needs a frames directory at startup, even though live eval calls
use `/api/detect_image`. Put any valid image in it:

```bash
mkdir -p /tmp/sam3-frames
python - <<'PY'
from PIL import Image
Image.new("RGB", (640, 360), "black").save("/tmp/sam3-frames/seed.jpg")
PY
```

Launch SAM3 and SAM2 manually if needed:

```bash
screen -L -dmS sam_4090_8213_8214 bash -lc '
cd /workspace/blupe-evals
source /venv/main/bin/activate
python scripts/sam3_prompt_ui.py --frames-dir /tmp/sam3-frames --host 127.0.0.1 --port 8213 --backend auto &
python scripts/sam2_track_ui.py --host 127.0.0.1 --port 8214 --device cuda --model-id facebook/sam2.1-hiera-base-plus
'
```

Health checks:

```bash
curl -fsS http://127.0.0.1:8213/health
curl -fsS http://127.0.0.1:8214/health
```

## Tunnels To The Mac

From the Mac, tunnel the remote services:

```bash
ssh -N \
  -L 8202:127.0.0.1:8202 \
  -L 8213:127.0.0.1:8213 \
  -L 8214:127.0.0.1:8214 \
  -p <vast-ssh-port> root@<vast-host>
```

## Local UI Launch

Run this from the Mac repo checkout:

```bash
SO101_PYTHON=/Users/andrew/miniconda3/bin/python \
SO101_CAMERA_RELAY_DEVICES="0 1 2" \
SO101_CAMERA_SPECS="front=http://127.0.0.1:8089/2 side=http://127.0.0.1:8089/1 wrist=http://127.0.0.1:8089/0" \
SO101_POLICY_URL=http://127.0.0.1:8202 \
SO101_POLICY_CAMERAS=front,wrist \
SO101_SUCCESS_SAM3_URL=http://127.0.0.1:8213/api/detect_image \
SO101_SUCCESS_SAM3_PROMPT="black cylinder along with insides" \
SO101_SUCCESS_SAM3_MIN_SCORE=0.25 \
SO101_SUCCESS_BALL_SAM3_PROMPT="light blue object" \
SO101_SUCCESS_BALL_SAM3_MIN_SCORE=0.25 \
SO101_SUCCESS_BALL_SAM3_EVERY_N_FRAMES=0 \
SO101_SAM2_TRACKER=image \
SO101_SUCCESS_BALL_SAM2_AUTO=1 \
SO101_SUCCESS_BALL_SAM2_URL=http://127.0.0.1:8214/api/track_image \
SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES=2 \
SO101_POLICY_REALTIME_CHUNKING=0 \
SO101_OPEN_BROWSER=0 \
scripts/launch_so101_eval_ui.sh
```

Open `http://127.0.0.1:8092/#setup`. In Live View, the camera row should be:

```text
Front | Side | Wrist | Masks
```

Use `Rerun SAM3` on the masks tile after changing object placement or prompts.
