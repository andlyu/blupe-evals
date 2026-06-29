# SO101 4090 SAM Eval Setup

This is the lightweight eval stack for running SO101 locally while offloading
vision and policy inference to a Vast 4090.

## Services

- MolmoAct2 policy HTTP server: `127.0.0.1:8202`
- SAM3 image prompt server: `127.0.0.1:8213`
- SAM2 stateful video tracker: `127.0.0.1:8214`
- Local SO101 eval UI: `http://127.0.0.1:8092/#setup`

## Two-Command Workflow

Start everything:

```bash
scripts/start_so101_eval_stack.sh
```

Verify everything is reachable:

```bash
scripts/check_so101_eval_stack.sh
```

Stop everything:

```bash
scripts/stop_so101_eval_stack.sh
```

The start script verifies or launches the GPU MolmoAct2/SAM3/SAM2 services, opens
local SSH tunnels for `8202`, `8213`, and `8214`, then starts the local camera relay
and eval UI. The stop script first asks the UI to stop recording/eval/policy motion,
then stops the UI, camera relay, tunnels, and remote GPU services.
SAM3 readiness defaults to `/`, because some deployed SAM3 service copies predate
the newer `/health` route.

When the Vast SSH target changes, copy `config/so101_eval_stack.env.example` to
`config/so101_eval_stack.local.env` and update `SO101_GPU_HOST` / `SO101_GPU_PORT`.

The success tracker uses SAM3 to seed the cup/cylinder mask and to refresh the
ball mask periodically. The stack default refreshes the ball with SAM3 every
`SO101_SUCCESS_BALL_SAM3_EVERY_N_FRAMES=100` success-tracker frames, about every
10 seconds at the default 10 Hz success loop, then uses SAM2 as the bridge
between SAM3 refreshes. SAM2 requests run in the background every
`SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES=2` success-tracker frames, about 5 Hz at
the default 10 Hz success loop, while `/api/success.mjpg` keeps drawing fresh
camera frames with the latest completed mask. Current defaults are `black
cylinder along with insides` for the cup/cylinder at minimum score `0.25`, and
`blue rubber ball` for the ball at minimum score `0.25`.

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

The SAM3 service needs a frames directory at startup, even though live eval calls
use `/api/detect_image`. Put any valid image in it:

```bash
mkdir -p /tmp/sam3-frames
python - <<'PY'
from PIL import Image
Image.new("RGB", (640, 360), "black").save("/tmp/sam3-frames/seed.jpg")
PY
```

Launch SAM3 and SAM2:

```bash
screen -L -dmS sam_4090_8213_8214 bash -lc '
cd /workspace/blupe-evals
source /venv/main/bin/activate
python scripts/sam3_prompt_ui.py --frames-dir /tmp/sam3-frames --host 127.0.0.1 --port 8213 --backend auto &
python scripts/sam2_video_track_ui.py --host 127.0.0.1 --port 8214 --device cuda --model-id facebook/sam2-hiera-tiny
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
SO101_SUCCESS_BALL_SAM3_PROMPT="blue rubber ball" \
SO101_SUCCESS_BALL_SAM3_MIN_SCORE=0.25 \
SO101_SUCCESS_BALL_SAM3_EVERY_N_FRAMES=100 \
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
