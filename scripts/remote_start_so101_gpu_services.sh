set -euo pipefail

REMOTE_ROOT="$1"
PYTHON="$2"
LOG_DIR="$3"
HF_HOME_DIR="$4"
POLICY_PORT="$5"
SAM3_PORT="$6"
SAM2_PORT="$7"
CHECKPOINT_PATH="$8"
POLICY_PATH="$9"
if [ "$POLICY_PATH" = "__none__" ]; then
  POLICY_PATH=""
fi
LORA_LLM_PATH="${10}"
if [ "$LORA_LLM_PATH" = "__none__" ]; then
  LORA_LLM_PATH=""
fi
LORA_VISION_PATH="${11}"
if [ "$LORA_VISION_PATH" = "__none__" ]; then
  LORA_VISION_PATH=""
fi
NORM_TAG="${12}"
IMAGE_KEYS_B64="${13}"
MODEL_DTYPE="${14}"
FLOW_STEPS="${15}"
NUM_ACTIONS="${16}"
EXPERIMENTS_DIR="${17}"
SAM3_BACKEND="${18}"
SAM3_MODEL_ID="${19}"
SAM3_FRAMES_DIR="${20}"
SAM3_READY_PATH="${21}"
SAM2_TRACKER="${22}"
SAM2_MODEL_ID="${23}"
SAM3_VIDEO_MODEL_ID="${24}"
SAM3_VIDEO_PROMPT="${25}"
SAM3_DETECT_TIMEOUT_S="${26}"
SAM3_DETECT_REQUEST_TIMEOUT_S="${27}"
YOLO_BALL_MODEL="${28}"
YOLO_BALL_IMGSZ="${29}"
YOLO_BALL_CONF="${30}"
SAM3_VIDEO_MAX_SESSION_FRAMES="${31:-300}"
IMAGE_KEYS="$(printf '%s' "$IMAGE_KEYS_B64" | base64 -d)"

mkdir -p "$LOG_DIR" "$SAM3_FRAMES_DIR"
cd "${REMOTE_ROOT}/blupe-evals"

wait_local_http() {
  local url="$1"
  local timeout_s="$2"
  local label="$3"
  local attempts=$((timeout_s * 2))
  local i
  for ((i = 0; i < attempts; i++)); do
    if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
      echo "${label}: ready"
      return 0
    fi
    sleep 0.5
  done
  echo "${label}: not ready yet (${url})" >&2
  return 1
}

sam3_detect_once() {
  "$PYTHON" - "$SAM3_PORT" "${SAM3_FRAMES_DIR}/seed.jpg" "$SAM3_DETECT_REQUEST_TIMEOUT_S" <<'PY'
import base64
import json
import sys
import urllib.error
import urllib.request

port = sys.argv[1]
image_path = sys.argv[2]
timeout_s = float(sys.argv[3])

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "image_b64": image_b64,
    "prompts": ["light blue object"],
    "max_masks": 1,
    "min_score": 0.0,
    "alpha": 0.65,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/api/detect_image",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
        if resp.status != 200:
            print(f"status={resp.status} body={body[:1000]!r}", file=sys.stderr)
            sys.exit(1)
        json.loads(body.decode("utf-8"))
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    print(f"status={exc.code} body={body[:2000]}", file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(repr(exc), file=sys.stderr)
    sys.exit(1)
PY
}

wait_sam3_detect() {
  local timeout_s="$1"
  local label="$2"
  local attempts=$((timeout_s / 5))
  local i
  if [ "$attempts" -lt 1 ]; then
    attempts=1
  fi
  for ((i = 0; i < attempts; i++)); do
    if sam3_detect_once >/dev/null 2>"${LOG_DIR}/sam3_detect_${SAM3_PORT}.last_error"; then
      echo "${label}: ready"
      return 0
    fi
    sleep 5
  done
  echo "${label}: detect_image not ready after ${timeout_s}s" >&2
  cat "${LOG_DIR}/sam3_detect_${SAM3_PORT}.last_error" >&2 || true
  tail -120 "${LOG_DIR}/sam3_${SAM3_PORT}.log" >&2 || true
  return 1
}

if [ ! -f "${SAM3_FRAMES_DIR}/seed.jpg" ]; then
  "$PYTHON" - <<PY
from PIL import Image
Image.new("RGB", (640, 360), "black").save("${SAM3_FRAMES_DIR}/seed.jpg")
PY
fi

if ! curl -fsS --max-time 2 "http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}" >/dev/null 2>&1; then
  pkill -f 'scripts/sam3_prompt_ui.py' >/dev/null 2>&1 || true
  nohup "$PYTHON" scripts/sam3_prompt_ui.py \
    --frames-dir "$SAM3_FRAMES_DIR" \
    --host 127.0.0.1 \
    --port "$SAM3_PORT" \
    --backend "$SAM3_BACKEND" \
    --model-id "$SAM3_MODEL_ID" \
    >"${LOG_DIR}/sam3_${SAM3_PORT}.log" 2>&1 &
  echo $! >"${LOG_DIR}/sam3_${SAM3_PORT}.pid"
fi

case "$SAM2_TRACKER" in
  image)
    SAM2_SCRIPT="scripts/sam2_track_ui.py"
    SAM2_EXPECTED_MODE="sam2_image"
    TRACKER_EXTRA_ARGS=(--model-id "$SAM2_MODEL_ID")
    ;;
  video)
    SAM2_SCRIPT="scripts/sam2_video_track_ui.py"
    SAM2_EXPECTED_MODE="sam2_video"
    TRACKER_EXTRA_ARGS=(--model-id "$SAM2_MODEL_ID")
    ;;
  sam3_video)
    SAM2_SCRIPT="scripts/sam3_video_track_ui.py"
    SAM2_EXPECTED_MODE="sam3_video"
    TRACKER_EXTRA_ARGS=(
      --model-id "$SAM3_VIDEO_MODEL_ID"
      --prompt "$SAM3_VIDEO_PROMPT"
      --inference-state-device cpu
      --video-storage-device cpu
      --max-session-frames "$SAM3_VIDEO_MAX_SESSION_FRAMES"
    )
    ;;
  yolo)
    if [ -z "$YOLO_BALL_MODEL" ]; then
      echo "SO101_SAM2_TRACKER=yolo requires SO101_YOLO_BALL_MODEL=/path/to/best.pt" >&2
      exit 1
    fi
    SAM2_SCRIPT="scripts/yolo_ball_track_ui.py"
    SAM2_EXPECTED_MODE="yolo_seg"
    TRACKER_EXTRA_ARGS=(--model "$YOLO_BALL_MODEL" --imgsz "$YOLO_BALL_IMGSZ" --conf "$YOLO_BALL_CONF")
    ;;
  *)
    echo "Unknown SO101_SAM2_TRACKER=${SAM2_TRACKER}; expected image, video, sam3_video, or yolo" >&2
    exit 1
    ;;
esac

SAM2_HEALTH="$(curl -fsS --max-time 2 "http://127.0.0.1:${SAM2_PORT}/health" 2>/dev/null || true)"
if ! printf '%s' "$SAM2_HEALTH" | grep -q "\"mode\":\"${SAM2_EXPECTED_MODE}\""; then
  pkill -f 'scripts/sam3_video_track_ui.py' >/dev/null 2>&1 || true
  pkill -f 'scripts/sam2_video_track_ui.py' >/dev/null 2>&1 || true
  pkill -f 'scripts/sam2_track_ui.py' >/dev/null 2>&1 || true
  pkill -f 'scripts/yolo_ball_track_ui.py' >/dev/null 2>&1 || true
  nohup "$PYTHON" "$SAM2_SCRIPT" \
    --host 127.0.0.1 \
    --port "$SAM2_PORT" \
    --device cuda \
    "${TRACKER_EXTRA_ARGS[@]}" \
    >"${LOG_DIR}/sam2_${SAM2_PORT}.log" 2>&1 &
  echo $! >"${LOG_DIR}/sam2_${SAM2_PORT}.pid"
fi

if ! curl -fsS --max-time 2 "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null 2>&1; then
  pkill -f 'scripts/molmoact2_policy_runner.py' >/dev/null 2>&1 || true
  checkpoint_args=(--checkpoint-path "$CHECKPOINT_PATH")
  if [ -n "$POLICY_PATH" ]; then
    checkpoint_args=(--policy-path "$POLICY_PATH")
  elif [ -n "$NORM_TAG" ]; then
    checkpoint_args+=(--norm-tag "$NORM_TAG")
  fi
  if [ -n "$LORA_LLM_PATH" ]; then
    checkpoint_args+=(--lora-llm-path "$LORA_LLM_PATH")
  fi
  if [ -n "$LORA_VISION_PATH" ]; then
    checkpoint_args+=(--lora-vision-path "$LORA_VISION_PATH")
  fi
  env_args=(HF_HOME="$HF_HOME_DIR")
  if [ -d "$EXPERIMENTS_DIR" ]; then
    env_args+=(PYTHONPATH="${EXPERIMENTS_DIR}:${EXPERIMENTS_DIR}/lerobot/src:${PYTHONPATH:-}")
  fi
  nohup env "${env_args[@]}" "$PYTHON" scripts/molmoact2_policy_runner.py \
    --host 127.0.0.1 \
    --port "$POLICY_PORT" \
    "${checkpoint_args[@]}" \
    --image-keys "$IMAGE_KEYS" \
    --device cuda \
    --model-dtype "$MODEL_DTYPE" \
    --num-flow-timesteps "$FLOW_STEPS" \
    --num-actions "$NUM_ACTIONS" \
    --no-enable-cuda-graph \
    >"${LOG_DIR}/molmoact2_${POLICY_PORT}.log" 2>&1 &
  echo $! >"${LOG_DIR}/molmoact2_${POLICY_PORT}.pid"
fi

wait_local_http "http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}" 60 "remote SAM3 port"
wait_sam3_detect "$SAM3_DETECT_TIMEOUT_S" "remote SAM3 detect"
wait_local_http "http://127.0.0.1:${SAM2_PORT}/health" 60 "remote ball tracker"
wait_local_http "http://127.0.0.1:${POLICY_PORT}/health" 240 "remote MolmoAct2"
