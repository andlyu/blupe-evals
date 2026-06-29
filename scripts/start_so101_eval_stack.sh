#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the local SO101 eval stack.
#
# Starts or verifies:
#   - remote MolmoAct2 policy server on :8202
#   - remote SAM3 prompt server on :8213
#   - remote SAM2 video tracker on :8214
#   - local SSH tunnels for :8202/:8213/:8214
#   - local camera relay on :8089
#   - local eval UI on :8092
#
# Optional per-machine overrides can live in:
#   config/so101_eval_stack.local.env

REPO_ROOT="${BLUPE_EVALS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOCAL_ENV="${SO101_EVAL_STACK_ENV:-${REPO_ROOT}/config/so101_eval_stack.local.env}"
if [ -f "$LOCAL_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$LOCAL_ENV"
  set +a
fi

GPU_HOST="${SO101_GPU_HOST:-ssh2.vast.ai}"
GPU_PORT="${SO101_GPU_PORT:-12394}"
GPU_USER="${SO101_GPU_USER:-root}"
GPU_SSH_KEY="${SO101_GPU_SSH_KEY:-}"
REMOTE_ROOT="${SO101_GPU_REMOTE_ROOT:-/workspace}"
REMOTE_PYTHON="${SO101_GPU_PYTHON:-/venv/main/bin/python}"
REMOTE_LOG_DIR="${SO101_GPU_LOG_DIR:-${REMOTE_ROOT}/logs}"
REMOTE_HF_HOME="${SO101_GPU_HF_HOME:-/root/.cache/huggingface}"

GPU_SERVICES="${SO101_START_GPU_SERVICES:-1}"
TUNNELS="${SO101_START_TUNNELS:-1}"
LOCAL_UI="${SO101_START_LOCAL_UI:-1}"
REPLACE_PORTS="${SO101_REPLACE_PORTS:-1}"

POLICY_PORT="${SO101_POLICY_PORT:-8202}"
SAM3_PORT="${SO101_SAM3_PORT:-8213}"
SAM2_PORT="${SO101_SAM2_PORT:-8214}"
TUNNEL_SCREEN="${SO101_GPU_TUNNEL_SCREEN:-so101_gpu_tunnels_8202_8213_8214}"
TUNNEL_LOG="${SO101_GPU_TUNNEL_LOG:-/tmp/${TUNNEL_SCREEN}.log}"
GPU_READY_TIMEOUT_S="${SO101_GPU_READY_TIMEOUT_S:-240}"
TUNNEL_READY_TIMEOUT_S="${SO101_TUNNEL_READY_TIMEOUT_S:-30}"

MOLMOACT2_CHECKPOINT_PATH="${MOLMOACT2_CHECKPOINT_PATH:-allenai/MolmoAct2-SO100_101}"
MOLMOACT2_POLICY_PATH="${MOLMOACT2_POLICY_PATH:-}"
REMOTE_POLICY_PATH="${MOLMOACT2_POLICY_PATH:-__none__}"
MOLMOACT2_NORM_TAG="${MOLMOACT2_NORM_TAG:-so100_so101_molmoact2}"
MOLMOACT2_IMAGE_KEYS="${MOLMOACT2_IMAGE_KEYS:-[\"observation.images.front\",\"observation.images.wrist\"]}"
MOLMOACT2_MODEL_DTYPE="${MOLMOACT2_MODEL_DTYPE:-float16}"
MOLMOACT2_NUM_FLOW_TIMESTEPS="${MOLMOACT2_NUM_FLOW_TIMESTEPS:-10}"
MOLMOACT2_NUM_ACTIONS="${MOLMOACT2_NUM_ACTIONS:-30}"
MOLMOACT2_EXPERIMENTS_DIR="${MOLMOACT2_EXPERIMENTS_DIR:-${REMOTE_ROOT}/molmoact2/experiments}"

SAM3_BACKEND="${SO101_SAM3_BACKEND:-transformers}"
SAM3_FRAMES_DIR="${SO101_SAM3_FRAMES_DIR:-${REMOTE_ROOT}/sam3-frames}"
SAM3_READY_PATH="${SO101_SAM3_READY_PATH:-/}"
SAM2_MODEL_ID="${SO101_SAM2_MODEL_ID:-facebook/sam2-hiera-tiny}"

UI_PORT="${SO101_WEB_PORT:-8092}"
CAMERA_RELAY_PORT="${SO101_CAMERA_RELAY_PORT:-8089}"
UI_URL="${SO101_UI_URL:-http://127.0.0.1:${UI_PORT}/#setup}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -p "$GPU_PORT"
)
if [ -n "$GPU_SSH_KEY" ]; then
  SSH_OPTS=(-i "$GPU_SSH_KEY" "${SSH_OPTS[@]}")
fi

stop_screen() {
  local name="$1"
  screen -S "$name" -X quit >/dev/null 2>&1 || true
}

stop_port_listener() {
  local port="$1"
  local label="$2"
  local pids
  if [ "$REPLACE_PORTS" = "0" ] || ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return 0
  fi
  echo "Stopping existing ${label} listener on :${port} (pid(s): ${pids})"
  kill $pids 2>/dev/null || true
  for _ in {1..20}; do
    if [ -z "$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]; then
      return 0
    fi
    sleep 0.1
  done
  kill -KILL $pids 2>/dev/null || true
}

wait_http() {
  local url="$1"
  local timeout_s="$2"
  local label="$3"
  local attempts=$((timeout_s * 2))
  local i
  if [ "$attempts" -lt 1 ]; then
    attempts=1
  fi
  for ((i = 0; i < attempts; i++)); do
    if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
      echo "${label}: ready"
      return 0
    fi
    sleep 0.5
  done
  echo "${label}: not ready after ${timeout_s}s (${url})" >&2
  return 1
}

post_json_best_effort() {
  local url="$1"
  curl -fsS --max-time 2 \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$url" >/dev/null 2>&1 || true
}

stop_existing_local_motion() {
  echo "Stopping any existing local SO101 recording/eval/policy before restart"
  post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/record/stop"
  post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/eval/stop"
  post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/stop"
}

start_remote_gpu_services() {
  echo "Ensuring GPU services on ${GPU_USER}@${GPU_HOST}:${GPU_PORT}"
  ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" bash -s -- \
    "$REMOTE_ROOT" \
    "$REMOTE_PYTHON" \
    "$REMOTE_LOG_DIR" \
    "$REMOTE_HF_HOME" \
    "$POLICY_PORT" \
    "$SAM3_PORT" \
    "$SAM2_PORT" \
    "$MOLMOACT2_CHECKPOINT_PATH" \
    "$REMOTE_POLICY_PATH" \
    "$MOLMOACT2_NORM_TAG" \
    "$MOLMOACT2_IMAGE_KEYS" \
    "$MOLMOACT2_MODEL_DTYPE" \
    "$MOLMOACT2_NUM_FLOW_TIMESTEPS" \
    "$MOLMOACT2_NUM_ACTIONS" \
    "$MOLMOACT2_EXPERIMENTS_DIR" \
    "$SAM3_BACKEND" \
    "$SAM3_FRAMES_DIR" \
    "$SAM3_READY_PATH" \
    "$SAM2_MODEL_ID" <<'REMOTE'
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
NORM_TAG="${10}"
IMAGE_KEYS="${11}"
MODEL_DTYPE="${12}"
FLOW_STEPS="${13}"
NUM_ACTIONS="${14}"
EXPERIMENTS_DIR="${15}"
SAM3_BACKEND="${16}"
SAM3_FRAMES_DIR="${17}"
SAM3_READY_PATH="${18}"
SAM2_MODEL_ID="${19}"

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
    >"${LOG_DIR}/sam3_${SAM3_PORT}.log" 2>&1 &
  echo $! >"${LOG_DIR}/sam3_${SAM3_PORT}.pid"
fi

if ! curl -fsS --max-time 2 "http://127.0.0.1:${SAM2_PORT}/health" >/dev/null 2>&1; then
  pkill -f 'scripts/sam2_video_track_ui.py' >/dev/null 2>&1 || true
  nohup "$PYTHON" scripts/sam2_video_track_ui.py \
    --host 127.0.0.1 \
    --port "$SAM2_PORT" \
    --device cuda \
    --model-id "$SAM2_MODEL_ID" \
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

wait_local_http "http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}" 60 "remote SAM3"
wait_local_http "http://127.0.0.1:${SAM2_PORT}/health" 60 "remote SAM2"
wait_local_http "http://127.0.0.1:${POLICY_PORT}/health" 240 "remote MolmoAct2"
REMOTE
}

start_tunnels() {
  echo "Starting local GPU tunnels on :${POLICY_PORT}, :${SAM3_PORT}, :${SAM2_PORT}"
  stop_screen "$TUNNEL_SCREEN"
  stop_port_listener "$POLICY_PORT" "MolmoAct2 tunnel"
  stop_port_listener "$SAM3_PORT" "SAM3 tunnel"
  stop_port_listener "$SAM2_PORT" "SAM2 tunnel"

  local screenrc="/tmp/${TUNNEL_SCREEN}.screenrc"
  printf 'logfile %s\nlogfile flush 1\n' "$TUNNEL_LOG" >"$screenrc"
  screen -c "$screenrc" -L -dmS "$TUNNEL_SCREEN" \
    ssh "${SSH_OPTS[@]}" \
      -N \
      -L "${POLICY_PORT}:127.0.0.1:${POLICY_PORT}" \
      -L "${SAM3_PORT}:127.0.0.1:${SAM3_PORT}" \
      -L "${SAM2_PORT}:127.0.0.1:${SAM2_PORT}" \
      "${GPU_USER}@${GPU_HOST}"

  wait_http "http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}" "$TUNNEL_READY_TIMEOUT_S" "local SAM3 tunnel"
  wait_http "http://127.0.0.1:${SAM2_PORT}/health" "$TUNNEL_READY_TIMEOUT_S" "local SAM2 tunnel"
  wait_http "http://127.0.0.1:${POLICY_PORT}/health" "$GPU_READY_TIMEOUT_S" "local MolmoAct2 tunnel"
}

start_local_ui() {
  echo "Starting local camera relay and SO101 eval UI"
  SO101_POLICY_URL="${SO101_POLICY_URL:-http://127.0.0.1:${POLICY_PORT}}" \
  SO101_POLICY_CAMERAS="${SO101_POLICY_CAMERAS:-front,wrist}" \
  SO101_SUCCESS_SAM3_URL="${SO101_SUCCESS_SAM3_URL:-http://127.0.0.1:${SAM3_PORT}/api/detect_image}" \
  SO101_SUCCESS_SAM3_PROMPT="${SO101_SUCCESS_SAM3_PROMPT:-black cylinder along with insides}" \
  SO101_SUCCESS_SAM3_MIN_SCORE="${SO101_SUCCESS_SAM3_MIN_SCORE:-0.25}" \
  SO101_SUCCESS_BALL_SAM3_PROMPT="${SO101_SUCCESS_BALL_SAM3_PROMPT:-blue rubber ball}" \
  SO101_SUCCESS_BALL_SAM3_MIN_SCORE="${SO101_SUCCESS_BALL_SAM3_MIN_SCORE:-0.25}" \
  SO101_SUCCESS_BALL_SAM2_AUTO="${SO101_SUCCESS_BALL_SAM2_AUTO:-1}" \
  SO101_SUCCESS_BALL_SAM2_URL="${SO101_SUCCESS_BALL_SAM2_URL:-http://127.0.0.1:${SAM2_PORT}/api/track_image}" \
  SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES="${SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES:-10}" \
  SO101_CAMERA_RELAY_DEVICES="${SO101_CAMERA_RELAY_DEVICES:-0 1 2}" \
  SO101_CAMERA_SPECS="${SO101_CAMERA_SPECS:-front=http://127.0.0.1:${CAMERA_RELAY_PORT}/2 side=http://127.0.0.1:${CAMERA_RELAY_PORT}/1 wrist=http://127.0.0.1:${CAMERA_RELAY_PORT}/0}" \
  SO101_OPEN_BROWSER="${SO101_OPEN_BROWSER:-1}" \
  SO101_AUTO_CONNECT="${SO101_AUTO_CONNECT:-1}" \
  "$REPO_ROOT/scripts/launch_so101_eval_ui.sh"
}

require_command screen
require_command ssh
require_command curl

stop_existing_local_motion
if [ "$GPU_SERVICES" != "0" ]; then
  start_remote_gpu_services
fi
if [ "$TUNNELS" != "0" ]; then
  start_tunnels
fi
if [ "$LOCAL_UI" != "0" ]; then
  start_local_ui
fi

echo
echo "SO101 eval stack is up:"
echo "  UI:       ${UI_URL}"
echo "  Cameras:  http://127.0.0.1:${CAMERA_RELAY_PORT}/health"
echo "  Policy:   http://127.0.0.1:${POLICY_PORT}/health"
echo "  SAM3:     http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}"
echo "  SAM2:     http://127.0.0.1:${SAM2_PORT}/health"
