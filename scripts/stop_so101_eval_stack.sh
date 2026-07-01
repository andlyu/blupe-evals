#!/usr/bin/env bash
set -euo pipefail

# Stop the SO101 eval stack started by start_so101_eval_stack.sh.
#
# Defaults:
#   - stop local policy/eval motion through the UI API
#   - stop local UI, camera relay, and GPU tunnels
#   - stop remote MolmoAct2/SAM3/tracker processes
#   - do not stop/destroy the Vast instance itself

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

POLICY_PORT="${SO101_POLICY_PORT:-8202}"
SAM3_PORT="${SO101_SAM3_PORT:-8213}"
SAM2_PORT="${SO101_SAM2_PORT:-8214}"
UI_PORT="${SO101_WEB_PORT:-8092}"
CAMERA_RELAY_PORT="${SO101_CAMERA_RELAY_PORT:-8089}"

UI_SCREEN="${SO101_UI_SCREEN:-so101_eval_8092}"
RELAY_SCREEN="${SO101_RELAY_SCREEN:-so101_camera_relay_8089}"
TUNNEL_SCREEN="${SO101_GPU_TUNNEL_SCREEN:-so101_gpu_tunnels_8202_8213_8214}"

STOP_LOCAL="${SO101_STOP_LOCAL:-1}"
STOP_REMOTE_SERVICES="${SO101_STOP_REMOTE_SERVICES:-1}"
STOP_VAST_INSTANCE="${SO101_STOP_VAST_INSTANCE:-0}"
VAST_INSTANCE_ID="${SO101_VAST_INSTANCE_ID:-}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -p "$GPU_PORT"
)
if [ -n "$GPU_SSH_KEY" ]; then
  SSH_OPTS=(-i "$GPU_SSH_KEY" "${SSH_OPTS[@]}")
fi

post_json_best_effort() {
  local url="$1"
  curl -fsS --max-time 2 \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$url" >/dev/null 2>&1 || true
}

stop_screen() {
  local name="$1"
  screen -S "$name" -X quit >/dev/null 2>&1 || true
}

stop_port_listener() {
  local port="$1"
  local label="$2"
  local pids
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return 0
  fi
  echo "Stopping ${label} listener on :${port} (pid(s): ${pids})"
  kill $pids 2>/dev/null || true
  for _ in {1..20}; do
    if [ -z "$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]; then
      return 0
    fi
    sleep 0.1
  done
  kill -KILL $pids 2>/dev/null || true
}

stop_local_stack() {
  echo "Stopping local SO101 UI motion, recording, UI, camera relay, and tunnels"
  post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/record/stop"
  post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/eval/stop"
  post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/stop"

  stop_screen "$UI_SCREEN"
  stop_screen "$RELAY_SCREEN"
  stop_screen "$TUNNEL_SCREEN"

  stop_port_listener "$UI_PORT" "SO101 eval UI"
  stop_port_listener "$CAMERA_RELAY_PORT" "SO101 camera relay"
  stop_port_listener "$POLICY_PORT" "MolmoAct2 tunnel"
  stop_port_listener "$SAM3_PORT" "SAM3 tunnel"
  stop_port_listener "$SAM2_PORT" "SAM2 tunnel"
}

stop_remote_services() {
  echo "Stopping remote GPU services on ${GPU_USER}@${GPU_HOST}:${GPU_PORT}"
  ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" bash -s -- "$REMOTE_ROOT" <<'REMOTE' || true
set -euo pipefail
REMOTE_ROOT="$1"
cd "${REMOTE_ROOT}/blupe-evals" 2>/dev/null || true
pkill -f 'scripts/molmoact2_policy_runner.py' >/dev/null 2>&1 || true
pkill -f 'scripts/sam3_prompt_ui.py' >/dev/null 2>&1 || true
pkill -f 'scripts/sam3_video_track_ui.py' >/dev/null 2>&1 || true
pkill -f 'scripts/sam2_video_track_ui.py' >/dev/null 2>&1 || true
pkill -f 'scripts/sam2_track_ui.py' >/dev/null 2>&1 || true
pkill -f 'scripts/yolo_ball_track_ui.py' >/dev/null 2>&1 || true
REMOTE
}

stop_vast_instance() {
  if [ "$STOP_VAST_INSTANCE" = "0" ]; then
    return 0
  fi
  if [ -z "$VAST_INSTANCE_ID" ]; then
    echo "SO101_STOP_VAST_INSTANCE=1 requires SO101_VAST_INSTANCE_ID" >&2
    return 1
  fi
  echo "Stopping Vast instance ${VAST_INSTANCE_ID}"
  vastai stop instance "$VAST_INSTANCE_ID"
}

if [ "$STOP_LOCAL" != "0" ]; then
  stop_local_stack
fi
if [ "$STOP_REMOTE_SERVICES" != "0" ]; then
  stop_remote_services
fi
stop_vast_instance

echo "SO101 eval stack stopped."
