#!/usr/bin/env bash
set -euo pipefail

# Fast local launcher for http://localhost:8092/#setup.
# It starts the camera relay and eval UI only; it does not wait for A100 policy
# model load.

REPO_ROOT="${BLUPE_EVALS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SO101_PYTHON:-python}"

UI_HOST="${SO101_WEB_HOST:-127.0.0.1}"
UI_PORT="${SO101_WEB_PORT:-8092}"
UI_URL="${SO101_UI_URL:-http://localhost:${UI_PORT}/#setup}"
UI_READY_TIMEOUT_S="${SO101_UI_READY_TIMEOUT_S:-5}"
CONNECT_TIMEOUT_S="${SO101_CONNECT_TIMEOUT_S:-3}"

POLICY_URL="${SO101_POLICY_URL:-http://127.0.0.1:8202}"
POLICY_CAMERAS="${SO101_POLICY_CAMERAS:-front,wrist}"
ROBOT_PORT="${SO101_ROBOT_PORT:-/dev/tty.usbmodem58FD0169761}"
ROBOT_ID="${SO101_ROBOT_ID:-blupe_follower}"
LEADER_PORT="${SO101_LEADER_PORT:-/dev/tty.usbmodem58FA1025151}"
LEADER_ID="${SO101_LEADER_ID:-blupe_leader}"
SUCCESS_TRACKING="${SO101_SUCCESS_TRACKING:-1}"
SAM2_TRACKING_AUTO="${SO101_SUCCESS_BALL_SAM2_AUTO:-0}"
SAM2_TRACKING_HEALTH_URL="${SO101_SUCCESS_BALL_SAM2_HEALTH_URL:-http://127.0.0.1:8214/health}"
SAM2_TRACKING_URL="${SO101_SUCCESS_BALL_SAM2_URL:-}"
OPEN_BROWSER="${SO101_OPEN_BROWSER:-1}"
AUTO_CONNECT="${SO101_AUTO_CONNECT:-1}"
REPLACE_PORTS="${SO101_REPLACE_PORTS:-1}"

CAMERA_RELAY_ENABLED="${SO101_CAMERA_RELAY_ENABLED:-1}"
CAMERA_RELAY_HOST="${SO101_CAMERA_RELAY_HOST:-127.0.0.1}"
CAMERA_RELAY_PORT="${SO101_CAMERA_RELAY_PORT:-8089}"
CAMERA_RELAY_DEVICES="${SO101_CAMERA_RELAY_DEVICES:-0 1 2}"
CAMERA_RELAY_WIDTH="${SO101_CAMERA_RELAY_WIDTH:-640}"
CAMERA_RELAY_HEIGHT="${SO101_CAMERA_RELAY_HEIGHT:-360}"
CAMERA_RELAY_FPS="${SO101_CAMERA_RELAY_FPS:-30}"
CAMERA_READY_TIMEOUT_S="${SO101_CAMERA_READY_TIMEOUT_S:-5}"
CAMERA_SPECS="${SO101_CAMERA_SPECS:-front=http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/2 side=http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/1 wrist=http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/0}"

RELAY_SCREEN="${SO101_RELAY_SCREEN:-so101_camera_relay_8089}"
UI_SCREEN="${SO101_UI_SCREEN:-so101_eval_8092}"
LOG_DIR="${SO101_LOG_DIR:-/tmp}"
RELAY_LOG="${SO101_RELAY_LOG:-${LOG_DIR}/${RELAY_SCREEN}.log}"
UI_LOG="${SO101_UI_LOG:-${LOG_DIR}/${UI_SCREEN}.log}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

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
  for _ in {1..15}; do
    if [ -z "$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]; then
      return 0
    fi
    sleep 0.1
  done
  kill -KILL $pids 2>/dev/null || true
  for _ in {1..15}; do
    if [ -z "$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]; then
      return 0
    fi
    sleep 0.1
  done
  echo "Could not stop ${label} listener on :${port}" >&2
  return 1
}

start_screen() {
  local name="$1"
  local log="$2"
  local screenrc="${LOG_DIR}/${name}.screenrc"
  shift 2
  printf 'logfile %s\nlogfile flush 1\n' "$log" >"$screenrc"
  screen -c "$screenrc" -L -dmS "$name" "$@"
}

wait_http() {
  local url="$1"
  local timeout_s="$2"
  local label="$3"
  local attempts=$((timeout_s * 10))
  local i
  if [ "$attempts" -lt 1 ]; then
    attempts=1
  fi
  for ((i = 0; i < attempts; i++)); do
    if curl -fsS --max-time 0.5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  echo "${label} did not become ready within ${timeout_s}s: ${url}" >&2
  return 1
}

post_json_best_effort() {
  local url="$1"
  local timeout_s="$2"
  curl -fsS --max-time "$timeout_s" \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$url" >/dev/null
}

require_command screen
require_command curl
require_command "$PYTHON"
PYTHON="$(command -v "$PYTHON")"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

stop_screen "$UI_SCREEN"
stop_port_listener "$UI_PORT" "SO101 eval UI"
if [ "$CAMERA_RELAY_ENABLED" != "0" ]; then
  stop_screen "$RELAY_SCREEN"
  stop_port_listener "$CAMERA_RELAY_PORT" "SO101 camera relay"
  read -r -a relay_devices <<< "$CAMERA_RELAY_DEVICES"
  start_screen "$RELAY_SCREEN" "$RELAY_LOG" \
    "$PYTHON" "$REPO_ROOT/YAM_control/camera_relay.py" \
    --devices "${relay_devices[@]}" \
    --port "$CAMERA_RELAY_PORT" \
    --width "$CAMERA_RELAY_WIDTH" \
    --height "$CAMERA_RELAY_HEIGHT" \
    --fps "$CAMERA_RELAY_FPS"

  if ! wait_http "http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/health" "$CAMERA_READY_TIMEOUT_S" "camera relay"; then
    echo "Continuing anyway; camera readiness will be visible in the UI." >&2
  fi
fi

ui_cmd=(
  "$PYTHON" "$REPO_ROOT/scripts/so101_web_intervene.py"
  --host "$UI_HOST"
  --port "$UI_PORT"
  --robot-port "$ROBOT_PORT"
  --robot-id "$ROBOT_ID"
  --leader-port "$LEADER_PORT"
  --leader-id "$LEADER_ID"
  --policy-url "$POLICY_URL"
)

IFS=',' read -r -a policy_camera_names <<< "$POLICY_CAMERAS"
for camera_name in "${policy_camera_names[@]}"; do
  camera_name="${camera_name//[[:space:]]/}"
  if [ -n "$camera_name" ]; then
    ui_cmd+=(--policy-camera "$camera_name")
  fi
done

read -r -a camera_specs <<< "$CAMERA_SPECS"
for camera_spec in "${camera_specs[@]}"; do
  camera_spec="${camera_spec//[[:space:]]/}"
  if [ -n "$camera_spec" ]; then
    ui_cmd+=(--camera "$camera_spec")
  fi
done

if [ "$SUCCESS_TRACKING" = "0" ]; then
  ui_cmd+=(--no-success-tracking)
fi

if [ "$SUCCESS_TRACKING" != "0" ]; then
  if [ "$SAM2_TRACKING_AUTO" != "0" ] && [ -z "$SAM2_TRACKING_URL" ] && curl -fsS --max-time 0.5 "$SAM2_TRACKING_HEALTH_URL" >/dev/null 2>&1; then
    SAM2_TRACKING_URL="${SAM2_TRACKING_HEALTH_URL%/health}/api/track_image"
  fi
  if [ -n "$SAM2_TRACKING_URL" ]; then
    export SO101_SUCCESS_BALL_SAM2_URL="$SAM2_TRACKING_URL"
    echo "SAM2 ball tracking: ${SAM2_TRACKING_URL}"
  else
    echo "SAM2 ball tracking: disabled; using SAM3 ball masks" >&2
  fi
fi

start_screen "$UI_SCREEN" "$UI_LOG" "${ui_cmd[@]}"
wait_http "http://127.0.0.1:${UI_PORT}/api/status?log_limit=1" "$UI_READY_TIMEOUT_S" "SO101 eval UI"

if [ "$AUTO_CONNECT" != "0" ]; then
  if ! post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/connect" "$CONNECT_TIMEOUT_S"; then
    echo "Robot auto-connect did not complete within ${CONNECT_TIMEOUT_S}s; use Connect/Read in the UI." >&2
  fi
fi

echo "SO101 eval UI: ${UI_URL}"
echo "UI screen: ${UI_SCREEN} (${UI_LOG})"
if [ "$CAMERA_RELAY_ENABLED" != "0" ]; then
  echo "Camera screen: ${RELAY_SCREEN} (${RELAY_LOG})"
fi

if [ "$OPEN_BROWSER" != "0" ] && command -v open >/dev/null 2>&1; then
  open "$UI_URL" >/dev/null 2>&1 || true
fi
