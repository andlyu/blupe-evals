#!/usr/bin/env bash
set -euo pipefail

# Start the SO101 web UI on the Jetson while calling a remote A100 policy server.
# Mac still only opens the browser through SSH tunnels.

REPO_ROOT="${BLUPE_EVALS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HOST="${SO101_WEB_HOST:-127.0.0.1}"
PORT="${SO101_WEB_PORT:-8092}"
POLICY_URL="${SO101_POLICY_URL:-}"
POLICY_CAMERAS="${SO101_POLICY_CAMERAS:-front,wrist}"
ROBOT_PORT="${SO101_ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${SO101_ROBOT_ID:-blupe_follower}"
LEADER_PORT="${SO101_LEADER_PORT:-/dev/ttyACM1}"
LEADER_ID="${SO101_LEADER_ID:-blupe_leader}"
SUCCESS_TRACKING="${SO101_SUCCESS_TRACKING:-1}"
CAMERA_RELAY_ENABLED="${SO101_CAMERA_RELAY_ENABLED:-1}"
CAMERA_RELAY_PORT="${SO101_CAMERA_RELAY_PORT:-8089}"
CAMERA_RELAY_DEVICES="${SO101_CAMERA_RELAY_DEVICES:-0 1 2}"
CAMERA_RELAY_WIDTH="${SO101_CAMERA_RELAY_WIDTH:-640}"
CAMERA_RELAY_HEIGHT="${SO101_CAMERA_RELAY_HEIGHT:-360}"
CAMERA_RELAY_FPS="${SO101_CAMERA_RELAY_FPS:-30}"
CAMERA_SPECS="${SO101_CAMERA_SPECS:-front=http://127.0.0.1:${CAMERA_RELAY_PORT}/2 side=http://127.0.0.1:${CAMERA_RELAY_PORT}/1 wrist=http://127.0.0.1:${CAMERA_RELAY_PORT}/0}"

if [ -z "$POLICY_URL" ]; then
  echo "Set SO101_POLICY_URL to the A100 policy server, e.g. http://127.0.0.1:8202 through a tunnel." >&2
  exit 2
fi

cmd=(
  python "$REPO_ROOT/scripts/so101_web_intervene.py"
  --host "$HOST"
  --port "$PORT"
  --robot-port "$ROBOT_PORT"
  --robot-id "$ROBOT_ID"
  --leader-port "$LEADER_PORT"
  --leader-id "$LEADER_ID"
  --policy-url "$POLICY_URL"
)

IFS=',' read -r -a camera_names <<< "$POLICY_CAMERAS"
for camera_name in "${camera_names[@]}"; do
  camera_name="${camera_name//[[:space:]]/}"
  if [ -n "$camera_name" ]; then
    cmd+=(--policy-camera "$camera_name")
  fi
done

read -r -a camera_specs <<< "$CAMERA_SPECS"
for camera_spec in "${camera_specs[@]}"; do
  camera_spec="${camera_spec//[[:space:]]/}"
  if [ -n "$camera_spec" ]; then
    cmd+=(--camera "$camera_spec")
  fi
done

if [ "$SUCCESS_TRACKING" = "0" ]; then
  cmd+=(--no-success-tracking)
fi

relay_pid=""
cleanup() {
  if [ -n "$relay_pid" ]; then
    kill "$relay_pid" 2>/dev/null || true
    wait "$relay_pid" 2>/dev/null || true
  fi
}

if [ "$CAMERA_RELAY_ENABLED" != "0" ]; then
  read -r -a relay_devices <<< "$CAMERA_RELAY_DEVICES"
  python "$REPO_ROOT/YAM_control/camera_relay.py" \
    --devices "${relay_devices[@]}" \
    --port "$CAMERA_RELAY_PORT" \
    --width "$CAMERA_RELAY_WIDTH" \
    --height "$CAMERA_RELAY_HEIGHT" \
    --fps "$CAMERA_RELAY_FPS" &
  relay_pid=$!
  trap cleanup EXIT INT TERM

  camera_relay_ready=0
  for _ in {1..20}; do
    if ! kill -0 "$relay_pid" 2>/dev/null; then
      wait "$relay_pid"
      exit 1
    fi
    if curl -fsS --max-time 1 "http://127.0.0.1:${CAMERA_RELAY_PORT}/health" >/dev/null; then
      camera_relay_ready=1
      break
    fi
    sleep 0.5
  done
  if [ "$camera_relay_ready" != "1" ]; then
    echo "Camera relay did not become healthy on :${CAMERA_RELAY_PORT}" >&2
    exit 1
  fi
  "${cmd[@]}"
else
  exec "${cmd[@]}"
fi
