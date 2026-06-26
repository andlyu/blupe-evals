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
SUCCESS_TRACKING="${SO101_SUCCESS_TRACKING:-0}"

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

if [ "$SUCCESS_TRACKING" = "0" ]; then
  cmd+=(--no-success-tracking)
fi

exec "${cmd[@]}"
