#!/usr/bin/env bash
set -euo pipefail

# Preflight health check for the SO101 eval stack.
#
# Default checks local services only:
#   - camera relay :8089
#   - MolmoAct2 tunnel/server :8202
#   - SAM3 tunnel/server :8213
#   - SAM2 tunnel/server :8214
#
# Optional:
#   SO101_CHECK_UI=1      also require eval UI :8092
#   SO101_CHECK_REMOTE=1  also SSH to the GPU and check remote-local services

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

POLICY_PORT="${SO101_POLICY_PORT:-8202}"
SAM3_PORT="${SO101_SAM3_PORT:-8213}"
SAM2_PORT="${SO101_SAM2_PORT:-8214}"
CAMERA_RELAY_PORT="${SO101_CAMERA_RELAY_PORT:-8089}"
UI_PORT="${SO101_WEB_PORT:-8092}"
SAM3_READY_PATH="${SO101_SAM3_READY_PATH:-/}"

CHECK_UI="${SO101_CHECK_UI:-0}"
CHECK_REMOTE="${SO101_CHECK_REMOTE:-0}"
TIMEOUT_S="${SO101_CHECK_TIMEOUT_S:-5}"

failures=0

check_http() {
  local label="$1"
  local url="$2"
  if curl -fsS --max-time "$TIMEOUT_S" "$url" >/dev/null 2>&1; then
    printf "ok   %-14s %s\n" "$label" "$url"
  else
    printf "fail %-14s %s\n" "$label" "$url" >&2
    failures=$((failures + 1))
  fi
}

check_local_json_hint() {
  local label="$1"
  local url="$2"
  if curl -fsS --max-time "$TIMEOUT_S" "$url" >/tmp/so101-check-response.json 2>/dev/null; then
    printf "ok   %-14s %s\n" "$label" "$url"
  else
    printf "fail %-14s %s\n" "$label" "$url" >&2
    failures=$((failures + 1))
  fi
}

check_remote() {
  local ssh_opts=(
    -o StrictHostKeyChecking=no
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -p "$GPU_PORT"
  )
  if [ -n "$GPU_SSH_KEY" ]; then
    ssh_opts=(-i "$GPU_SSH_KEY" "${ssh_opts[@]}")
  fi
  if ssh "${ssh_opts[@]}" "${GPU_USER}@${GPU_HOST}" bash -s -- \
    "$POLICY_PORT" "$SAM3_PORT" "$SAM2_PORT" "$SAM3_READY_PATH" "$TIMEOUT_S" <<'REMOTE'
set -euo pipefail
POLICY_PORT="$1"
SAM3_PORT="$2"
SAM2_PORT="$3"
SAM3_READY_PATH="$4"
TIMEOUT_S="$5"
curl -fsS --max-time "$TIMEOUT_S" "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null
curl -fsS --max-time "$TIMEOUT_S" "http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}" >/dev/null
curl -fsS --max-time "$TIMEOUT_S" "http://127.0.0.1:${SAM2_PORT}/health" >/dev/null
REMOTE
  then
    printf "ok   %-14s %s@%s:%s\n" "remote-gpu" "$GPU_USER" "$GPU_HOST" "$GPU_PORT"
  else
    printf "fail %-14s %s@%s:%s\n" "remote-gpu" "$GPU_USER" "$GPU_HOST" "$GPU_PORT" >&2
    failures=$((failures + 1))
  fi
}

check_http "camera-relay" "http://127.0.0.1:${CAMERA_RELAY_PORT}/health"
check_local_json_hint "policy" "http://127.0.0.1:${POLICY_PORT}/health"
check_http "sam3" "http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}"
check_local_json_hint "sam2" "http://127.0.0.1:${SAM2_PORT}/health"

if [ "$CHECK_UI" != "0" ]; then
  check_local_json_hint "eval-ui" "http://127.0.0.1:${UI_PORT}/api/status?log_limit=1"
fi
if [ "$CHECK_REMOTE" != "0" ]; then
  check_remote
fi

if [ "$failures" -ne 0 ]; then
  echo "SO101 eval stack preflight failed: ${failures} check(s) failed." >&2
  exit 1
fi

echo "SO101 eval stack preflight passed."
