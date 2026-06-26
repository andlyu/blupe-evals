#!/usr/bin/env bash
set -euo pipefail

# Start the MolmoAct2 HTTP policy server on the A100 box.
# Run this inside the Python environment that can import the MolmoAct2-enabled
# LeRobot checkout used for training/inference.

REPO_ROOT="${BLUPE_EVALS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HOST="${MOLMOACT2_HOST:-0.0.0.0}"
PORT="${MOLMOACT2_PORT:-8202}"
IMAGE_KEYS="${MOLMOACT2_IMAGE_KEYS:-[\"observation.images.front\",\"observation.images.wrist\"]}"
STATE_KEY="${MOLMOACT2_STATE_KEY:-observation.state}"
DEVICE="${MOLMOACT2_DEVICE:-cuda}"
NUM_ACTIONS="${MOLMOACT2_NUM_ACTIONS:-30}"
ACTION_DIM="${MOLMOACT2_ACTION_DIM:-6}"
MODEL_DTYPE="${MOLMOACT2_MODEL_DTYPE:-bfloat16}"
FLOW_STEPS="${MOLMOACT2_NUM_FLOW_TIMESTEPS:-8}"

if [ -n "${MOLMOACT2_EXPERIMENTS_DIR:-}" ]; then
  export PYTHONPATH="${MOLMOACT2_EXPERIMENTS_DIR}:${MOLMOACT2_EXPERIMENTS_DIR}/lerobot/src:${PYTHONPATH:-}"
fi

cmd=(
  python "$REPO_ROOT/scripts/molmoact2_policy_runner.py"
  --host "$HOST"
  --port "$PORT"
  --image-keys "$IMAGE_KEYS"
  --state-key "$STATE_KEY"
  --device "$DEVICE"
  --num-actions "$NUM_ACTIONS"
  --action-dim "$ACTION_DIM"
  --model-dtype "$MODEL_DTYPE"
  --num-flow-timesteps "$FLOW_STEPS"
)

if [ -n "${MOLMOACT2_POLICY_PATH:-}" ]; then
  cmd+=(--policy-path "$MOLMOACT2_POLICY_PATH")
elif [ -n "${MOLMOACT2_CHECKPOINT_PATH:-}" ]; then
  cmd+=(--checkpoint-path "$MOLMOACT2_CHECKPOINT_PATH")
  if [ -n "${MOLMOACT2_NORM_TAG:-}" ]; then
    cmd+=(--norm-tag "$MOLMOACT2_NORM_TAG")
  fi
else
  echo "Set MOLMOACT2_POLICY_PATH for a LeRobot-saved policy, or MOLMOACT2_CHECKPOINT_PATH for a base HF checkpoint." >&2
  exit 2
fi

exec "${cmd[@]}"
