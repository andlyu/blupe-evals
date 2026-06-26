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
MODEL_DTYPE="${MOLMOACT2_MODEL_DTYPE:-float32}"
FLOW_STEPS="${MOLMOACT2_NUM_FLOW_TIMESTEPS:-10}"
ENABLE_CUDA_GRAPH="${MOLMOACT2_ENABLE_CUDA_GRAPH:-1}"
NORM_TAG="${MOLMOACT2_NORM_TAG:-}"

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

if [ "$ENABLE_CUDA_GRAPH" = "0" ]; then
  cmd+=(--no-enable-cuda-graph)
else
  cmd+=(--enable-cuda-graph)
fi

if [ -n "${MOLMOACT2_POLICY_PATH:-}" ]; then
  cmd+=(--policy-path "$MOLMOACT2_POLICY_PATH")
elif [ -n "${MOLMOACT2_CHECKPOINT_PATH:-}" ]; then
  cmd+=(--checkpoint-path "$MOLMOACT2_CHECKPOINT_PATH")
  if [ -z "$NORM_TAG" ] && [ "$MOLMOACT2_CHECKPOINT_PATH" = "allenai/MolmoAct2-SO100_101" ]; then
    NORM_TAG="so100_so101_molmoact2"
  fi
  if [ -n "$NORM_TAG" ]; then
    cmd+=(--norm-tag "$NORM_TAG")
  fi
else
  echo "Set MOLMOACT2_POLICY_PATH for a LeRobot-saved policy, or MOLMOACT2_CHECKPOINT_PATH for a base HF checkpoint." >&2
  exit 2
fi

exec "${cmd[@]}"
