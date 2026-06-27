#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-molmoact2-so101-pickup-mix-blueball-v21-plus-general500-2val-segments-s10k-eval50}"
SAVE_FOLDER="${SAVE_FOLDER:-/backup/outputs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-/backup/logs}"
EXPERIMENTS_DIR="${MOLMOACT2_EXPERIMENTS_DIR:-/workspace/molmoact2/experiments}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/venv/bin/torchrun}"
SCRIPT_PATH="${SCRIPT_PATH:-/workspace/blupe-evals/scripts/run_molmoact2_experiments_lora.py}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "${SAVE_FOLDER}" "${LOG_DIR}" "${HF_HOME}"
if [[ ! -f "${HF_HOME}/token" && -f /root/.cache/huggingface/token ]]; then
  cp /root/.cache/huggingface/token "${HF_HOME}/token"
  chmod 600 "${HF_HOME}/token"
fi

cleanup_full_checkpoints() {
  "${PYTHON_BIN}" - "${SAVE_FOLDER}" <<'PY'
import re
import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
if not run_dir.exists():
    raise SystemExit(0)

step_dirs = []
for path in run_dir.iterdir():
    if not path.is_dir():
        continue
    match = re.fullmatch(r"step(\d+)", path.name)
    if match:
        step_dirs.append((int(match.group(1)), path))

if len(step_dirs) <= 1:
    raise SystemExit(0)

step_dirs.sort()
for _, path in step_dirs[:-1]:
    heavy = path / "model_and_optim"
    if heavy.exists():
        shutil.rmtree(heavy)
PY
}

cleanup_loop() {
  while true; do
    cleanup_full_checkpoints || true
    sleep "${CHECKPOINT_CLEANUP_INTERVAL_S:-300}"
  done
}

common_args=(
  --mixture pickup_mix_blueball_plus_general500
  --dataset-spec 'andlyu/so100_so101_original_500eps_camera12@0-158|so100_so101_original_500eps|observation.images.camera1,observation.images.camera2|0.5|SO100/SO101 original manipulation task'
  --dataset-spec 'andlyu/pick_up_ball_v21@0-6|so101_pick_up_ball_v21|observation.images.front,observation.images.wrist|0.1666666667|single SO-101 follower arm picking up a ball'
  --dataset-spec 'andlyu/pick_up_ball_v21_pt2@0-22|so101_pick_up_ball_v21|observation.images.front,observation.images.wrist|0.1666666667|single SO-101 follower arm picking up a ball'
  --dataset-spec 'andlyu/move_blue_ball_training_v21@0-5|so101_move_blue_ball_v21|observation.images.front,observation.images.wrist|0.1666666667|single SO-101 follower arm moving a blue ball'
  --validation-dataset-spec 'andlyu/so100_so101_original_500eps_camera12@159-167|so100_so101_original_500eps|observation.images.camera1,observation.images.camera2|1.0|SO100/SO101 original manipulation task'
  --validation-dataset-spec 'andlyu/move_blue_ball_training_v21@6-7|so101_move_blue_ball_v21|observation.images.front,observation.images.wrist|1.0|single SO-101 follower arm moving a blue ball'
  --custom-tag so101_pick_up_ball_v21
  --custom-tag so101_move_blue_ball_v21
  --run-name "${RUN_NAME}"
  --save-folder "${SAVE_FOLDER}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
  --device-batch-size "${DEVICE_BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-2}"
  --eval-interval "${EVAL_INTERVAL:-50}"
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-64}"
  --eval-device-batch-size "${EVAL_DEVICE_BATCH_SIZE:-1}"
  --save-keep "${SAVE_KEEP:-20}"
  --save-merged-lora
)

run_phase() {
  local max_duration="$1"
  local save_interval="$2"
  local phase_name="$3"
  local dry_run_args=()
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    dry_run_args=(--dry-run)
  fi

  echo "=== ${phase_name}: max_duration=${max_duration} save_interval=${save_interval} ==="
  cd "${EXPERIMENTS_DIR}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    "${PYTHON_BIN}" \
      "${SCRIPT_PATH}" \
      "${common_args[@]}" \
      --max-duration "${max_duration}" \
      --save-interval "${save_interval}" \
      "${dry_run_args[@]}" \
      2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}.log"
  else
    "${TORCHRUN_BIN}" --standalone --nproc-per-node=1 \
      "${SCRIPT_PATH}" \
      "${common_args[@]}" \
      --max-duration "${max_duration}" \
      --save-interval "${save_interval}" \
      "${dry_run_args[@]}" \
      2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}.log"
  fi
  cleanup_full_checkpoints || true
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  run_phase 250 250 dry_run
  exit 0
fi

cleanup_loop &
cleanup_pid="$!"
trap 'kill "${cleanup_pid}" 2>/dev/null || true' EXIT

run_phase 250 250 phase_step250
run_phase 500 500 phase_step500
run_phase 10000 1000 phase_step10000

cleanup_full_checkpoints || true
echo "Training complete: ${SAVE_FOLDER}"
