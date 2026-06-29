#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-molmoact2-so101-intervene3-plus-general169-val4x4-s10k-base-norm-step1000-fresh}"
SAVE_FOLDER="${SAVE_FOLDER:-/backup/outputs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-/backup/logs}"
EXPERIMENTS_DIR="${MOLMOACT2_EXPERIMENTS_DIR:-/workspace/molmoact2/experiments}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/venv/bin/torchrun}"
SCRIPT_PATH="${SCRIPT_PATH:-/workspace/blupe-evals/scripts/run_molmoact2_experiments_lora.py}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

GENERAL_SAMPLE_WEIGHT="${GENERAL_SAMPLE_WEIGHT:-0.5}"
INTERVENE_SAMPLE_WEIGHT="${INTERVENE_SAMPLE_WEIGHT:-0.1666666667}"
INTERVENE_2_SAMPLE_WEIGHT="${INTERVENE_2_SAMPLE_WEIGHT:-0.1666666667}"
INTERVENE_3_SAMPLE_WEIGHT="${INTERVENE_3_SAMPLE_WEIGHT:-0.1666666667}"
LOAD_PATH="${LOAD_PATH:-}"
RESET_OPTIMIZER_STATE="${RESET_OPTIMIZER_STATE:-0}"
RESET_TRAINER_STATE="${RESET_TRAINER_STATE:-0}"

mkdir -p "${SAVE_FOLDER}" "${LOG_DIR}" "${HF_HOME}"
if [[ ! -f "${HF_HOME}/token" && -f /root/.cache/huggingface/token ]]; then
  cp /root/.cache/huggingface/token "${HF_HOME}/token"
  chmod 600 "${HF_HOME}/token"
fi

cleanup_full_checkpoints() {
  "${PYTHON_BIN}" - "${SAVE_FOLDER}" "${MERGED_SAVE_KEEP:-3}" <<'PY'
import re
import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
merged_keep = max(0, int(sys.argv[2]))
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

merged_dirs = []
for path in run_dir.iterdir():
    if not path.is_dir():
        continue
    match = re.fullmatch(r"step(\d+)(?:-lora)?-merged", path.name)
    if match:
        merged_dirs.append((int(match.group(1)), path))

if merged_keep and len(merged_dirs) > merged_keep:
    merged_dirs.sort()
    for _, path in merged_dirs[:-merged_keep]:
        shutil.rmtree(path)
PY
}

cleanup_loop() {
  while true; do
    cleanup_full_checkpoints || true
    sleep "${CHECKPOINT_CLEANUP_INTERVAL_S:-300}"
  done
}

common_args=(
  --mixture intervene3_v21_plus_general169
  --dataset-spec "andlyu/so100_so101_original_500eps_camera12@0-164|so100_so101_original_500eps|observation.images.camera1,observation.images.camera2|${GENERAL_SAMPLE_WEIGHT}|SO100/SO101 original manipulation task"
  --dataset-spec "andlyu/so101-ball-cup-intervene-edited_v21@0-12|so101_ball_cup_intervene_edited_v21|observation.images.front,observation.images.wrist|${INTERVENE_SAMPLE_WEIGHT}|single SO-101 follower arm moving a light blue ball to a tall black cylinder"
  --dataset-spec "andlyu/so101-ball-cup-intervene-edited_2_v21@0-12|so101_ball_cup_intervene_edited_2_v21|observation.images.front,observation.images.wrist|${INTERVENE_2_SAMPLE_WEIGHT}|single SO-101 follower arm moving a light blue ball to a tall black cylinder"
  --dataset-spec "andlyu/so101-ball-cup-intervene-edited_3_v21@0-14|so101_ball_cup_intervene_edited_3_v21|observation.images.front,observation.images.wrist|${INTERVENE_3_SAMPLE_WEIGHT}|single SO-101 follower arm moving a light blue ball to a tall black cylinder"
  --validation-dataset-spec 'andlyu/so100_so101_original_500eps_camera12@165-168|so100_so101_original_500eps|observation.images.camera1,observation.images.camera2|1.0|SO100/SO101 original manipulation task'
  --validation-dataset-spec 'andlyu/so101-ball-cup-intervene-edited_v21@13-16|so101_ball_cup_intervene_edited_v21|observation.images.front,observation.images.wrist|1.0|single SO-101 follower arm moving a light blue ball to a tall black cylinder'
  --validation-dataset-spec 'andlyu/so101-ball-cup-intervene-edited_2_v21@13-16|so101_ball_cup_intervene_edited_2_v21|observation.images.front,observation.images.wrist|1.0|single SO-101 follower arm moving a light blue ball to a tall black cylinder'
  --validation-dataset-spec 'andlyu/so101-ball-cup-intervene-edited_3_v21@15-18|so101_ball_cup_intervene_edited_3_v21|observation.images.front,observation.images.wrist|1.0|single SO-101 follower arm moving a light blue ball to a tall black cylinder'
  --norm-stats-tag "${NORM_STATS_TAG:-so100_so101_molmoact2}"
  --custom-tag so101_ball_cup_intervene_edited_v21
  --custom-tag so101_ball_cup_intervene_edited_2_v21
  --custom-tag so101_ball_cup_intervene_edited_3_v21
  --run-name "${RUN_NAME}"
  --save-folder "${SAVE_FOLDER}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
  --device-batch-size "${DEVICE_BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-2}"
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-64}"
  --eval-device-batch-size "${EVAL_DEVICE_BATCH_SIZE:-1}"
  --save-keep "${SAVE_KEEP:-20}"
)
if [[ -n "${LOAD_PATH}" ]]; then
  common_args+=(--load-path "${LOAD_PATH}")
  if [[ "${RESET_OPTIMIZER_STATE}" == "1" ]]; then
    common_args+=(--reset-optimizer-state)
  fi
  if [[ "${RESET_TRAINER_STATE}" == "1" ]]; then
    common_args+=(--reset-trainer-state)
  fi
fi

MERGED_CHECKPOINT_STEPS="${MERGED_CHECKPOINT_STEPS:-250,500,1000}"

should_save_merged_for_step() {
  local step="$1"
  [[ ",${MERGED_CHECKPOINT_STEPS}," == *",${step},"* ]]
}

run_phase() {
  local max_duration="$1"
  local save_interval="$2"
  local phase_name="$3"
  local eval_interval="$4"
  local merged_step="${5:-}"
  local dry_run_args=()
  local phase_args=("${common_args[@]}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    dry_run_args=(--dry-run)
  fi
  if [[ -n "${merged_step}" ]] && should_save_merged_for_step "${merged_step}"; then
    phase_args+=(--save-merged-lora)
  fi

  echo "=== ${phase_name}: max_duration=${max_duration} save_interval=${save_interval} eval_interval=${eval_interval} merged_step=${merged_step:-none} merged_steps=${MERGED_CHECKPOINT_STEPS} ==="
  cd "${EXPERIMENTS_DIR}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    "${PYTHON_BIN}" \
      "${SCRIPT_PATH}" \
      "${phase_args[@]}" \
      --eval-interval "${eval_interval}" \
      --max-duration "${max_duration}" \
      --save-interval "${save_interval}" \
      "${dry_run_args[@]}" \
      2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}.log"
  else
    "${TORCHRUN_BIN}" --standalone --nproc-per-node=1 \
      "${SCRIPT_PATH}" \
      "${phase_args[@]}" \
      --eval-interval "${eval_interval}" \
      --max-duration "${max_duration}" \
      --save-interval "${save_interval}" \
      "${dry_run_args[@]}" \
      2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}.log"
  fi
  cleanup_full_checkpoints || true
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  run_phase 50 50 dry_run "${EARLY_EVAL_INTERVAL:-10}"
  exit 0
fi

cleanup_loop &
cleanup_pid="$!"
trap 'kill "${cleanup_pid}" 2>/dev/null || true' EXIT

START_PHASE="${START_PHASE:-50}"
if (( START_PHASE <= 50 )); then
  run_phase 50 50 phase_step50 "${EARLY_EVAL_INTERVAL:-10}"
fi
if (( START_PHASE <= 250 )); then
  run_phase 250 250 phase_step250 "${EVAL_INTERVAL:-50}" 250
fi
if (( START_PHASE <= 500 )); then
  run_phase 500 500 phase_step500 "${EVAL_INTERVAL:-50}" 500
fi
if (( START_PHASE <= 1000 )); then
  run_phase 1000 1000 phase_step1000 "${EVAL_INTERVAL:-50}" 1000
fi
if (( START_PHASE <= 10000 )); then
  run_phase 10000 1000 phase_step10000 "${EVAL_INTERVAL:-50}"
fi

cleanup_full_checkpoints || true
echo "Training complete: ${SAVE_FOLDER}"
