#!/usr/bin/env bash
set -euo pipefail

# Deploy a MolmoAct2 inference server to a running Vast instance.
#
# Intended use:
#   Run this from the training/A100 host, where /workspace/blupe-evals and
#   /workspace/molmoact2 exist. The target Vast instance should already be
#   launched with a Python 3.12 + CUDA image, e.g.:
#     vastai/pytorch:cuda-13.0.3-auto
#
# Required env:
#   REMOTE_HOST=216.234.102.170
#   REMOTE_PORT=11422
#
# Recommended env:
#   CHECKPOINT_LOCAL=/backup/outputs/<run>/<step>-merged
#   or
#   CHECKPOINT_HF_REPO=alyudre/<uploaded-merged-checkpoint-repo>
#
# Optional env:
#   REMOTE_USER=root
#   RUN_NAME=molmoact2-so101-intervene-plus-general169-val2x4-s10k-base-norm-earlyval10
#   STEP=step250
#   NORM_TAG=so100_so101_molmoact2
#   POLICY_PORT=8202
#   REMOTE_ROOT=/workspace
#   PYTHON=/venv/main/bin/python
#   SSH_KEY=/root/.ssh/vast_4090_transfer_ed25519

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST}"
REMOTE_PORT="${REMOTE_PORT:?set REMOTE_PORT}"
REMOTE_USER="${REMOTE_USER:-root}"
RUN_NAME="${RUN_NAME:-molmoact2-so101-intervene-plus-general169-val2x4-s10k-base-norm-earlyval10}"
STEP="${STEP:-step250}"
NORM_TAG="${NORM_TAG:-so100_so101_molmoact2}"
POLICY_PORT="${POLICY_PORT:-8202}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
SSH_KEY="${SSH_KEY:-}"

LOCAL_BLUPE="${LOCAL_BLUPE:-/workspace/blupe-evals}"
LOCAL_MOLMOACT2="${LOCAL_MOLMOACT2:-/workspace/molmoact2}"
CHECKPOINT_LOCAL="${CHECKPOINT_LOCAL:-/backup/outputs/${RUN_NAME}/${STEP}-merged}"
CHECKPOINT_HF_REPO="${CHECKPOINT_HF_REPO:-}"
REMOTE_CHECKPOINT="${REMOTE_CHECKPOINT:-${REMOTE_ROOT}/checkpoints/${RUN_NAME}/${STEP}-merged}"

SSH_OPTS=(-o StrictHostKeyChecking=no -p "$REMOTE_PORT")
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS=(-i "$SSH_KEY" "${SSH_OPTS[@]}")
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
RSH=(ssh "${SSH_OPTS[@]}")

echo "[deploy] target=${REMOTE} port=${REMOTE_PORT}"
echo "[deploy] remote checkpoint=${REMOTE_CHECKPOINT}"

"${RSH[@]}" "$REMOTE" "mkdir -p '${REMOTE_ROOT}' '${REMOTE_ROOT}/checkpoints/${RUN_NAME}'"

echo "[deploy] syncing source trees"
rsync -a --delete -e "$(printf '%q ' "${RSH[@]}")" "${LOCAL_BLUPE}/" "${REMOTE}:${REMOTE_ROOT}/blupe-evals/"
rsync -a --delete -e "$(printf '%q ' "${RSH[@]}")" "${LOCAL_MOLMOACT2}/" "${REMOTE}:${REMOTE_ROOT}/molmoact2/"

echo "[deploy] ensuring runtime deps"
"${RSH[@]}" "$REMOTE" "PYTHON='${PYTHON}' '${REMOTE_ROOT}/blupe-evals/scripts/bootstrap_molmoact2_inference_env.sh'"

if [[ -n "$CHECKPOINT_HF_REPO" ]]; then
  echo "[deploy] downloading merged checkpoint from HF repo ${CHECKPOINT_HF_REPO}"
  "${RSH[@]}" "$REMOTE" "mkdir -p '${REMOTE_CHECKPOINT}' && HF_HOME='${REMOTE_ROOT}/.hf_home' ${PYTHON} - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('${CHECKPOINT_HF_REPO}', local_dir='${REMOTE_CHECKPOINT}')
print('${REMOTE_CHECKPOINT}')
PY"
elif [[ -f "${CHECKPOINT_LOCAL}/model.pt" && -f "${CHECKPOINT_LOCAL}/config.yaml" ]]; then
  echo "[deploy] rsyncing local merged checkpoint ${CHECKPOINT_LOCAL}"
  rsync -a --partial --info=progress2 -e "$(printf '%q ' "${RSH[@]}")" "${CHECKPOINT_LOCAL}/" "${REMOTE}:${REMOTE_CHECKPOINT}/"
else
  echo "[deploy] missing merged checkpoint: ${CHECKPOINT_LOCAL}" >&2
  echo "[deploy] set CHECKPOINT_HF_REPO or CHECKPOINT_LOCAL to a directory containing model.pt and config.yaml" >&2
  exit 2
fi

echo "[deploy] starting policy server"
"${RSH[@]}" "$REMOTE" "cat > '${REMOTE_ROOT}/start_molmoact2_policy.sh' <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_ROOT}/blupe-evals'
export HF_HOME='${REMOTE_ROOT}/.hf_home'
export PYTHONPATH='${REMOTE_ROOT}/molmoact2/experiments/lerobot/src:${REMOTE_ROOT}/molmoact2/experiments'
exec '${PYTHON}' scripts/molmoact2_policy_runner.py \
  --host 0.0.0.0 \
  --port '${POLICY_PORT}' \
  --checkpoint-path '${REMOTE_CHECKPOINT}' \
  --norm-tag '${NORM_TAG}'
SH
chmod +x '${REMOTE_ROOT}/start_molmoact2_policy.sh'
pkill -f 'molmoact2_policy_runner.py' || true
nohup '${REMOTE_ROOT}/start_molmoact2_policy.sh' > '${REMOTE_ROOT}/molmoact2_policy.log' 2>&1 &
echo \$! > '${REMOTE_ROOT}/molmoact2_policy.pid'"

echo "[deploy] started. Logs: ${REMOTE}:${REMOTE_ROOT}/molmoact2_policy.log"
echo "[deploy] health: curl http://${REMOTE_HOST}:${POLICY_PORT}/health if port is directly mapped, otherwise use the Vast host port mapping."
