#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export GIT_LFS_SKIP_SMUDGE=1

PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"

apt-get update
apt-get install -y git git-lfs ffmpeg libgl1 libglib2.0-0 python3-venv
git lfs install

if [ ! -x "${PYTHON_BIN}" ]; then
  python3 -m venv /workspace/venv
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install hf_transfer

cd /workspace
if [ ! -d molmoact2 ]; then
  git clone https://github.com/allenai/molmoact2.git
fi

cd /workspace/molmoact2/experiments
"${PYTHON_BIN}" -m pip install -e ".[train]" debugpy
"${PYTHON_BIN}" -m pip install -e ./lerobot
"${PYTHON_BIN}" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); import olmo, omegaconf, debugpy, lerobot; print('experiments imports ok')"
