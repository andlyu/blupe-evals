#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the Python side of a MolmoAct2 inference Vast instance.
#
# Run on the remote inference box. This is intentionally separate from the
# deploy script so it can run in parallel with model/checkpoint downloads.
#
# Example:
#   /workspace/blupe-evals/scripts/bootstrap_molmoact2_inference_env.sh

PYTHON="${PYTHON:-/venv/main/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found/executable: $PYTHON" >&2
  exit 2
fi

echo "[bootstrap] python=$("$PYTHON" --version)"

"$PYTHON" -m pip install -q \
  transformers==5.12.1 \
  peft==0.18.1 \
  huggingface_hub \
  opencv-python-headless \
  cached_path \
  omegaconf \
  rich \
  beartype \
  einops \
  einx \
  datasets \
  decord \
  av \
  imageio \
  imageio-ffmpeg \
  moviepy \
  torchmetrics \
  wandb \
  scipy \
  openai \
  editdistance

echo "[bootstrap] installed MolmoAct2 inference deps"

"$PYTHON" - <<'PY'
import importlib

mods = [
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "peft",
    "cv2",
    "cached_path",
    "omegaconf",
    "datasets",
    "decord",
    "av",
    "imageio",
    "moviepy",
    "torchmetrics",
    "wandb",
    "scipy",
    "openai",
    "editdistance",
]

for name in mods:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "ok")
    print(f"[bootstrap] {name} {version}")
PY
