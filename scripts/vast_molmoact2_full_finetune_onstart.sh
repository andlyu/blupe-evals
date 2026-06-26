#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export GIT_LFS_SKIP_SMUDGE=1

apt-get update
apt-get install -y git git-lfs ffmpeg libgl1 libglib2.0-0 python3-venv
git lfs install

python3 -m venv /workspace/venv
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install hf_transfer

cd /workspace
if [ ! -d molmoact2 ]; then
  git clone https://github.com/allenai/molmoact2.git
fi
cd molmoact2
git submodule update --init lerobot
cd lerobot

"${PYTHON_BIN}" -m pip install -e ".[molmoact2]"
"${PYTHON_BIN}" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

mkdir -p /workspace/blupe_training
cat > /workspace/blupe_training/run_molmoact2_full_finetune.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-repo", default=os.environ.get("DATASET_REPO", "andlyu/move_blue_ball_training_v21"))
parser.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", "/workspace/lerobot_data/andlyu/move_blue_ball_training_v21"))
parser.add_argument("--dataset-revision", default=os.environ.get("DATASET_REVISION", "main"))
parser.add_argument("--checkpoint", default=os.environ.get("MOLMOACT2_CHECKPOINT", "allenai/MolmoAct2"))
parser.add_argument("--job-name", default=os.environ.get("RUN_NAME", "molmoact2-so101-move-blue-ball-success"))
parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/workspace/outputs/molmoact2-so101-move-blue-ball-success"))
parser.add_argument("--image-keys", default=os.environ.get("IMAGE_KEYS", '["observation.images.front","observation.images.wrist"]'))
parser.add_argument("--setup-type", default=os.environ.get("SETUP_TYPE", "single SO-101 follower arm moving a blue ball"))
parser.add_argument("--control-mode", default=os.environ.get("CONTROL_MODE", "absolute joint pose"))
parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "1000")))
parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
parser.add_argument("--save-freq", type=int, default=int(os.environ.get("SAVE_FREQ", "250")))
parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "molmoact-so101"))
parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
args = parser.parse_args()

cmd = [
    sys.executable, "-m", "accelerate.commands.launch", "--num_processes=1", "--mixed_precision=bf16",
    "-m", "lerobot.scripts.lerobot_train",
    f"--dataset.repo_id={args.dataset_repo}",
    f"--dataset.root={args.dataset_root}",
    f"--dataset.revision={args.dataset_revision}",
    "--dataset.video_backend=pyav",
    "--dataset.image_transforms.enable=true",
    "--policy.type=molmoact2",
    f"--policy.checkpoint_path={args.checkpoint}",
    "--policy.device=cuda",
    "--policy.action_mode=both",
    "--policy.chunk_size=10",
    "--policy.n_action_steps=10",
    f"--policy.setup_type={args.setup_type}",
    f"--policy.control_mode={args.control_mode}",
    f"--policy.image_keys={args.image_keys}",
    "--policy.model_dtype=bfloat16",
    "--policy.num_flow_timesteps=8",
    "--policy.gradient_checkpointing=true",
    "--policy.freeze_embedding=true",
    "--policy.normalize_gripper=false",
    "--policy.enable_knowledge_insulation=false",
    "--policy.push_to_hub=false",
    "--wandb.enable=true",
    f"--wandb.project={args.wandb_project}",
    f"--job_name={args.job_name}",
    f"--output_dir={args.output_dir}",
    f"--steps={args.steps}",
    f"--batch_size={args.batch_size}",
    f"--num_workers={args.num_workers}",
    "--log_freq=1",
    "--eval_freq=-1",
    "--save_checkpoint=true",
    f"--save_freq={args.save_freq}",
]
if args.wandb_entity:
    cmd.append(f"--wandb.entity={args.wandb_entity}")
print(subprocess.list2cmdline(cmd), flush=True)
subprocess.run(cmd, check=True)
PY

"${PYTHON_BIN}" /workspace/blupe_training/run_molmoact2_full_finetune.py 2>&1 | tee /workspace/molmoact2_full_finetune.log
