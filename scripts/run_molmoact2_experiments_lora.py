#!/usr/bin/env python3
"""Register the blue-ball dataset and launch MolmoAct2 experiments LoRA training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _set_default_env(experiments_dir: Path, lerobot_data_root: str) -> None:
    _load_env_file(Path("/workspace/blupe_training/wandb.env"))
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("MOLMO_DATA_DIR", lerobot_data_root)
    os.environ.setdefault("LEROBOT_DATA_ROOT", lerobot_data_root)
    os.environ.setdefault("LEROBOT_VIDEO_BACKEND", "pyav")
    os.environ.setdefault("WANDB_PROJECT", "molmoact-so101")
    os.environ.setdefault("WANDB_ENTITY", "")

    pythonpath = [
        str(experiments_dir),
        str(experiments_dir / "lerobot" / "src"),
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)

    for path in reversed(pythonpath[:2]):
        if path not in sys.path:
            sys.path.insert(0, path)


def _register_move_blue_ball_mixture() -> None:
    from launch_scripts import data_mixtures

    def build_move_blue_ball():
        return data_mixtures.build_single_lerobot_mixture(
            name="move_blue_ball",
            tag="so101_move_blue_ball",
            repo_ids=["andlyu/move_blue_ball_training"],
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.front",
                "observation.images.wrist",
            ],
            normalize_gripper=True,
            setup_type="single SO-101 follower arm moving a blue ball",
            control_mode="absolute joint pose",
            action_horizon=30,
            n_action_steps=30,
        )

    data_mixtures.MOLMOACT2_LEROBOT_MIXTURES["move_blue_ball"] = build_move_blue_ball


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default=os.environ.get("MOLMOACT2_EXPERIMENTS_DIR", "/workspace/molmoact2/experiments"))
    parser.add_argument("--lerobot-data-root", default=os.environ.get("LEROBOT_DATA_ROOT", "/workspace/lerobot_data"))
    parser.add_argument("--checkpoint", default=os.environ.get("MOLMOACT2_CHECKPOINT", "allenai/MolmoAct2-SO100_101"))
    parser.add_argument("--mixture", default="move_blue_ball")
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "molmoact2-so101-move-blue-ball-lora"))
    parser.add_argument("--save-folder", default=os.environ.get("SAVE_FOLDER", "/workspace/outputs/molmoact2-so101-move-blue-ball-lora"))
    parser.add_argument("--max-duration", type=int, default=int(os.environ.get("MAX_DURATION", "1000")))
    parser.add_argument("--device-batch-size", type=int, default=int(os.environ.get("DEVICE_BATCH_SIZE", "1")))
    parser.add_argument("--global-batch-size", type=int, default=int(os.environ.get("GLOBAL_BATCH_SIZE", "8")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    parser.add_argument("--data-timeout", type=int, default=int(os.environ.get("DATA_TIMEOUT", "-1")))
    parser.add_argument("--save-interval", type=int, default=int(os.environ.get("SAVE_INTERVAL", "0")))
    parser.add_argument("--lora-save-interval", type=int, default=int(os.environ.get("LORA_SAVE_INTERVAL", "250")))
    parser.add_argument("--save-keep", type=int, default=int(os.environ.get("SAVE_KEEP", "20")))
    parser.add_argument("--lora-save-keep", type=int, default=int(os.environ.get("LORA_SAVE_KEEP", "20")))
    parser.add_argument("--lora-rank", type=int, default=int(os.environ.get("LORA_RANK", "64")))
    parser.add_argument("--log-interval", type=int, default=int(os.environ.get("LOG_INTERVAL", "1")))
    parser.add_argument("--save-merged-lora", action="store_true")
    parser.add_argument("--offline-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    _set_default_env(experiments_dir, args.lerobot_data_root)
    if args.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"

    _register_move_blue_ball_mixture()

    data_timeout = args.data_timeout
    if data_timeout < 0:
        data_timeout = 0 if args.num_workers == 0 else 900
    save_interval = args.save_interval if args.save_interval > 0 else args.max_duration

    train_argv = [
        "launch_scripts/train_lerobot.py",
        args.checkpoint,
        args.mixture,
        f"--wandb.name={args.run_name}",
        f"--max_duration={args.max_duration}",
        f"--device_batch_size={args.device_batch_size}",
        f"--global_batch_size={args.global_batch_size}",
        f"--log_interval={args.log_interval}",
        f"--num_workers={args.num_workers}",
        "--pin_memory=true",
        f"--data.timeout={data_timeout}",
        f"--save_interval={save_interval}",
        f"--save_lora_interval={args.lora_save_interval}",
        f"--save_num_checkpoints_to_keep={args.save_keep}",
        f"--save_lora_num_checkpoints_to_keep={args.lora_save_keep}",
        f"--save_folder={args.save_folder}",
        "--packing=false",
        "--dynamic_seq_len=true",
        "--ft_vlm=true",
        "--ft_action_expert=true",
        "--ft_embedding=lm_head",
        "--lora_enable=true",
        f"--lora_rank={args.lora_rank}",
        f"--save_merged_lora_checkpoint={str(args.save_merged_lora).lower()}",
        "--llm_learning_rate=5e-5",
        "--vit_learning_rate=5e-5",
        "--connector_learning_rate=5e-5",
        "--action_expert_learning_rate=5e-5",
        "--num_flow_timesteps=8",
        "--mask_action_dim_padding=true",
        "--random_camera_order=none",
        "--frame_loading_backend=torchcodec_exact",
        "--use_annotated_task=false",
        "--sample_annotated_task=false",
    ]

    print(" ".join(train_argv), flush=True)
    if args.dry_run:
        return 0

    from launch_scripts import train_lerobot

    sys.argv = train_argv
    train_lerobot.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
