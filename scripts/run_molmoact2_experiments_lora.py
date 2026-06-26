#!/usr/bin/env python3
"""Register the blue-ball dataset and launch MolmoAct2 experiments LoRA training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
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
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

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


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    tag: str
    camera_keys: list[str]
    rate: float
    setup_type: str


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_dataset_spec(value: str) -> DatasetSpec:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--dataset-spec must be 'repo_id|tag|camera_key_1,camera_key_2|rate|setup_type'"
        )
    repo_id, tag, camera_keys_text, rate_text, setup_type = parts
    camera_keys = _parse_csv(camera_keys_text)
    if not repo_id or not tag or len(camera_keys) < 1 or not setup_type:
        raise argparse.ArgumentTypeError(
            "--dataset-spec repo_id, tag, camera keys, and setup_type must be non-empty"
        )
    try:
        rate = float(rate_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--dataset-spec rate must be a float") from exc
    if rate <= 0:
        raise argparse.ArgumentTypeError("--dataset-spec rate must be positive")
    return DatasetSpec(
        repo_id=repo_id,
        tag=tag,
        camera_keys=camera_keys,
        rate=rate,
        setup_type=setup_type,
    )


def _build_lerobot_mixture_payload(mixture_name: str, specs: list[DatasetSpec]):
    from launch_scripts import data_mixtures

    data_mixture = []
    metadata_per_tag = {}
    for spec in specs:
        partial_mixture, partial_metadata = data_mixtures.build_single_lerobot_mixture(
            name=mixture_name,
            tag=spec.tag,
            repo_ids=[spec.repo_id],
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=spec.camera_keys,
            normalize_gripper=True,
            setup_type=spec.setup_type,
            control_mode="absolute joint pose",
            action_horizon=30,
            n_action_steps=30,
            rate=spec.rate,
        )
        data_mixture.extend(partial_mixture)
        metadata_per_tag.update(partial_metadata)
    return data_mixture, metadata_per_tag


def _register_lerobot_mixture(
    *,
    mixture_name: str,
    specs: list[DatasetSpec],
) -> None:
    from launch_scripts import data_mixtures

    def build_mixture():
        return _build_lerobot_mixture_payload(mixture_name, specs)

    data_mixtures.MOLMOACT2_LEROBOT_MIXTURES[mixture_name] = build_mixture


def _load_env_json(name: str) -> object | None:
    raw = os.environ.get(name)
    if raw:
        return json.loads(raw)
    path = os.environ.get(f"{name}_PATH")
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _bare_lerobot_tag(tag: str) -> str:
    prefix = "lerobot:"
    return tag[len(prefix):] if tag.startswith(prefix) else tag


def _repo_root_for_spec(lerobot_data_root: str, repo_id: str) -> Path:
    root_base = Path(lerobot_data_root)
    repo_path = Path(repo_id)
    if repo_path.is_absolute():
        return repo_path
    return root_base / repo_path


def _read_dataset_counts(root: Path) -> tuple[int, int]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return 0, 0
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return 0, 0
    return int(info.get("total_episodes") or 0), int(info.get("total_frames") or 0)


def _data_mix_metrics(
    *,
    specs: list[DatasetSpec],
    lerobot_data_root: str,
    custom_tags: set[str],
) -> dict[str, float]:
    custom_rate = 0.0
    general_rate = 0.0
    custom_episodes = 0
    general_episodes = 0
    custom_frames = 0
    general_frames = 0
    for spec in specs:
        episodes, frames = _read_dataset_counts(_repo_root_for_spec(lerobot_data_root, spec.repo_id))
        if spec.tag in custom_tags:
            custom_rate += float(spec.rate)
            custom_episodes += episodes
            custom_frames += frames
        else:
            general_rate += float(spec.rate)
            general_episodes += episodes
            general_frames += frames

    total_rate = custom_rate + general_rate
    total_episodes = custom_episodes + general_episodes
    total_frames = custom_frames + general_frames
    metrics = {
        "data_mix/custom_sample_pct": 100.0 * custom_rate / total_rate if total_rate else 0.0,
        "data_mix/general_sample_pct": 100.0 * general_rate / total_rate if total_rate else 0.0,
        "data_mix/custom_episode_pct": 100.0 * custom_episodes / total_episodes if total_episodes else 0.0,
        "data_mix/general_episode_pct": 100.0 * general_episodes / total_episodes if total_episodes else 0.0,
        "data_mix/custom_frame_pct": 100.0 * custom_frames / total_frames if total_frames else 0.0,
        "data_mix/general_frame_pct": 100.0 * general_frames / total_frames if total_frames else 0.0,
        "data_mix/custom_sample_weight": custom_rate,
        "data_mix/general_sample_weight": general_rate,
        "data_mix/custom_episodes": float(custom_episodes),
        "data_mix/general_episodes": float(general_episodes),
        "data_mix/custom_frames": float(custom_frames),
        "data_mix/general_frames": float(general_frames),
    }
    return metrics


def _patch_data_mix_metrics(extra_metrics: dict[str, float]) -> None:
    import olmo.train.trainer as trainer_mod

    original = trainer_mod.lerobot_tag_sampling_rate_metrics

    def lerobot_tag_sampling_rate_metrics_with_mix(*args, **kwargs):
        metrics = dict(original(*args, **kwargs))
        metrics.update(extra_metrics)
        return metrics

    trainer_mod.lerobot_tag_sampling_rate_metrics = lerobot_tag_sampling_rate_metrics_with_mix


def _patch_validation_evaluators(
    train_lerobot,
    *,
    validation_specs: list[DatasetSpec],
    eval_interval: int,
    eval_max_examples: int,
    eval_device_batch_size: int,
    log_interval: int,
) -> None:
    from launch_scripts.lerobot_utils.env import _store_env_json_in_file
    from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig

    original_run_trainer = train_lerobot.run_trainer

    def run_trainer_with_validation(conf):
        repo_to_tag = _load_env_json("LEROBOT_REPO_TO_TAG") or {}
        tag_metadata = _load_env_json("LEROBOT_TAG_METADATA") or {}
        if not isinstance(repo_to_tag, dict) or not isinstance(tag_metadata, dict):
            raise RuntimeError("LeRobot tag metadata was not initialized before validation injection")

        evaluators = list(conf.evaluators or [])
        for spec in validation_specs:
            repo_to_tag[spec.repo_id] = spec.tag
            _, spec_metadata = _build_lerobot_mixture_payload("validation", [spec])
            for raw_tag, metadata in spec_metadata.items():
                tag_metadata.setdefault(_bare_lerobot_tag(str(raw_tag)), metadata)

            val_mixture, _ = _build_lerobot_mixture_payload("validation", [spec])
            val_data = replace(
                conf.data,
                kwargs_mixture=val_mixture,
                shuffle=False,
                drop_last=False,
                seed=conf.data.seed + len(evaluators) + 1,
                packing=None,
                skip_overlong_examples=False,
            )
            evaluators.append(
                LossDatasetEvaluatorConfig(
                    label=f"val_{spec.tag}",
                    data=val_data,
                    device_batch_size=eval_device_batch_size,
                    max_examples=eval_max_examples if eval_max_examples > 0 else None,
                    console_log_interval=log_interval,
                    response_logits_only=bool(conf.response_logits_only),
                )
            )

        _store_env_json_in_file("LEROBOT_REPO_TO_TAG", repo_to_tag)
        _store_env_json_in_file("LEROBOT_TAG_METADATA", tag_metadata)
        conf.evaluators = evaluators
        conf.eval_interval = eval_interval
        conf.inf_eval_interval = -1
        return original_run_trainer(conf)

    train_lerobot.run_trainer = run_trainer_with_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default=os.environ.get("MOLMOACT2_EXPERIMENTS_DIR", "/workspace/molmoact2/experiments"))
    parser.add_argument("--lerobot-data-root", default=os.environ.get("LEROBOT_DATA_ROOT", "/workspace/lerobot_data"))
    parser.add_argument("--checkpoint", default=os.environ.get("MOLMOACT2_CHECKPOINT", "allenai/MolmoAct2-SO100_101"))
    parser.add_argument("--mixture", default="move_blue_ball")
    parser.add_argument("--dataset-repo-id", default=os.environ.get("DATASET_REPO_ID", "andlyu/move_blue_ball_training"))
    parser.add_argument("--dataset-tag", default=os.environ.get("DATASET_TAG", "so101_move_blue_ball"))
    parser.add_argument(
        "--camera-keys",
        default=os.environ.get("CAMERA_KEYS", "observation.images.front,observation.images.wrist"),
    )
    parser.add_argument(
        "--setup-type",
        default=os.environ.get("SETUP_TYPE", "single SO-101 follower arm moving a blue ball"),
    )
    parser.add_argument(
        "--dataset-spec",
        action="append",
        type=_parse_dataset_spec,
        default=[],
        help=(
            "Add a dataset to a multi-tag mixture as "
            "'repo_id|tag|camera_key_1,camera_key_2|rate|setup_type'. "
            "When provided, these replace --dataset-repo-id/--dataset-tag/--camera-keys."
        ),
    )
    parser.add_argument(
        "--validation-dataset-spec",
        action="append",
        type=_parse_dataset_spec,
        default=[],
        help=(
            "Add a held-out LeRobot dataset to validation loss tracking as "
            "'repo_id|tag|camera_key_1,camera_key_2|rate|setup_type'. Tags should match training tags."
        ),
    )
    parser.add_argument("--eval-interval", type=int, default=int(os.environ.get("EVAL_INTERVAL", "25")))
    parser.add_argument("--eval-max-examples", type=int, default=int(os.environ.get("EVAL_MAX_EXAMPLES", "0")))
    parser.add_argument("--eval-device-batch-size", type=int, default=int(os.environ.get("EVAL_DEVICE_BATCH_SIZE", "1")))
    parser.add_argument(
        "--custom-tag",
        action="append",
        default=[],
        help="Training tag to count as custom/new data in W&B data_mix metrics. Repeatable.",
    )
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

    specs = args.dataset_spec or [
        DatasetSpec(
            repo_id=args.dataset_repo_id,
            tag=args.dataset_tag,
            camera_keys=_parse_csv(args.camera_keys),
            rate=1.0,
            setup_type=args.setup_type,
        )
    ]
    _register_lerobot_mixture(mixture_name=args.mixture, specs=specs)
    custom_tags = set(args.custom_tag or [args.dataset_tag])
    data_mix_metrics = _data_mix_metrics(
        specs=specs,
        lerobot_data_root=args.lerobot_data_root,
        custom_tags=custom_tags,
    )

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
    print("data_mix=" + json.dumps(data_mix_metrics, sort_keys=True), flush=True)
    if args.validation_dataset_spec:
        print(
            "validation="
            + json.dumps(
                [
                    {
                        "repo_id": spec.repo_id,
                        "tag": spec.tag,
                        "camera_keys": spec.camera_keys,
                        "rate": spec.rate,
                    }
                    for spec in args.validation_dataset_spec
                ],
                sort_keys=True,
            ),
            flush=True,
        )
    if args.dry_run:
        return 0

    from launch_scripts import train_lerobot

    _patch_data_mix_metrics(data_mix_metrics)

    if args.validation_dataset_spec:
        _patch_validation_evaluators(
            train_lerobot,
            validation_specs=args.validation_dataset_spec,
            eval_interval=args.eval_interval,
            eval_max_examples=args.eval_max_examples,
            eval_device_batch_size=args.eval_device_batch_size,
            log_interval=args.log_interval,
        )

    sys.argv = train_argv
    train_lerobot.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
