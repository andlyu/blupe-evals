#!/usr/bin/env python3
"""Register the blue-ball dataset and launch MolmoAct2 experiments LoRA training."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
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


def _parse_episode_selector(raw_episodes: str) -> list[int]:
    episodes: list[int] = []
    for part in raw_episodes.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)
            if end < start:
                start, end = end, start
            episodes.extend(range(start, end + 1))
        else:
            episodes.append(int(part))
    if not episodes:
        raise ValueError(f"no episodes in selector {raw_episodes!r}")
    return sorted(set(episodes))


def _split_repo_episode_selector(repo_id: str) -> tuple[str, list[int] | None]:
    if "@" not in repo_id:
        return repo_id, None
    base_repo_id, raw_episodes = repo_id.split("@", 1)
    base_repo_id = base_repo_id.strip()
    if not base_repo_id:
        raise ValueError(f"invalid episode-qualified repo id {repo_id!r}")
    return base_repo_id, _parse_episode_selector(raw_episodes)


def _parse_dataset_spec(value: str) -> DatasetSpec:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--dataset-spec must be 'repo_id[@episodes]|tag|camera_key_1,camera_key_2|rate|setup_type'"
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


def _metric_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return slug or "unknown"


def _repo_segment_name(repo_id: str) -> str:
    base_repo_id, _episodes = _split_repo_episode_selector(repo_id)
    return _metric_slug(base_repo_id.rsplit("/", 1)[-1])


def _episode_selector_text(repo_id: str) -> str:
    if "@" not in repo_id:
        return "all"
    return repo_id.split("@", 1)[1].strip() or "all"


def _repo_root_for_spec(lerobot_data_root: str, repo_id: str) -> Path:
    repo_id, _episodes = _split_repo_episode_selector(repo_id)
    root_base = Path(lerobot_data_root)
    repo_path = Path(repo_id)
    if repo_path.is_absolute():
        return repo_path
    return root_base / repo_path


def _selected_episode_frame_count(root: Path, episodes: list[int]) -> int | None:
    try:
        import pandas as pd
    except Exception:
        return None

    selected = set(episodes)
    data_root = root / "data"
    if data_root.exists():
        frames = 0
        for path in sorted(data_root.glob("**/*.parquet")):
            try:
                df = pd.read_parquet(path, columns=["episode_index"])
            except Exception:
                continue
            if "episode_index" not in df:
                continue
            frames += int(df["episode_index"].isin(selected).sum())
        return frames

    episodes_root = root / "meta" / "episodes"
    frames = 0
    found = False
    if episodes_root.exists():
        for path in sorted(episodes_root.glob("**/*.parquet")):
            try:
                df = pd.read_parquet(path, columns=["episode_index", "length"])
            except Exception:
                continue
            if "episode_index" not in df or "length" not in df:
                continue
            mask = df["episode_index"].isin(selected)
            if bool(mask.any()):
                found = True
                frames += int(df.loc[mask, "length"].sum())
        if found:
            return frames

    return None


def _read_dataset_counts(root: Path) -> tuple[int, int]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return 0, 0
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return 0, 0
    return int(info.get("total_episodes") or 0), int(info.get("total_frames") or 0)


def _read_dataset_counts_for_spec(lerobot_data_root: str, repo_id: str) -> tuple[int, int]:
    root = _repo_root_for_spec(lerobot_data_root, repo_id)
    total_episodes, total_frames = _read_dataset_counts(root)
    _base_repo_id, selected_episodes = _split_repo_episode_selector(repo_id)
    if selected_episodes is None:
        return total_episodes, total_frames
    selected_count = len(selected_episodes)
    selected_frames = _selected_episode_frame_count(root, selected_episodes)
    if selected_frames is None and total_episodes > 0:
        selected_frames = round(total_frames * selected_count / total_episodes)
    return selected_count, int(selected_frames or 0)


def _data_segment_rows(
    *,
    specs: list[DatasetSpec],
    validation_specs: list[DatasetSpec],
    lerobot_data_root: str,
    custom_tags: set[str],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    train_rate_total = sum(float(spec.rate) for spec in specs)

    def add_row(split: str, spec: DatasetSpec, sample_pct: float) -> None:
        episodes, frames = _read_dataset_counts_for_spec(lerobot_data_root, spec.repo_id)
        group = "custom" if spec.tag in custom_tags else "general"
        rows.append(
            {
                "split": split,
                "segment": _repo_segment_name(spec.repo_id),
                "repo_id": _split_repo_episode_selector(spec.repo_id)[0],
                "episode_selector": _episode_selector_text(spec.repo_id),
                "tag": spec.tag,
                "group": group,
                "sample_weight": float(spec.rate),
                "sample_pct": sample_pct,
                "episodes": float(episodes),
                "frames": float(frames),
            }
        )

    for spec in specs:
        sample_pct = 100.0 * float(spec.rate) / train_rate_total if train_rate_total else 0.0
        add_row("train", spec, sample_pct)
    for spec in validation_specs:
        add_row("validation", spec, 0.0)
    return rows


def _data_segment_scalar_metrics(rows: list[dict[str, float | str]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        split = str(row["split"])
        segment = _metric_slug(str(row["segment"]))
        prefix = f"data_segments/{split}/{segment}"
        metrics[f"{prefix}/episodes"] = float(row["episodes"])
        metrics[f"{prefix}/frames"] = float(row["frames"])
        if split == "train":
            metrics[f"{prefix}/sample_pct"] = float(row["sample_pct"])
            metrics[f"{prefix}/sample_weight"] = float(row["sample_weight"])
    return metrics


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
        episodes, frames = _read_dataset_counts_for_spec(lerobot_data_root, spec.repo_id)
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


def _wandb_table(rows: list[dict[str, float | str]]):
    import wandb

    columns = list(rows[0])
    return wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in rows])


def _data_mix_rows(extra_metrics: dict[str, float]) -> list[dict[str, float | str]]:
    return [
        {
            "source": "custom",
            "sample_pct": extra_metrics["data_mix/custom_sample_pct"],
            "episode_pct": extra_metrics["data_mix/custom_episode_pct"],
            "frame_pct": extra_metrics["data_mix/custom_frame_pct"],
            "sample_weight": extra_metrics["data_mix/custom_sample_weight"],
            "episodes": extra_metrics["data_mix/custom_episodes"],
            "frames": extra_metrics["data_mix/custom_frames"],
        },
        {
            "source": "general",
            "sample_pct": extra_metrics["data_mix/general_sample_pct"],
            "episode_pct": extra_metrics["data_mix/general_episode_pct"],
            "frame_pct": extra_metrics["data_mix/general_frame_pct"],
            "sample_weight": extra_metrics["data_mix/general_sample_weight"],
            "episodes": extra_metrics["data_mix/general_episodes"],
            "frames": extra_metrics["data_mix/general_frames"],
        },
    ]


def _log_data_mix_bar_charts(
    extra_metrics: dict[str, float],
    segment_rows: list[dict[str, float | str]],
) -> bool:
    import wandb

    if wandb.run is None:
        return False

    rows = _data_mix_rows(extra_metrics)
    table = _wandb_table(rows)
    payload = {
        "data_mix/table": table,
        "data_mix/sample_pct_bar": wandb.plot.bar(
            table,
            "source",
            "sample_pct",
            title="Data Mix: Sampling %",
        ),
        "data_mix/episode_pct_bar": wandb.plot.bar(
            table,
            "source",
            "episode_pct",
            title="Data Mix: Episode %",
        ),
        "data_mix/frame_pct_bar": wandb.plot.bar(
            table,
            "source",
            "frame_pct",
            title="Data Mix: Frame %",
        ),
    }

    if segment_rows:
        segment_table = _wandb_table(segment_rows)
        payload["data_segments/table"] = segment_table
        train_rows = [row for row in segment_rows if row["split"] == "train"]
        validation_rows = [row for row in segment_rows if row["split"] == "validation"]
        if train_rows:
            train_table = _wandb_table(train_rows)
            payload["data_segments/train_table"] = train_table
            payload["data_segments/train_episodes_bar"] = wandb.plot.bar(
                train_table,
                "segment",
                "episodes",
                title="Train Segments: Episodes",
            )
            payload["data_segments/train_frames_bar"] = wandb.plot.bar(
                train_table,
                "segment",
                "frames",
                title="Train Segments: Frames",
            )
        if validation_rows:
            validation_table = _wandb_table(validation_rows)
            payload["data_segments/validation_table"] = validation_table
            payload["data_segments/validation_episodes_bar"] = wandb.plot.bar(
                validation_table,
                "segment",
                "episodes",
                title="Validation Segments: Episodes",
            )
            payload["data_segments/validation_frames_bar"] = wandb.plot.bar(
                validation_table,
                "segment",
                "frames",
                title="Validation Segments: Frames",
            )

    wandb.log(payload, step=0, commit=False)
    return True


def _patch_data_mix_metrics(
    extra_metrics: dict[str, float],
    segment_rows: list[dict[str, float | str]],
) -> None:
    import olmo.train.trainer as trainer_mod

    original = trainer_mod.lerobot_tag_sampling_rate_metrics
    logged_chart = False

    def lerobot_tag_sampling_rate_metrics_with_mix(*args, **kwargs):
        nonlocal logged_chart
        metrics = dict(original(*args, **kwargs))
        metrics.update(extra_metrics)
        if not logged_chart:
            try:
                logged_chart = _log_data_mix_bar_charts(extra_metrics, segment_rows)
            except Exception as exc:
                print(f"warning: failed to log W&B data mix bar chart: {exc}", file=sys.stderr, flush=True)
                logged_chart = True
        return metrics

    trainer_mod.lerobot_tag_sampling_rate_metrics = lerobot_tag_sampling_rate_metrics_with_mix


def _runtime_split_metrics(
    *,
    elapsed_seconds: float,
    train_step_seconds: float,
    validation_seconds: float,
) -> dict[str, float]:
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    train_step_seconds = max(float(train_step_seconds), 0.0)
    validation_seconds = max(float(validation_seconds), 0.0)
    training_seconds = max(elapsed_seconds - validation_seconds, 0.0)
    overhead_seconds = max(training_seconds - train_step_seconds, 0.0)

    def pct(seconds: float) -> float:
        return 100.0 * seconds / elapsed_seconds if elapsed_seconds else 0.0

    return {
        "runtime/elapsed_seconds": elapsed_seconds,
        "runtime/training_seconds": training_seconds,
        "runtime/validation_seconds": validation_seconds,
        "runtime/train_step_seconds": train_step_seconds,
        "runtime/overhead_seconds": overhead_seconds,
        "runtime/training_pct": pct(training_seconds),
        "runtime/validation_pct": pct(validation_seconds),
        "runtime/train_step_pct": pct(train_step_seconds),
        "runtime/overhead_pct": pct(overhead_seconds),
    }


def _patch_runtime_split_metrics() -> None:
    import olmo.train.trainer as trainer_mod

    if getattr(trainer_mod.Trainer, "_blupe_runtime_split_patched", False):
        return

    original_fit = trainer_mod.Trainer.fit
    original_loss_eval = trainer_mod.Trainer.loss_eval
    original_train_step = trainer_mod.Trainer.train_step

    def ensure_runtime_state(trainer) -> None:
        if not hasattr(trainer, "_blupe_runtime_start_time"):
            trainer._blupe_runtime_start_time = time.perf_counter()
            trainer._blupe_train_step_seconds = 0.0
            trainer._blupe_validation_seconds = 0.0

    def runtime_metrics(trainer) -> dict[str, float]:
        ensure_runtime_state(trainer)
        elapsed_seconds = time.perf_counter() - trainer._blupe_runtime_start_time
        return _runtime_split_metrics(
            elapsed_seconds=elapsed_seconds,
            train_step_seconds=trainer._blupe_train_step_seconds,
            validation_seconds=trainer._blupe_validation_seconds,
        )

    def fit_with_runtime_split(trainer, *args, **kwargs):
        trainer._blupe_runtime_start_time = time.perf_counter()
        trainer._blupe_train_step_seconds = 0.0
        trainer._blupe_validation_seconds = 0.0
        return original_fit(trainer, *args, **kwargs)

    def train_step_with_runtime_split(trainer, *args, **kwargs):
        ensure_runtime_state(trainer)
        t0 = time.perf_counter()
        metrics = original_train_step(trainer, *args, **kwargs)
        trainer._blupe_train_step_seconds += time.perf_counter() - t0
        if isinstance(metrics, dict):
            metrics.update(runtime_metrics(trainer))
        return metrics

    def loss_eval_with_runtime_split(trainer, *args, **kwargs):
        ensure_runtime_state(trainer)
        t0 = time.perf_counter()
        metrics = original_loss_eval(trainer, *args, **kwargs)
        trainer._blupe_validation_seconds += time.perf_counter() - t0
        if isinstance(metrics, dict):
            metrics = dict(metrics)
            metrics.update(runtime_metrics(trainer))
        return metrics

    trainer_mod.Trainer.fit = fit_with_runtime_split
    trainer_mod.Trainer.train_step = train_step_with_runtime_split
    trainer_mod.Trainer.loss_eval = loss_eval_with_runtime_split
    trainer_mod.Trainer._blupe_runtime_split_patched = True


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
    tag_counts = Counter(spec.tag for spec in validation_specs)

    def validation_label(spec: DatasetSpec) -> str:
        label = f"val_{spec.tag}"
        if tag_counts[spec.tag] <= 1:
            return label
        repo_id, _ = _split_repo_episode_selector(spec.repo_id)
        repo_slug = repo_id.rsplit("/", 1)[-1]
        repo_slug = re.sub(r"[^A-Za-z0-9_]+", "_", repo_slug).strip("_")
        return f"{label}_{repo_slug}"

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

            val_data = replace(
                conf.data,
                dataset=f"lerobot:{spec.repo_id}",
                mixture=None,
                root_size_mixture=None,
                kwargs_mixture=None,
                shuffle=False,
                drop_last=False,
                seed=conf.data.seed + len(evaluators) + 1,
                packing=None,
                skip_overlong_examples=False,
            )
            evaluators.append(
                LossDatasetEvaluatorConfig(
                    label=validation_label(spec),
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
    parser.add_argument("--save-keep", type=int, default=int(os.environ.get("SAVE_KEEP", "20")))
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
    data_segment_rows = _data_segment_rows(
        specs=specs,
        validation_specs=args.validation_dataset_spec,
        lerobot_data_root=args.lerobot_data_root,
        custom_tags=custom_tags,
    )
    data_mix_metrics.update(_data_segment_scalar_metrics(data_segment_rows))

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
        f"--save_num_checkpoints_to_keep={args.save_keep}",
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
    print("data_segments=" + json.dumps(data_segment_rows, sort_keys=True), flush=True)
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

    _patch_data_mix_metrics(data_mix_metrics, data_segment_rows)
    _patch_runtime_split_metrics()

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
