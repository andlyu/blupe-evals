#!/usr/bin/env python3
"""Split a local LeRobot dataset into train/validation datasets by episode."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_image(value: Any) -> Image.Image:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if torch is not None and torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"unsupported image shape {arr.shape}")
    if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _parse_episode_list(value: str) -> list[int]:
    out: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(int(raw))
    return out


def _feature_subset(info: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    source = info.get("features") or {}
    features: dict[str, Any] = {}
    for key in [STATE_KEY, ACTION_KEY, *camera_keys]:
        value = source.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"missing feature spec for {key}")
        feature = dict(value)
        if isinstance(feature.get("shape"), list):
            feature["shape"] = tuple(feature["shape"])
        features[key] = feature
    return features


def _create_output_dataset(
    *,
    repo_id: str,
    root: Path,
    source_info: dict[str, Any],
    camera_keys: list[str],
    fps: int,
    overwrite: bool,
    vcodec: str,
    encoder_threads: int,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        if not overwrite:
            raise SystemExit(f"output root exists: {root}; pass --overwrite")
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=str(source_info.get("robot_type") or "so100_so101"),
        features=_feature_subset(source_info, camera_keys),
        use_videos=True,
        vcodec=vcodec,
        batch_encoding_size=1,
        encoder_threads=encoder_threads,
    )


def _copy_episodes(
    *,
    source_repo_id: str,
    source_root: Path,
    output_repo_id: str,
    output_root: Path,
    episodes: list[int],
    source_info: dict[str, Any],
    camera_keys: list[str],
    fps: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = _create_output_dataset(
        repo_id=output_repo_id,
        root=output_root,
        source_info=source_info,
        camera_keys=camera_keys,
        fps=fps,
        overwrite=args.overwrite,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
    )
    frames = 0
    try:
        for episode in episodes:
            ds = LeRobotDataset(
                source_repo_id,
                root=source_root,
                episodes=[episode],
                video_backend=args.video_backend,
            )
            added = 0
            for idx in range(len(ds)):
                item = ds[idx]
                row = {
                    "task": str(item.get("task") or "SO100/SO101 task"),
                    STATE_KEY: _to_numpy(item[STATE_KEY]).astype(np.float32),
                    ACTION_KEY: _to_numpy(item[ACTION_KEY]).astype(np.float32),
                }
                for camera_key in camera_keys:
                    row[camera_key] = _to_image(item[camera_key])
                out.add_frame(row)
                added += 1
            if added == 0:
                out.clear_episode_buffer()
                raise RuntimeError(f"episode {episode} produced no frames")
            out.save_episode(parallel_encoding=not args.no_parallel_encoding)
            frames += added
    finally:
        out.finalize()
    return {
        "repo_id": output_repo_id,
        "root": str(output_root),
        "episodes": episodes,
        "frames": frames,
    }


def split_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = Path(args.source_root)
    source_info = _read_json(source_root / "meta" / "info.json")
    source = LeRobotDataset(args.source_repo_id, root=source_root, video_backend=args.video_backend)
    total_episodes = int(source.num_episodes)
    if total_episodes <= 0:
        raise SystemExit("source dataset has no episodes")

    if args.val_episodes:
        val_episodes = _parse_episode_list(args.val_episodes)
    else:
        if args.val_count <= 0 or args.val_count >= total_episodes:
            raise SystemExit("--val-count must be between 1 and total_episodes - 1")
        val_episodes = list(range(total_episodes - args.val_count, total_episodes))
    invalid = [episode for episode in val_episodes if episode < 0 or episode >= total_episodes]
    if invalid:
        raise SystemExit(f"validation episode indices out of range: {invalid}")
    val_set = set(val_episodes)
    train_episodes = [episode for episode in range(total_episodes) if episode not in val_set]
    if not train_episodes:
        raise SystemExit("training split would be empty")

    camera_keys = list(source.meta.camera_keys)
    if not camera_keys:
        raise SystemExit("source dataset has no camera keys")

    train = _copy_episodes(
        source_repo_id=args.source_repo_id,
        source_root=source_root,
        output_repo_id=args.train_repo_id,
        output_root=Path(args.train_root),
        episodes=train_episodes,
        source_info=source_info,
        camera_keys=camera_keys,
        fps=int(source.fps),
        args=args,
    )
    val = _copy_episodes(
        source_repo_id=args.source_repo_id,
        source_root=source_root,
        output_repo_id=args.val_repo_id,
        output_root=Path(args.val_root),
        episodes=val_episodes,
        source_info=source_info,
        camera_keys=camera_keys,
        fps=int(source.fps),
        args=args,
    )
    return {
        "source_repo_id": args.source_repo_id,
        "source_root": str(source_root),
        "total_episodes": total_episodes,
        "camera_keys": camera_keys,
        "train": train,
        "validation": val,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--val-repo-id", required=True)
    parser.add_argument("--val-count", type=int, default=1)
    parser.add_argument("--val-episodes", default="")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--no-parallel-encoding", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(json.dumps(split_dataset(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
