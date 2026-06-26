#!/usr/bin/env python3
"""Sample one episode from random MolmoAct2 SO100/SO101 datasets.

The upstream MolmoAct2 SO100/SO101 mixture contains many LeRobot v2.1 repos.
LeRobot 0.5.x cannot load those directly without converting the full repo, so
this script reads one selected v2.1 episode directly and writes it into a new
v3.0 LeRobot dataset with canonical camera keys:

  - observation.images.camera1
  - observation.images.camera2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CAMERA1_KEY = "observation.images.camera1"
CAMERA2_KEY = "observation.images.camera2"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class SampledEpisode:
    repo_id: str
    revision: str
    codebase_version: str
    episode_index: int
    task: str
    source_camera_keys: list[str]
    frames: int


@dataclass
class SampleReport:
    selected: list[SampledEpisode] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def add_skip(self, repo_id: str, reason: str) -> None:
        self.skipped.append({"repo_id": repo_id, "reason": reason})


def _add_experiments_dir(experiments_dir: Path) -> None:
    for path in (experiments_dir, experiments_dir / "lerobot" / "src"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_original_repo_ids(experiments_dir: Path) -> list[str]:
    _add_experiments_dir(experiments_dir)
    try:
        from launch_scripts.data_constants import SO100_SO101_MOLMOACT2
    except Exception as exc:
        raise SystemExit(
            f"failed to import SO100_SO101_MOLMOACT2 from {experiments_dir}"
        ) from exc
    repos = [str(repo).removeprefix("lerobot:").strip() for repo in SO100_SO101_MOLMOACT2]
    repos = [repo for repo in repos if repo]
    if not repos:
        raise SystemExit("SO100_SO101_MOLMOACT2 is empty")
    return repos


def _safe_dir_name(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id).strip("_") or "repo"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _snapshot_metadata(repo_id: str, root: Path, revisions: list[str]) -> tuple[str, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    last_exc: Exception | None = None
    for revision in revisions:
        revision_root = root / revision.replace("/", "__")
        if revision_root.exists():
            shutil.rmtree(revision_root)
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=revision_root,
                allow_patterns=["meta/*", "meta/**"],
            )
            info = _read_json(revision_root / "meta" / "info.json")
            return revision, info
        except Exception as exc:
            last_exc = exc
            shutil.rmtree(revision_root, ignore_errors=True)
    raise RuntimeError(f"metadata download failed: {last_exc}")


def _camera_keys_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    keys = [
        str(key)
        for key, value in features.items()
        if str(key).startswith("observation.images")
        and isinstance(value, dict)
        and value.get("dtype") in {"video", "image"}
    ]
    return keys


def _feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...]:
    features = info.get("features") or {}
    value = features.get(key) or {}
    return tuple(int(v) for v in value.get("shape") or ())


def _valid_vector(value: Any, *, dim: int = 6) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape != (dim,):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _to_uint8_hwc(value: Any) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None and torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, Image.Image):
        arr = np.asarray(value.convert("RGB"))
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
    return arr


def _resize_image(value: Any, *, width: int, height: int) -> Image.Image:
    arr = _to_uint8_hwc(value)
    image = Image.fromarray(arr, mode="RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    return image


def _decode_video(path: Path, *, width: int, height: int, max_frames: int = 0) -> list[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {path}")
    frames: list[Image.Image] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_resize_image(rgb, width=width, height=height))
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def _make_output_dataset(args: argparse.Namespace):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.output_root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"output root exists: {root}; pass --overwrite")
        shutil.rmtree(root)

    features = {
        STATE_KEY: {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        CAMERA1_KEY: {
            "dtype": "video",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channels"],
        },
        CAMERA2_KEY: {
            "dtype": "video",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        robot_type="so100_so101",
        features=features,
        use_videos=True,
        vcodec=args.vcodec,
        batch_encoding_size=1,
        encoder_threads=args.encoder_threads,
    )


def _save_frames(dataset: Any, rows: list[dict[str, Any]], *, parallel_encoding: bool) -> int:
    added = 0
    for row in rows:
        dataset.add_frame(row)
        added += 1
    if added == 0:
        dataset.clear_episode_buffer()
        return 0
    dataset.save_episode(parallel_encoding=parallel_encoding)
    return added


def _load_v3_episode_rows(
    *,
    repo_id: str,
    root: Path,
    revision: str,
    episode_index: int,
    camera_keys: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from olmo.data.lerobot_wrapper import _MolmoLeRobotDataset as LeRobotDataset
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[no-redef]

    ds = LeRobotDataset(
        repo_id,
        root=root,
        episodes=[episode_index],
        revision=revision,
        download_videos=True,
        video_backend=args.video_backend,
    )
    if len(camera_keys) < 2:
        raise RuntimeError(f"expected at least two cameras, got {camera_keys}")
    source_cameras = camera_keys[:2]
    rows: list[dict[str, Any]] = []
    task = ""
    max_len = min(len(ds), args.max_frames_per_episode or len(ds))
    for idx in range(max_len):
        item = ds[idx]
        state = _valid_vector(item.get(STATE_KEY))
        action = _valid_vector(item.get(ACTION_KEY))
        if state is None or action is None:
            continue
        task = str(item.get("task") or task or "")
        rows.append(
            {
                "task": task or "SO100/SO101 task",
                STATE_KEY: state,
                ACTION_KEY: action,
                CAMERA1_KEY: _resize_image(
                    item[source_cameras[0]],
                    width=args.image_width,
                    height=args.image_height,
                ),
                CAMERA2_KEY: _resize_image(
                    item[source_cameras[1]],
                    width=args.image_width,
                    height=args.image_height,
                ),
            }
        )
    return rows, task or "SO100/SO101 task"


def _load_v21_episode_rows(
    *,
    repo_id: str,
    root: Path,
    revision: str,
    info: dict[str, Any],
    episode_index: int,
    camera_keys: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    from huggingface_hub import snapshot_download
    import pandas as pd

    if len(camera_keys) < 2:
        raise RuntimeError(f"expected at least two cameras, got {camera_keys}")
    source_cameras = camera_keys[:2]
    chunks_size = int(info.get("chunks_size") or 1000)
    episode_chunk = int(episode_index) // chunks_size
    data_path = str(info["data_path"]).format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
    )
    video_paths = [
        str(info["video_path"]).format(
            episode_chunk=episode_chunk,
            episode_index=int(episode_index),
            video_key=camera_key,
        )
        for camera_key in source_cameras
    ]
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=[
            "meta/*",
            "meta/**",
            data_path,
            *video_paths,
        ],
    )
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks if "task_index" in row}
    ep_meta = next((row for row in episodes if int(row.get("episode_index", -1)) == int(episode_index)), {})
    task = ""
    ep_tasks = ep_meta.get("tasks")
    if isinstance(ep_tasks, list) and ep_tasks:
        task = str(ep_tasks[0])
    if not task and task_by_index:
        task = next(iter(task_by_index.values()))
    task = task or "SO100/SO101 task"

    df = pd.read_parquet(root / data_path)
    camera_frames = [
        _decode_video(
            root / video_path,
            width=args.image_width,
            height=args.image_height,
            max_frames=args.max_frames_per_episode,
        )
        for video_path in video_paths
    ]
    length = min(len(df), *(len(frames) for frames in camera_frames))
    if args.max_frames_per_episode:
        length = min(length, int(args.max_frames_per_episode))

    rows: list[dict[str, Any]] = []
    for idx in range(length):
        row = df.iloc[idx]
        state = _valid_vector(row.get(STATE_KEY))
        action = _valid_vector(row.get(ACTION_KEY))
        if state is None or action is None:
            continue
        rows.append(
            {
                "task": task,
                STATE_KEY: state,
                ACTION_KEY: action,
                CAMERA1_KEY: camera_frames[0][idx],
                CAMERA2_KEY: camera_frames[1][idx],
            }
        )
    return rows, task


def _load_episode_rows(
    *,
    repo_id: str,
    repo_root: Path,
    revision: str,
    info: dict[str, Any],
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], SampledEpisode]:
    if _feature_shape(info, STATE_KEY) != (6,) or _feature_shape(info, ACTION_KEY) != (6,):
        raise RuntimeError(
            f"incompatible state/action shapes: state={_feature_shape(info, STATE_KEY)} "
            f"action={_feature_shape(info, ACTION_KEY)}"
        )
    camera_keys = _camera_keys_from_info(info)
    if len(camera_keys) < 2:
        raise RuntimeError(f"not enough cameras: {camera_keys}")

    total_episodes = int(info.get("total_episodes") or 0)
    if total_episodes <= 0:
        raise RuntimeError("no episodes")
    episode_index = rng.randrange(total_episodes)

    codebase_version = str(info.get("codebase_version") or "")
    data_root = repo_root / revision.replace("/", "__")
    if codebase_version.startswith("v2"):
        rows, task = _load_v21_episode_rows(
            repo_id=repo_id,
            root=data_root,
            revision=revision,
            info=info,
            episode_index=episode_index,
            camera_keys=camera_keys,
            args=args,
        )
    else:
        rows, task = _load_v3_episode_rows(
            repo_id=repo_id,
            root=data_root,
            revision=revision,
            episode_index=episode_index,
            camera_keys=camera_keys,
            args=args,
        )
    if not rows:
        raise RuntimeError("selected episode produced no usable rows")
    return rows, SampledEpisode(
        repo_id=repo_id,
        revision=revision,
        codebase_version=codebase_version,
        episode_index=episode_index,
        task=task,
        source_camera_keys=camera_keys[:2],
        frames=len(rows),
    )


def _write_manifest(output_root: Path, args: argparse.Namespace, report: SampleReport) -> None:
    payload = {
        "repo_id": args.repo_id,
        "count_requested": args.count,
        "seed": args.seed,
        "fps": args.fps,
        "image_size": [args.image_width, args.image_height],
        "camera_keys": [CAMERA1_KEY, CAMERA2_KEY],
        "selected": [episode.__dict__ for episode in report.selected],
        "skipped": report.skipped,
    }
    (output_root / "blupe_sample_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="/workspace/molmoact2/experiments")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--cache-root", default="/tmp/blupe_so100_so101_sample_cache")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--revision", action="append", default=[])
    parser.add_argument("--keep-source-cache", action="store_true")
    parser.add_argument("--no-parallel-encoding", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.image_width <= 0 or args.image_height <= 0:
        raise SystemExit("--image-width and --image-height must be positive")

    experiments_dir = Path(args.experiments_dir)
    repos = _load_original_repo_ids(experiments_dir)
    rng = random.Random(args.seed)
    rng.shuffle(repos)
    if args.candidate_limit:
        repos = repos[: args.candidate_limit]

    revisions = args.revision or ["v3.0", "main", "v2.1"]
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    report = SampleReport()
    dataset = None if args.dry_run else _make_output_dataset(args)

    try:
        for repo_id in repos:
            if len(report.selected) >= args.count:
                break
            repo_root = cache_root / _safe_dir_name(repo_id)
            try:
                revision, info = _snapshot_metadata(repo_id, repo_root, revisions)
                rows, sampled = _load_episode_rows(
                    repo_id=repo_id,
                    repo_root=repo_root,
                    revision=revision,
                    info=info,
                    rng=rng,
                    args=args,
                )
                if not args.dry_run and dataset is not None:
                    _save_frames(
                        dataset,
                        rows,
                        parallel_encoding=not args.no_parallel_encoding,
                    )
                report.selected.append(sampled)
                print(
                    json.dumps(
                        {
                            "selected": len(report.selected),
                            "repo_id": sampled.repo_id,
                            "revision": sampled.revision,
                            "episode_index": sampled.episode_index,
                            "frames": sampled.frames,
                            "task": sampled.task,
                            "source_camera_keys": sampled.source_camera_keys,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                report.add_skip(repo_id, reason[:500])
                print(json.dumps({"skip": repo_id, "reason": reason[:300]}), flush=True)
            finally:
                if not args.keep_source_cache:
                    shutil.rmtree(repo_root, ignore_errors=True)
    finally:
        if dataset is not None:
            dataset.finalize()

    if len(report.selected) < args.count:
        raise SystemExit(
            f"only sampled {len(report.selected)} usable episodes out of requested {args.count}; "
            f"skipped {len(report.skipped)}"
        )
    if not args.dry_run:
        _write_manifest(output_root, args, report)
    else:
        print(json.dumps({"dry_run_selected": [item.__dict__ for item in report.selected]}, indent=2))

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "repo_id": args.repo_id,
                "selected": len(report.selected),
                "skipped": len(report.skipped),
                "frames": sum(item.frames for item in report.selected),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
