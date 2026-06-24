from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable


DEFAULT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL in {path}:{line_num}: {exc}") from exc
    return rows


def _frame_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return (stem[: -len(suffix)], int(suffix))
    return (stem, -1)


def _episode_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = [Path(p) for p in args.episode_dir]
    if args.episodes_root:
        root = Path(args.episodes_root)
        if not root.exists():
            raise SystemExit(f"episodes root not found: {root}")
        dirs.extend(
            path
            for path in sorted(root.iterdir())
            if path.is_dir() and (path / "episode_meta.json").exists()
        )
    unique: list[Path] = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise SystemExit("provide --episode-dir or --episodes-root containing episode_meta.json files")
    return unique


def _camera_names(meta: dict[str, Any], requested: list[str]) -> list[str]:
    if requested:
        return [name if name.startswith("cam") else f"cam{name}" for name in requested]
    cameras = meta.get("cameras")
    if isinstance(cameras, list) and cameras:
        names = [str(cam.get("name") or f"cam{cam.get('id')}") for cam in cameras]
        return [name for name in names if name and name != "camNone"]
    return ["cam0", "cam1"]


def _camera_key(camera_name: str) -> str:
    return f"observation.images.{camera_name}"


def _image_shape(path: Path) -> tuple[int, int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit("Pillow is required to read episode images") from exc

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    return (height, width, 3)


def _load_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _valid_vector(value: Any, size: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(values) != size:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _episode_plan(
    episode_dir: Path,
    requested_cameras: list[str],
    max_frames: int,
) -> dict[str, Any]:
    meta = _read_json(episode_dir / "episode_meta.json")
    sample_file = str(meta.get("sample_file") or "lerobot_samples.jsonl")
    samples = _read_jsonl(episode_dir / sample_file)
    joints = [str(j) for j in meta.get("joints", DEFAULT_JOINTS)]
    cameras = _camera_names(meta, requested_cameras)
    frame_paths: dict[str, list[Path]] = {}
    for camera in cameras:
        cam_dir = episode_dir / camera
        frames = sorted(cam_dir.glob("frame_*.jpg"), key=_frame_sort_key)
        if not frames:
            frames = sorted(cam_dir.glob("frame_*.jpeg"), key=_frame_sort_key)
        if not frames:
            frames = sorted(cam_dir.glob("frame_*.png"), key=_frame_sort_key)
        frame_paths[camera] = frames

    available_lengths = [len(samples), *(len(paths) for paths in frame_paths.values())]
    length = min(available_lengths) if available_lengths else 0
    if max_frames > 0:
        length = min(length, max_frames)

    usable = 0
    skipped = 0
    for idx in range(length):
        state = _valid_vector(samples[idx].get("observation_state"), len(joints))
        action = _valid_vector(samples[idx].get("action"), len(joints))
        if state is None or action is None:
            skipped += 1
        else:
            usable += 1

    return {
        "episode_dir": episode_dir,
        "meta": meta,
        "samples": samples,
        "joints": joints,
        "cameras": cameras,
        "frame_paths": frame_paths,
        "length": length,
        "usable": usable,
        "skipped": skipped,
        "task": str(meta.get("task") or ""),
    }


def _print_plan(plans: Iterable[dict[str, Any]]) -> None:
    for plan in plans:
        frame_counts = {camera: len(paths) for camera, paths in plan["frame_paths"].items()}
        print(
            json.dumps(
                {
                    "episode_dir": str(plan["episode_dir"]),
                    "task": plan["task"],
                    "samples": len(plan["samples"]),
                    "camera_frames": frame_counts,
                    "common_length": plan["length"],
                    "usable_frames": plan["usable"],
                    "skipped_frames": plan["skipped"],
                },
                indent=2,
            )
        )


def _build_features(plan: dict[str, Any], use_videos: bool) -> dict[str, dict[str, Any]]:
    joints = plan["joints"]
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joints),),
            "names": joints,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joints),),
            "names": joints,
        },
    }
    for camera, frames in plan["frame_paths"].items():
        if not frames:
            raise SystemExit(f"no frames for {camera} in {plan['episode_dir']}")
        features[_camera_key(camera)] = {
            "dtype": "video" if use_videos else "image",
            "shape": _image_shape(frames[0]),
            "names": ["height", "width", "channels"],
        }
    return features


def _create_or_resume_dataset(
    *,
    root: Path,
    repo_id: str,
    fps: int,
    features: dict[str, dict[str, Any]],
    use_videos: bool,
    append: bool,
    overwrite: bool,
):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "lerobot is not installed in this Python environment. Install LeRobot in the "
            "environment used for dataset conversion, then rerun this script."
        ) from exc

    if root.exists() and overwrite:
        shutil.rmtree(root)

    if root.exists() and append:
        resume = getattr(LeRobotDataset, "resume", None)
        if resume is not None:
            return resume(repo_id=repo_id, root=root)
        return LeRobotDataset(repo_id=repo_id, root=root)
    if root.exists():
        raise SystemExit(f"dataset root already exists: {root}; pass --append or --overwrite")

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type="so101_follower",
        features=features,
        use_videos=use_videos,
    )


def _add_episode(
    dataset,
    plan: dict[str, Any],
    task: str,
    parallel_encoding: bool,
) -> int:
    import numpy as np

    added = 0
    joints = plan["joints"]
    for idx in range(plan["length"]):
        sample = plan["samples"][idx]
        state = _valid_vector(sample.get("observation_state"), len(joints))
        action = _valid_vector(sample.get("action"), len(joints))
        if state is None or action is None:
            continue

        frame = {
            "task": task,
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }
        for camera, paths in plan["frame_paths"].items():
            frame[_camera_key(camera)] = _load_rgb(paths[idx])

        dataset.add_frame(frame)
        added += 1

    if added == 0:
        dataset.clear_episode_buffer()
        raise SystemExit(f"no usable frames in {plan['episode_dir']}")
    dataset.save_episode(parallel_encoding=parallel_encoding)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert recorded SO-101 episode folders into a LeRobotDataset."
    )
    parser.add_argument("--episode-dir", action="append", default=[], help="Episode directory to add.")
    parser.add_argument("--episodes-root", default="", help="Directory containing episode subdirectories.")
    parser.add_argument("--root", default="", help="Output LeRobot dataset root.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. local/blupe-so101.")
    parser.add_argument("--task", default="", help="Override task text for all converted episodes.")
    parser.add_argument("--fps", type=int, default=0, help="Dataset FPS. Defaults to episode metadata FPS.")
    parser.add_argument("--camera", action="append", default=[], help="Camera to include, e.g. cam0. Repeatable.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="Append to an existing local LeRobot dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Remove and recreate --root before writing.")
    parser.add_argument("--no-videos", action="store_true", help="Store images instead of encoded videos.")
    parser.add_argument("--no-parallel-encoding", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source episodes without writing.")
    args = parser.parse_args()

    episode_dirs = _episode_dirs(args)
    plans = [_episode_plan(path, args.camera, args.max_frames) for path in episode_dirs]
    _print_plan(plans)
    if args.dry_run:
        return
    if not args.root:
        raise SystemExit("--root is required unless --dry-run is set")

    if any(plan["usable"] == 0 for plan in plans):
        bad = [str(plan["episode_dir"]) for plan in plans if plan["usable"] == 0]
        raise SystemExit(f"episodes have no usable state/action samples: {bad}")

    first = plans[0]
    fps = args.fps or int(round(float(first["meta"].get("fps") or 0)))
    if fps <= 0:
        raise SystemExit("--fps is required when episode metadata has no positive fps")

    features = _build_features(first, use_videos=not args.no_videos)
    dataset = _create_or_resume_dataset(
        root=Path(args.root),
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        use_videos=not args.no_videos,
        append=args.append,
        overwrite=args.overwrite,
    )
    try:
        total = 0
        for plan in plans:
            task = args.task.strip() or plan["task"] or "SO-101 episode"
            total += _add_episode(
                dataset,
                plan,
                task,
                parallel_encoding=not args.no_parallel_encoding,
            )
            print(f"saved episode {plan['episode_dir']} frames={plan['usable']} task={task!r}")
    finally:
        dataset.finalize()

    print(f"wrote LeRobot dataset root={args.root} repo_id={args.repo_id} frames={total}")


if __name__ == "__main__":
    main()
