#!/usr/bin/env python3
"""Import a local LeRobot/LeLab dataset into editable raw recordings.

The dataset editor in this repo works on raw recording folders under
`recordings/`. LeLab/LeRobot writes v3 datasets as parquet metadata plus camera
videos. This script materializes each LeRobot episode into the raw recording
layout so it can be segmented and exported by the station UI.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEROBOT_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_RECORDINGS_ROOT = REPO_ROOT / "recordings"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return slug.strip("_.-") or "lerobot_dataset"


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        return Path(args.dataset_root).expanduser().resolve()
    if not args.repo_id:
        raise SystemExit("provide --dataset-root or --repo-id")
    return (DEFAULT_LEROBOT_ROOT / args.repo_id).expanduser().resolve()


def _camera_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    keys = []
    for key, spec in features.items():
        if not key.startswith("observation.images."):
            continue
        if isinstance(spec, dict) and spec.get("dtype") in {"video", "image"}:
            keys.append(key)
    return sorted(keys)


def _camera_name(key: str) -> str:
    name = key.rsplit(".", 1)[-1]
    if name.startswith("cam") and name[3:].isdigit():
        return {"cam0": "front", "cam1": "side", "cam2": "wrist"}.get(name, name)
    return name


def _read_parquets(paths: list[Path]):
    import pandas as pd

    if not paths:
        raise SystemExit("dataset has no parquet data files")
    frames = [pd.read_parquet(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _episode_tasks(dataset_root: Path) -> dict[int, str]:
    import pandas as pd

    tasks: dict[int, str] = {}
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            idx = int(row.get("episode_index", len(tasks)))
            values = row.get("tasks")
            if values is None:
                continue
            values = _jsonable(values)
            if isinstance(values, list):
                tasks[idx] = ", ".join(str(item) for item in values)
            else:
                tasks[idx] = str(values)
    return tasks


def _video_path(dataset_root: Path, info: dict[str, Any], camera_key: str, chunk: int, file_index: int) -> Path:
    template = str(info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    rel = template.format(video_key=camera_key, chunk_index=chunk, file_index=file_index)
    return dataset_root / rel


class _PyAVVideoReader:
    def __init__(self, path: Path):
        import av

        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._frames = self._container.decode(self._stream)
        self._next_index = 0
        self._last = None

    def read(self, frame_index: int):
        if frame_index < self._next_index - 1:
            raise RuntimeError("PyAV fallback only supports forward frame reads")
        while self._next_index <= frame_index:
            self._last = next(self._frames)
            self._next_index += 1
        return self._last.to_ndarray(format="bgr24")

    def release(self) -> None:
        self._container.close()


def _write_video_frames(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    camera_key: str,
    camera_name: str,
    episode_rows,
    out_dir: Path,
) -> int:
    import cv2

    cam_dir = out_dir / camera_name
    cam_dir.mkdir(parents=True, exist_ok=True)
    frames_jsonl = cam_dir / "frames.jsonl"
    frame_count = 0
    video_cache: dict[tuple[int, int], Any] = {}

    with frames_jsonl.open("w") as frames_out:
        for local_idx, row in enumerate(episode_rows.to_dict("records")):
            chunk = int(row.get(f"videos/{camera_key}/chunk_index", row.get("data/chunk_index", 0)) or 0)
            file_index = int(row.get(f"videos/{camera_key}/file_index", row.get("data/file_index", 0)) or 0)
            cache_key = (chunk, file_index)
            reader = video_cache.get(cache_key)
            if reader is None:
                path = _video_path(dataset_root, info, camera_key, chunk, file_index)
                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    reader = cap
                else:
                    reader = _PyAVVideoReader(path)
                video_cache[cache_key] = reader
            source_frame = int(row.get("frame_index", local_idx) or local_idx)
            if isinstance(reader, _PyAVVideoReader):
                try:
                    image = reader.read(source_frame)
                except Exception as exc:
                    raise SystemExit(f"could not read {camera_key} frame {source_frame}: {exc}") from exc
            else:
                reader.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                ok, image = reader.read()
                if not ok or image is None:
                    path = _video_path(dataset_root, info, camera_key, chunk, file_index)
                    reader.release()
                    reader = _PyAVVideoReader(path)
                    video_cache[cache_key] = reader
                    try:
                        image = reader.read(source_frame)
                    except Exception as exc:
                        raise SystemExit(f"could not read {camera_key} frame {source_frame}: {exc}") from exc
            rel_path = f"{camera_name}/frame_{local_idx:06d}.jpg"
            if not cv2.imwrite(str(out_dir / rel_path), image):
                raise SystemExit(f"could not write frame: {out_dir / rel_path}")
            frames_out.write(
                json.dumps(
                    {
                        "index": local_idx,
                        "timestamp": float(row.get("timestamp", local_idx / float(info.get("fps") or 30))),
                        "path": rel_path,
                        "source_frame_index": source_frame,
                    }
                )
                + "\n"
            )
            frame_count += 1

    for reader in video_cache.values():
        reader.release()
    return frame_count


def import_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = _resolve_dataset_root(args)
    info = _read_json(dataset_root / "meta" / "info.json")
    camera_keys = _camera_keys(info)
    if not camera_keys:
        raise SystemExit(f"no video/image camera features found in {dataset_root / 'meta' / 'info.json'}")

    data_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    data = _read_parquets(data_paths).sort_values(["episode_index", "frame_index"])
    tasks = _episode_tasks(dataset_root)
    recordings_root = Path(args.recordings_root).expanduser().resolve()
    recordings_root.mkdir(parents=True, exist_ok=True)

    repo_label = args.repo_id or dataset_root.name
    prefix = _safe_slug(args.name_prefix or f"lelab_{repo_label.replace('/', '_')}")
    fps = int(round(float(info.get("fps") or 30)))
    outputs = []

    for episode_index, episode_rows in data.groupby("episode_index", sort=True):
        episode_index = int(episode_index)
        out_dir = recordings_root / f"session_{prefix}_ep{episode_index:03d}"
        if out_dir.exists():
            if args.overwrite:
                shutil.rmtree(out_dir)
            else:
                raise SystemExit(f"output exists: {out_dir}; pass --overwrite to replace it")
        out_dir.mkdir(parents=True)

        cameras = []
        counts: dict[str, int] = {"samples": int(len(episode_rows))}
        for camera_id, camera_key in enumerate(camera_keys):
            name = _camera_name(camera_key)
            counts[name] = _write_video_frames(
                dataset_root=dataset_root,
                info=info,
                camera_key=camera_key,
                camera_name=name,
                episode_rows=episode_rows,
                out_dir=out_dir,
            )
            cameras.append(
                {
                    "id": camera_id,
                    "name": name,
                    "url": "",
                    "frames_dir": name,
                    "frames_file": f"{name}/frames.jsonl",
                    "lerobot_key": f"observation.images.{name}",
                    "source_lerobot_key": camera_key,
                }
            )

        with (out_dir / "samples.jsonl").open("w") as f:
            for local_idx, row in enumerate(episode_rows.to_dict("records")):
                sample = {
                    "index": local_idx,
                    "timestamp": float(row.get("timestamp", local_idx / fps)),
                    "frame_index": int(row.get("frame_index", local_idx)),
                    "episode_index": episode_index,
                    "observation_state": _jsonable(row.get("observation.state")),
                    "action": _jsonable(row.get("action")),
                    "source_index": int(row.get("index", local_idx)),
                }
                f.write(json.dumps(sample) + "\n")

        meta = {
            "source": "lelab_lerobot",
            "source_dataset_root": str(dataset_root),
            "source_repo_id": args.repo_id or "",
            "source_episode_index": episode_index,
            "robot_type": info.get("robot_type", ""),
            "task": tasks.get(episode_index, ""),
            "fps": fps,
            "sample_file": "samples.jsonl",
            "cameras": cameras,
            "counts": counts,
        }
        (out_dir / "session_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (out_dir / "segment_manifest.json").write_text(json.dumps({"segments": []}, indent=2) + "\n")
        outputs.append({"episode_index": episode_index, "path": str(out_dir), "counts": counts})

    return {"dataset_root": str(dataset_root), "recordings": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="", help="Local LeRobot/LeLab dataset root.")
    parser.add_argument("--repo-id", default="", help="Repo id under ~/.cache/huggingface/lerobot, e.g. andlyu/rebot_hackathon_v4.")
    parser.add_argument("--recordings-root", default=str(DEFAULT_RECORDINGS_ROOT))
    parser.add_argument("--name-prefix", default="", help="Output session prefix after session_.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(json.dumps(import_dataset(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
