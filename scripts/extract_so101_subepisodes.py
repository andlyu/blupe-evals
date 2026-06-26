#!/usr/bin/env python3
"""Extract labeled SO-101 subepisodes from one long recording."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES_ROOT = DEFAULT_REPO_ROOT / "episodes"
VALID_OUTCOMES = {"success", "failure"}
VALID_TYPES = {"teleop", "intervention"}


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
    rows: list[dict[str, Any]] = []
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _safe_slug(value: str, fallback: str = "segment") -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip().lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-_.") or fallback


def _timestamp(row: dict[str, Any]) -> float | None:
    try:
        return float(row.get("timestamp_s", row.get("timestamp")))
    except (TypeError, ValueError):
        return None


def _select_time_range(rows: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        ts = _timestamp(row)
        if ts is None:
            continue
        if start_s <= ts <= end_s:
            selected.append(row)
    return selected


def _metadata_path(source_dir: Path) -> Path:
    if (source_dir / "session_meta.json").exists():
        return source_dir / "session_meta.json"
    return source_dir / "episode_meta.json"


def _sample_file(meta: dict[str, Any], source_dir: Path) -> str:
    if "sample_file" in meta:
        return str(meta.get("sample_file") or "lerobot_samples.jsonl")
    if (source_dir / "samples.jsonl").exists():
        return "samples.jsonl"
    return "lerobot_samples.jsonl"


def _camera_specs(meta: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    known: list[dict[str, Any]] = []
    cameras = meta.get("cameras")
    if isinstance(cameras, list):
        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            name = str(cam.get("name") or f"cam{cam.get('id')}").strip()
            if not name or name == "camNone":
                continue
            frames_dir = str(cam.get("frames_dir") or name)
            frames_file = str(cam.get("frames_file") or f"{frames_dir}/frames.jsonl")
            known.append({**cam, "name": name, "frames_dir": frames_dir, "frames_file": frames_file})
    if not known:
        known = [
            {
                "id": idx,
                "name": f"cam{idx}",
                "frames_dir": f"cam{idx}",
                "frames_file": f"cam{idx}/frames.jsonl",
                "lerobot_key": f"observation.images.cam{idx}",
            }
            for idx in (0, 1)
        ]
    if not requested:
        return known
    requested_set = {str(name).strip() for name in requested if str(name).strip()}
    return [cam for cam in known if str(cam["name"]) in requested_set or str(cam["frames_dir"]) in requested_set]


def _camera_names(meta: dict[str, Any], requested: list[str]) -> list[str]:
    specs = _camera_specs(meta, requested)
    if specs:
        return [str(cam["name"]) for cam in specs]
    if requested:
        return [str(name).strip() for name in requested if str(name).strip()]
    return ["cam0", "cam1"]


def _normalize_segment(raw: dict[str, Any], index: int) -> dict[str, Any]:
    try:
        start_s = float(raw.get("start_s", raw.get("start", raw.get("from_s"))))
        end_s = float(raw.get("end_s", raw.get("end", raw.get("to_s"))))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"segment {index} needs numeric start_s and end_s") from exc
    if end_s <= start_s:
        raise SystemExit(f"segment {index} end_s must be greater than start_s")
    task = str(raw.get("task") or raw.get("prompt") or "").strip()
    if not task:
        raise SystemExit(f"segment {index} needs task/prompt text")
    outcome = str(raw.get("outcome") or "success").strip().lower()
    if outcome not in VALID_OUTCOMES:
        raise SystemExit(f"segment {index} outcome must be success or failure")
    segment_type = str(raw.get("type") or "teleop").strip().lower()
    if segment_type not in VALID_TYPES:
        raise SystemExit(f"segment {index} type must be teleop or intervention")
    return {
        "index": index,
        "start_s": start_s,
        "end_s": end_s,
        "task": task,
        "outcome": outcome,
        "type": segment_type,
        "notes": str(raw.get("notes") or "").strip(),
    }


def _load_segments(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_segments: list[dict[str, Any]] = []
    if args.segments_json:
        text = sys.stdin.read() if args.segments_json == "-" else Path(args.segments_json).read_text()
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("segments", [])
        if not isinstance(payload, list):
            raise SystemExit("--segments-json must contain a JSON list or an object with a segments list")
        for item in payload:
            if not isinstance(item, dict):
                raise SystemExit("each JSON segment must be an object")
            raw_segments.append(item)
    for spec in args.segment:
        parts = spec.split(":", 3)
        if len(parts) < 3:
            raise SystemExit("--segment must be START:END:TASK[:OUTCOME]")
        raw: dict[str, Any] = {"start_s": parts[0], "end_s": parts[1], "task": parts[2]}
        if len(parts) == 4:
            raw["outcome"] = parts[3]
        raw_segments.append(raw)
    if not raw_segments:
        raise SystemExit("provide --segments-json or at least one --segment")
    return [_normalize_segment(raw, idx) for idx, raw in enumerate(raw_segments, start=1)]


def _prepare_output_dir(output_root: Path, name: str, overwrite: bool) -> Path:
    out_dir = output_root / name
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return out_dir
    base = out_dir
    while out_dir.exists():
        out_dir = base.with_name(f"{base.name}_{uuid.uuid4().hex[:8]}")
    out_dir.mkdir(parents=True)
    return out_dir


def _copy_segment(
    *,
    source_dir: Path,
    output_root: Path,
    source_meta: dict[str, Any],
    samples: list[dict[str, Any]],
    camera_rows: dict[str, list[dict[str, Any]]],
    camera_specs: dict[str, dict[str, Any]],
    segment: dict[str, Any],
    fps: float,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    selected_samples = _select_time_range(samples, segment["start_s"], segment["end_s"])
    selected_camera_rows = {
        camera: _select_time_range(rows, segment["start_s"], segment["end_s"])
        for camera, rows in camera_rows.items()
    }
    lengths = [len(selected_samples), *(len(rows) for rows in selected_camera_rows.values())]
    common_len = min(lengths) if lengths else 0
    if common_len <= 0:
        raise SystemExit(
            f"segment {segment['index']} has no common samples/frames in "
            f"{segment['start_s']}..{segment['end_s']}s"
        )

    task_slug = _safe_slug(segment["task"], fallback=f"segment-{segment['index']:04d}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"so101_{segment['outcome']}_busyboard_{task_slug}_{stamp}_seg{segment['index']:04d}"
    out_dir = output_root / out_name if dry_run else _prepare_output_dir(output_root, out_name, overwrite)

    summary = {
        "segment": segment["index"],
        "task": segment["task"],
        "outcome": segment["outcome"],
        "start_s": segment["start_s"],
        "end_s": segment["end_s"],
        "frames": common_len,
        "output_dir": str(out_dir),
        "camera_counts": {camera: len(rows) for camera, rows in selected_camera_rows.items()},
        "sample_count": len(selected_samples),
    }
    if dry_run:
        return summary

    duration_s = common_len / fps if fps > 0 else max(0.0, segment["end_s"] - segment["start_s"])
    cameras_meta = []
    source_camera_meta = {
        str(cam.get("name") or f"cam{cam.get('id')}"): cam
        for cam in source_meta.get("cameras", [])
        if isinstance(cam, dict)
    }
    for camera in camera_rows:
        source_cam = camera_specs.get(camera) or source_camera_meta.get(camera, {})
        cam_id_raw = source_cam.get("id")
        try:
            cam_id = int(cam_id_raw)
        except (TypeError, ValueError):
            cam_id = int(camera.replace("cam", "")) if camera.replace("cam", "").isdigit() else 0
        frames_dir = str(source_cam.get("frames_dir") or camera)
        cameras_meta.append(
            {
                "id": cam_id,
                "name": camera,
                "url": source_cam.get("url", ""),
                "frames_dir": frames_dir,
                "frames_file": f"{frames_dir}/frames.jsonl",
                "lerobot_key": source_cam.get("lerobot_key") or f"observation.images.{camera}",
            }
        )

    meta = {
        "format": "blupe_so101_episode",
        "format_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": segment["task"],
        "outcome": segment["outcome"],
        "reason": "manifest_segment",
        "robot_type": source_meta.get("robot_type", "so101_follower"),
        "joints": source_meta.get("joints", DEFAULT_JOINTS),
        "state_units": source_meta.get("state_units", "degrees"),
        "action_units": source_meta.get("action_units", "degrees"),
        "fps": float(fps),
        "duration_s": round(duration_s, 6),
        "trajectory_time_source": "extracted_from_source_timestamp_s",
        "sample_file": "lerobot_samples.jsonl",
        "cameras": cameras_meta,
        "collection_type": segment["type"],
        "source_recording": source_dir.name,
        "source_episode": source_dir.name,
        "source_path": str(source_dir),
        "source_task": source_meta.get("task", ""),
        "source_capture_mode": source_meta.get("capture_mode", ""),
        "source_time_range_s": [segment["start_s"], segment["end_s"]],
        "segment_index": segment["index"],
        "segment_type": segment["type"],
        "notes": segment["notes"],
    }
    _write_json(out_dir / "episode_meta.json", meta)

    for new_idx, sample in enumerate(selected_samples[:common_len]):
        source_ts = _timestamp(sample)
        new_sample = dict(sample)
        new_sample["source_sample_idx"] = sample.get("sample_idx", sample.get("index"))
        new_sample["source_timestamp_s"] = source_ts
        new_sample["source_wall_elapsed_s"] = sample.get("wall_elapsed_s")
        new_sample["source_monotonic_s"] = sample.get("monotonic_s")
        new_sample["sample_idx"] = new_idx
        new_sample["timestamp_s"] = round(new_idx / fps, 6) if fps > 0 else round((source_ts or segment["start_s"]) - segment["start_s"], 6)
        new_sample["wall_elapsed_s"] = new_sample["timestamp_s"]
        new_sample["collection_type"] = segment["type"]
        new_sample["task"] = segment["task"]
        _append_jsonl(out_dir / "lerobot_samples.jsonl", new_sample)

    for camera, rows in selected_camera_rows.items():
        source_frames_dir = str((camera_specs.get(camera) or {}).get("frames_dir") or camera)
        target_frames_dir = str((camera_specs.get(camera) or {}).get("frames_dir") or camera)
        for new_idx, frame_row in enumerate(rows[:common_len]):
            frame_name = str(frame_row.get("frame") or frame_row.get("path") or "")
            if not frame_name:
                raise SystemExit(f"missing frame name for {camera} segment {segment['index']}")
            frame_path = Path(frame_name)
            source_frame = (
                source_dir / frame_path
                if frame_path.parts and frame_path.parts[0] == source_frames_dir
                else source_dir / source_frames_dir / frame_path.name
            )
            if not source_frame.exists():
                raise SystemExit(f"missing source frame: {source_frame}")
            target_name = f"frame_{new_idx:05d}{source_frame.suffix.lower() or '.jpg'}"
            target_frame = out_dir / target_frames_dir / target_name
            target_frame.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_frame, target_frame)
            source_ts = _timestamp(frame_row)
            new_frame_row = dict(frame_row)
            new_frame_row["source_frame_idx"] = frame_row.get("frame_idx", frame_row.get("index"))
            new_frame_row["source_frame"] = frame_name
            new_frame_row["source_timestamp_s"] = source_ts
            new_frame_row["source_wall_elapsed_s"] = frame_row.get("wall_elapsed_s")
            new_frame_row["source_monotonic_s"] = frame_row.get("monotonic_s")
            new_frame_row["frame_idx"] = new_idx
            new_frame_row["frame"] = target_name
            new_frame_row["timestamp_s"] = round(new_idx / fps, 6) if fps > 0 else round((source_ts or segment["start_s"]) - segment["start_s"], 6)
            new_frame_row["wall_elapsed_s"] = new_frame_row["timestamp_s"]
            _append_jsonl(out_dir / target_frames_dir / "frames.jsonl", new_frame_row)

    result = {
        "format": "blupe_so101_episode_result",
        "format_version": 1,
        "outcome": segment["outcome"],
        "reason": "manifest_segment",
        "started_at": meta["created_at"],
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": round(duration_s, 6),
        "counts": {"samples": common_len, **{camera: common_len for camera in camera_rows}},
        "source_episode": source_dir.name,
        "source_recording": source_dir.name,
        "source_time_range_s": [segment["start_s"], segment["end_s"]],
        "task": segment["task"],
        "segment_type": segment["type"],
    }
    _write_json(out_dir / "episode_result.json", result)
    return summary


def extract_segments(
    *,
    source_dir: Path,
    output_root: Path,
    segments: list[dict[str, Any]],
    cameras: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    meta_path = _metadata_path(source_dir)
    source_meta = _read_json(meta_path)
    if not source_meta:
        raise SystemExit(f"source recording has no episode_meta.json or session_meta.json: {source_dir}")
    sample_file = _sample_file(source_meta, source_dir)
    samples = _read_jsonl(source_dir / sample_file)
    if not samples:
        raise SystemExit(f"source episode has no samples: {source_dir / sample_file}")

    camera_specs_list = _camera_specs(source_meta, cameras or [])
    if not camera_specs_list:
        raise SystemExit(f"source recording has no matching cameras for request: {cameras}")
    camera_specs = {str(cam["name"]): cam for cam in camera_specs_list}
    camera_names = [str(cam["name"]) for cam in camera_specs_list]
    camera_rows = {
        camera: _read_jsonl(source_dir / str(camera_specs[camera].get("frames_file") or f"{camera}/frames.jsonl"))
        for camera in camera_names
    }
    empty_cameras = [camera for camera, rows in camera_rows.items() if not rows]
    if empty_cameras:
        raise SystemExit(f"source episode missing camera frames: {empty_cameras}")

    fps = float(source_meta.get("fps") or 0)
    if fps <= 0:
        raise SystemExit("source episode metadata has no positive fps")

    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _copy_segment(
            source_dir=source_dir,
            output_root=output_root,
            source_meta=source_meta,
            samples=samples,
            camera_rows=camera_rows,
            camera_specs=camera_specs,
            segment=segment,
            fps=fps,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        for segment in segments
    ]
    return {
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "dry_run": bool(dry_run),
        "segments": results,
        "episode_count": len(results),
        "total_frames": sum(int(item["frames"]) for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Long recording episode directory.")
    parser.add_argument("--output-root", default=str(DEFAULT_EPISODES_ROOT), help="Where subepisode folders are written.")
    parser.add_argument("--segments-json", default="", help="JSON file, or '-' for stdin. List of {start_s,end_s,task,outcome}.")
    parser.add_argument("--segment", action="append", default=[], help="START:END:TASK[:OUTCOME]. Repeatable.")
    parser.add_argument("--camera", action="append", default=[], help="Camera to include, e.g. cam0. Repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = extract_segments(
        source_dir=Path(args.source_dir),
        output_root=Path(args.output_root),
        segments=_load_segments(args),
        cameras=args.camera,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
