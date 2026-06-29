#!/usr/bin/env python3
"""Compress recorded SO-101 episodes into a local LeRobot dataset.

By default this selects successful episode folders that do not already have a
compression marker, converts them to LeRobot with video encoding enabled, then
marks each raw episode with the dataset location. Use --upload to publish the
compressed dataset folder to Hugging Face Hub after conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import episode_to_lerobot_dataset as converter


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES_ROOT = DEFAULT_REPO_ROOT / "episodes"
DEFAULT_DATASETS_ROOT = DEFAULT_REPO_ROOT / "datasets" / "lerobot"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-_.") or "so101-dataset"


def _candidate_episodes(
    root: Path,
    include_failures: bool,
    include_compressed: bool,
) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    allowed = {"success"}
    if include_failures:
        allowed.add("failure")
    for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime):
        if not path.is_dir() or not (path / "episode_meta.json").exists():
            continue
        result = _read_json(path / "episode_result.json")
        outcome = str(result.get("outcome") or _read_json(path / "episode_meta.json").get("outcome") or "")
        if outcome not in allowed:
            continue
        if result.get("compressed_dataset") and not include_compressed:
            continue
        out.append(path)
    return out


def _default_paths(args: argparse.Namespace, episodes: list[Path]) -> tuple[Path, str]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_prefix = "so101-ball-cup"
    if episodes:
        first = episodes[0].name
        parts = first.rsplit("_", 2)
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            stamp = f"{parts[-2]}_{parts[-1]}"
        collection_type = str(_read_json(episodes[0] / "episode_meta.json").get("collection_type") or "")
        if collection_type.startswith("busyboard"):
            dataset_prefix = "so101-busyboard"
    suffix = "mixed" if args.include_failures else "success"
    dataset_name = _safe_slug(args.dataset_name or f"{dataset_prefix}-{suffix}-{stamp}")
    root = Path(args.root) if args.root else Path(args.datasets_root) / dataset_name
    repo_id = args.repo_id or f"local/{dataset_name}"
    return root, repo_id


def _mark_compressed(episodes: list[Path], *, root: Path, repo_id: str, frames: int, uploaded: bool) -> None:
    payload = {
        "root": str(root),
        "repo_id": repo_id,
        "frames": int(frames),
        "uploaded": bool(uploaded),
        "compressed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for path in episodes:
        for name in ("episode_result.json", "episode_meta.json"):
            json_path = path / name
            data = _read_json(json_path)
            if not data:
                continue
            data["compressed_dataset"] = dict(payload)
            _write_json(json_path, data)


def _upload_folder(root: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit("huggingface_hub is not installed in this environment") from exc
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(root),
        commit_message=f"Upload SO-101 LeRobot dataset {root.name}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--datasets-root", default=str(DEFAULT_DATASETS_ROOT))
    parser.add_argument("--root", default="", help="Output local LeRobot dataset root.")
    parser.add_argument("--repo-id", default="", help="LeRobot/HF repo id.")
    parser.add_argument("--dataset-name", default="", help="Dataset folder/repo name when --root is omitted.")
    parser.add_argument("--include-failures", action="store_true", help="Include failure episodes too.")
    parser.add_argument("--include-compressed", action="store_true", help="Re-use episodes already marked compressed.")
    parser.add_argument("--append", action="store_true", help="Append to an existing local dataset root.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate the local dataset root.")
    parser.add_argument("--camera", action="append", default=[], help="Camera to include. Repeat for multiple cameras.")
    parser.add_argument("--skip-unusable", action="store_true", help="Skip successful episodes with no common usable frames.")
    parser.add_argument("--parallel-encoding", action="store_true", help="Encode videos in parallel. Sequential is safer on macOS.")
    parser.add_argument("--upload", action="store_true", help="Upload the compressed dataset to Hugging Face Hub.")
    parser.add_argument("--private", action="store_true", help="Create/upload the HF dataset as private.")
    parser.add_argument("--mark-existing-root", action="store_true", help="Mark selected raw episodes as compressed by --root without re-encoding.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    episodes_root = Path(args.episodes_root)
    episodes = _candidate_episodes(
        episodes_root,
        include_failures=args.include_failures,
        include_compressed=args.include_compressed,
    )
    if not episodes:
        print("no matching uncompressed episodes")
        return 0

    all_plans = [converter._episode_plan(path, args.camera, 0) for path in episodes]
    converter._print_plan(all_plans)
    unusable = [plan for plan in all_plans if int(plan["usable"]) == 0]
    if unusable and args.skip_unusable:
        for plan in unusable:
            print(f"skipping unusable episode {plan['episode_dir']}")
        plans = [plan for plan in all_plans if int(plan["usable"]) > 0]
    else:
        plans = all_plans
    if not plans:
        print("no usable episodes")
        return 0

    if any(plan["usable"] == 0 for plan in plans):
        bad = [str(plan["episode_dir"]) for plan in plans if plan["usable"] == 0]
        raise SystemExit(f"episodes have no usable state/action samples: {bad}")

    episodes = [Path(plan["episode_dir"]) for plan in plans]
    root, repo_id = _default_paths(args, episodes)
    total_frames = sum(int(plan["usable"]) for plan in plans)
    print(json.dumps({"dataset_root": str(root), "repo_id": repo_id, "episodes": len(episodes), "frames": total_frames}, indent=2))
    if args.dry_run:
        return 0

    if args.mark_existing_root:
        if not root.exists():
            raise SystemExit(f"dataset root not found: {root}")
        uploaded = False
        if args.upload:
            if repo_id.startswith("local/"):
                raise SystemExit("--upload requires --repo-id with a Hugging Face namespace, not local/...")
            _upload_folder(root, repo_id, private=args.private)
            uploaded = True
            print(f"uploaded dataset repo_id={repo_id}")
        _mark_compressed(episodes, root=root, repo_id=repo_id, frames=total_frames, uploaded=uploaded)
        print(f"marked compressed root={root} repo_id={repo_id} frames={total_frames}")
        return 0

    if root.exists() and args.overwrite:
        shutil.rmtree(root)

    first = plans[0]
    fps = int(round(float(first["meta"].get("fps") or 0)))
    if fps <= 0:
        raise SystemExit("episode metadata has no positive fps")
    features = converter._build_features(first, use_videos=True)
    dataset = converter._create_or_resume_dataset(
        root=root,
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        append=args.append,
        overwrite=False,
    )

    total = 0
    try:
        for plan in plans:
            task = plan["task"] or "SO-101 episode"
            frames = converter._add_episode(dataset, plan, task, parallel_encoding=args.parallel_encoding)
            total += frames
            print(f"saved episode {plan['episode_dir']} frames={frames} task={task!r}")
    finally:
        dataset.finalize()

    uploaded = False
    if args.upload:
        if repo_id.startswith("local/"):
            raise SystemExit("--upload requires --repo-id with a Hugging Face namespace, not local/...")
        _upload_folder(root, repo_id, private=args.private)
        uploaded = True
        print(f"uploaded dataset repo_id={repo_id}")

    _mark_compressed(episodes, root=root, repo_id=repo_id, frames=total, uploaded=uploaded)
    print(f"wrote LeRobot dataset root={root} repo_id={repo_id} frames={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
