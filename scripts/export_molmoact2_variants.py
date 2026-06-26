#!/usr/bin/env python3
"""Export the four SO-101 MolmoAct2 LeRobot dataset variants.

Variants:
- teleop_front
- teleop_side
- teleop_intervention_front
- teleop_intervention_side
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import episode_to_lerobot_dataset as converter


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES_ROOT = DEFAULT_REPO_ROOT / "episodes"
DEFAULT_DATASETS_ROOT = DEFAULT_REPO_ROOT / "datasets" / "molmoact2"
DEFAULT_CAMERAS = ("front", "side")
VARIANT_SPECS = (
    ("teleop_front", ("teleop",), "front"),
    ("teleop_side", ("teleop",), "side"),
    ("teleop_intervention_front", ("teleop", "intervention"), "front"),
    ("teleop_intervention_side", ("teleop", "intervention"), "side"),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _episode_type(meta: dict[str, Any], result: dict[str, Any]) -> str:
    raw = (
        meta.get("type")
        or meta.get("segment_type")
        or meta.get("collection_type")
        or result.get("type")
        or result.get("segment_type")
        or result.get("collection_type")
        or ""
    )
    value = str(raw).strip().lower()
    if value in {"normal", "manual", "teleoperation", "busyboard_subepisode", "policy_execute"}:
        return "teleop"
    if value in {"intervene", "human_intervention"}:
        return "intervention"
    return value or "teleop"


def _episode_outcome(meta: dict[str, Any], result: dict[str, Any]) -> str:
    return str(result.get("outcome") or meta.get("outcome") or "").strip().lower()


def _has_camera(episode_dir: Path, meta: dict[str, Any], camera: str) -> bool:
    cameras = meta.get("cameras")
    if isinstance(cameras, list) and cameras:
        names = {
            str(cam.get("name") or f"cam{cam.get('id')}")
            for cam in cameras
            if isinstance(cam, dict)
        }
        if camera not in names:
            return False
    return (episode_dir / camera).is_dir()


def _episode_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "episode_meta.json").exists()
    )


def _select_episodes(
    root: Path,
    allowed_types: tuple[str, ...],
    camera: str,
    include_failures: bool,
) -> list[Path]:
    selected = []
    for path in _episode_dirs(root):
        meta = _read_json(path / "episode_meta.json")
        result = _read_json(path / "episode_result.json")
        outcome = _episode_outcome(meta, result)
        if outcome and outcome != "success" and not include_failures:
            continue
        if _episode_type(meta, result) not in allowed_types:
            continue
        if not _has_camera(path, meta, camera):
            continue
        selected.append(path)
    return selected


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _export_variant(
    *,
    name: str,
    episode_dirs: list[Path],
    camera: str,
    root: Path,
    repo_id: str,
    fps: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    plans = [converter._episode_plan(path, [camera], 0) for path in episode_dirs]
    frames = sum(int(plan["usable"]) for plan in plans)
    manifest = {
        "variant": name,
        "camera": camera,
        "lerobot_image_key": f"observation.images.{camera}",
        "dataset_root": str(root),
        "repo_id": repo_id,
        "episodes": [str(path) for path in episode_dirs],
        "episode_count": len(episode_dirs),
        "frames": frames,
    }
    if dry_run or not episode_dirs:
        return manifest
    if any(plan["usable"] == 0 for plan in plans):
        bad = [str(plan["episode_dir"]) for plan in plans if plan["usable"] == 0]
        raise SystemExit(f"{name} has episodes with no usable samples: {bad}")
    if root.exists() and overwrite:
        shutil.rmtree(root)
    first = plans[0]
    dataset_fps = fps or int(round(float(first["meta"].get("fps") or 0)))
    if dataset_fps <= 0:
        raise SystemExit(f"{name}: --fps is required when episode metadata has no positive fps")
    features = converter._build_features(first, use_videos=True)
    dataset = converter._create_or_resume_dataset(
        root=root,
        repo_id=repo_id,
        fps=dataset_fps,
        features=features,
        use_videos=True,
        append=False,
        overwrite=False,
    )
    try:
        for plan in plans:
            task = plan["task"] or "SO-101 episode"
            converter._add_episode(dataset, plan, task, parallel_encoding=True)
    finally:
        dataset.finalize()
    _write_manifest(root / "blupe_molmoact2_variant.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--datasets-root", default=str(DEFAULT_DATASETS_ROOT))
    parser.add_argument("--repo-prefix", default="local/so101-molmoact2")
    parser.add_argument("--variant", action="append", default=[], help="Variant name to export. Repeatable.")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested = set(args.variant)
    specs = [spec for spec in VARIANT_SPECS if not requested or spec[0] in requested]
    if requested:
        known = {name for name, _, _ in VARIANT_SPECS}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"unknown variants: {sorted(unknown)}; known variants: {sorted(known)}")

    episodes_root = Path(args.episodes_root)
    datasets_root = Path(args.datasets_root)
    manifests = []
    for name, allowed_types, camera in specs:
        episodes = _select_episodes(
            episodes_root,
            allowed_types=allowed_types,
            camera=camera,
            include_failures=args.include_failures,
        )
        root = datasets_root / name
        repo_id = f"{args.repo_prefix}-{name.replace('_', '-')}"
        manifests.append(
            _export_variant(
                name=name,
                episode_dirs=episodes,
                camera=camera,
                root=root,
                repo_id=repo_id,
                fps=args.fps,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )
    print(json.dumps({"variants": manifests}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
