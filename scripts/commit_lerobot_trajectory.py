#!/usr/bin/env python3
"""Append successful raw policy trajectories to a LeRobot dataset.

The policy runner writes pending trajectories first so failed attempts do not pollute
training data. Run this after the success check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DEFAULT_DATASET_ROOT = "datasets/lerobot/blupe-yam-policy"
DEFAULT_REPO_ID = "local/blupe-yam-policy"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing trajectory metadata: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError as exc:
        raise SystemExit(f"missing trajectory steps: {path}") from exc
    rows = []
    for line_num, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL in {path}:{line_num}: {exc}") from exc
    return rows


def _valid_vector(value: Any, size: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(values) != size or not all(math.isfinite(v) for v in values):
        return None
    return values


def _valid_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _trajectory_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = [Path(p) for p in args.trajectory_dir]
    if args.pending_root:
        root = Path(args.pending_root)
        if not root.exists():
            raise SystemExit(f"pending root not found: {root}")
        dirs.extend(path for path in sorted(root.iterdir()) if (path / "episode.json").exists())

    unique = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise SystemExit("provide --trajectory-dir or --pending-root")
    return unique


def _load_plan(path: Path, allow_incomplete: bool) -> dict[str, Any]:
    meta = _read_json(path / "episode.json")
    steps = _read_jsonl(path / str(meta.get("steps_file") or "steps.jsonl"))
    status = str(meta.get("status") or "")
    if status != "complete" and not allow_incomplete:
        raise SystemExit(f"{path} status is {status!r}; pass --allow-incomplete to override")

    arm_joints = [str(name) for name in meta.get("joint_names") or DEFAULT_ARM_JOINTS]
    names = arm_joints + ["gripper"]
    usable = 0
    skipped = 0
    for step in steps:
        if _valid_vector(step.get("observation.state"), len(arm_joints)) is None:
            skipped += 1
            continue
        if _valid_vector(step.get("action"), len(arm_joints)) is None:
            skipped += 1
            continue
        usable += 1

    dts = [
        _valid_float(step.get("rollout_dt_s"), 0.0)
        for step in steps
        if _valid_float(step.get("rollout_dt_s"), 0.0) > 0
    ]
    return {
        "path": path,
        "meta": meta,
        "steps": steps,
        "arm_joints": arm_joints,
        "names": names,
        "usable": usable,
        "skipped": skipped,
        "median_dt_s": statistics.median(dts) if dts else 0.0,
        "task": str(meta.get("task") or meta.get("policy") or "YAM policy rollout"),
    }


def _infer_fps(plans: list[dict[str, Any]]) -> int:
    dts = [float(plan["median_dt_s"]) for plan in plans if float(plan["median_dt_s"]) > 0]
    if not dts:
        return 20
    fps = int(round(1.0 / statistics.median(dts)))
    return max(1, min(120, fps))


def _features(names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
    }


def _create_or_resume_dataset(
    *,
    root: Path,
    repo_id: str,
    fps: int,
    robot_type: str,
    features: dict[str, dict[str, Any]],
    overwrite: bool,
):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "lerobot is not installed in this Python environment. Run this from the "
            "LeRobot environment on the robot or workstation."
        ) from exc

    if root.exists() and overwrite:
        shutil.rmtree(root)
    if root.exists():
        return LeRobotDataset.resume(repo_id=repo_id, root=root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=False,
    )


def _add_episode(dataset, plan: dict[str, Any], task: str) -> int:
    import numpy as np

    added = 0
    arm_joints = plan["arm_joints"]
    for step in plan["steps"]:
        obs_arm = _valid_vector(step.get("observation.state"), len(arm_joints))
        act_arm = _valid_vector(step.get("action"), len(arm_joints))
        if obs_arm is None or act_arm is None:
            continue
        obs_gripper = _valid_float(step.get("observation.gripper"), _valid_float(step.get("gripper"), 0.0))
        act_gripper = _valid_float(step.get("gripper"), obs_gripper)
        dataset.add_frame(
            {
                "task": task,
                "observation.state": np.asarray(obs_arm + [obs_gripper], dtype=np.float32),
                "action": np.asarray(act_arm + [act_gripper], dtype=np.float32),
            }
        )
        added += 1
    if added == 0:
        dataset.clear_episode_buffer()
        raise SystemExit(f"no usable state/action steps in {plan['path']}")
    dataset.save_episode()
    return added


def _print_plan(plans: list[dict[str, Any]], fps: int) -> None:
    for plan in plans:
        print(
            json.dumps(
                {
                    "trajectory_dir": str(plan["path"]),
                    "run_id": plan["meta"].get("run_id"),
                    "status": plan["meta"].get("status"),
                    "task": plan["task"],
                    "steps": len(plan["steps"]),
                    "usable_steps": plan["usable"],
                    "skipped_steps": plan["skipped"],
                    "feature_names": plan["names"],
                    "median_rollout_dt_s": plan["median_dt_s"],
                    "dataset_fps": fps,
                },
                indent=2,
            )
        )


def _mark_committed(path: Path, *, root: Path, repo_id: str, fps: int, frames: int) -> None:
    meta_path = path / "episode.json"
    meta = _read_json(meta_path)
    meta["lerobot_commit"] = {
        "repo_id": repo_id,
        "root": str(root),
        "fps": fps,
        "frames": frames,
        "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append successful pending YAM policy trajectories to a LeRobot dataset."
    )
    parser.add_argument("--trajectory-dir", action="append", default=[], help="Pending trajectory directory.")
    parser.add_argument("--pending-root", default="", help="Append every trajectory under this root.")
    parser.add_argument("--root", default=DEFAULT_DATASET_ROOT, help="Local LeRobot dataset root.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Dataset repo id.")
    parser.add_argument("--robot-type", default="yam_follower")
    parser.add_argument("--task", default="", help="Override task text for all appended episodes.")
    parser.add_argument("--fps", type=int, default=0, help="Dataset FPS; defaults to median rollout step rate.")
    parser.add_argument("--overwrite", action="store_true", help="Remove and recreate --root before writing.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow non-complete pending trajectories.")
    parser.add_argument("--mark-committed", action="store_true", help="Write commit metadata back to episode.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print conversion plan without writing.")
    args = parser.parse_args()

    plans = [_load_plan(path, args.allow_incomplete) for path in _trajectory_dirs(args)]
    if any(plan["usable"] == 0 for plan in plans):
        bad = [str(plan["path"]) for plan in plans if plan["usable"] == 0]
        raise SystemExit(f"trajectories have no usable state/action samples: {bad}")
    first_names = plans[0]["names"]
    mismatched = [str(plan["path"]) for plan in plans if plan["names"] != first_names]
    if mismatched:
        raise SystemExit(f"trajectory feature names do not match first trajectory: {mismatched}")

    fps = args.fps if args.fps > 0 else _infer_fps(plans)
    _print_plan(plans, fps)
    if args.dry_run:
        return

    root = Path(args.root)
    dataset = _create_or_resume_dataset(
        root=root,
        repo_id=args.repo_id,
        fps=fps,
        robot_type=args.robot_type,
        features=_features(first_names),
        overwrite=args.overwrite,
    )
    try:
        total = 0
        per_plan_frames = {}
        for plan in plans:
            task = args.task.strip() or plan["task"]
            frames = _add_episode(dataset, plan, task)
            per_plan_frames[str(plan["path"])] = frames
            total += frames
            print(f"saved episode {plan['path']} frames={frames} task={task!r}")
    finally:
        dataset.finalize()

    if args.mark_committed:
        for plan in plans:
            _mark_committed(
                plan["path"],
                root=root,
                repo_id=args.repo_id,
                fps=fps,
                frames=per_plan_frames[str(plan["path"])],
            )
    print(f"wrote LeRobot dataset root={root} repo_id={args.repo_id} frames={total}")


if __name__ == "__main__":
    main()
