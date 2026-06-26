#!/usr/bin/env python3
"""Convert SO101 LeRobot dataset joint columns between calibration conventions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blupe_evals.station.joint_conventions import (  # noqa: E402
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
    robot_state_to_policy_state,
)

DEFAULT_COLUMNS = ("observation.state", "action")
STATS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _vector_array(values: Any, *, column: str) -> np.ndarray:
    try:
        arr = np.stack([np.asarray(value, dtype=np.float32) for value in values])
    except Exception as exc:
        raise SystemExit(f"{column} must contain fixed-size numeric vectors") from exc
    if arr.ndim != 2:
        raise SystemExit(f"{column} must be a 2D vector column, got shape {arr.shape}")
    return arr


def _stats_for_array(arr: np.ndarray) -> dict[str, list[float] | list[int]]:
    arr64 = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr64.min(axis=0).tolist(),
        "max": arr64.max(axis=0).tolist(),
        "mean": arr64.mean(axis=0).tolist(),
        "std": arr64.std(axis=0).tolist(),
        "count": [int(arr64.shape[0])],
        "q01": np.quantile(arr64, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr64, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr64, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr64, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr64, 0.99, axis=0).tolist(),
    }


def convert_robot_v3_to_policy_v21(values: np.ndarray) -> np.ndarray:
    return robot_state_to_policy_state(
        values,
        policy_to_robot_signs=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
        policy_to_robot_offsets_deg=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    ).astype(np.float32)


def _replace_vector_column(df, column: str) -> tuple[Any, np.ndarray, np.ndarray]:
    before = _vector_array(df[column].to_numpy(), column=column)
    after = convert_robot_v3_to_policy_v21(before)
    next_df = df.copy()
    next_df[column] = [row.copy() for row in after]
    return next_df, before, after


def _update_stats_json(dataset_root: Path, arrays_by_column: dict[str, np.ndarray]) -> None:
    stats_path = dataset_root / "meta" / "stats.json"
    stats = _read_json(stats_path)
    for column, arr in arrays_by_column.items():
        stats[column] = _stats_for_array(arr)
    _write_json(stats_path, stats)


def _update_episode_stats(dataset_root: Path, arrays_by_column_by_episode: dict[str, dict[int, np.ndarray]]) -> None:
    import pandas as pd

    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        next_df = df.copy()
        for row_pos, episode_index in enumerate(next_df["episode_index"].astype(int).tolist()):
            row_idx = next_df.index[row_pos]
            for column, episode_arrays in arrays_by_column_by_episode.items():
                arr = episode_arrays.get(episode_index)
                if arr is None:
                    continue
                stats = _stats_for_array(arr)
                for name in STATS:
                    stats_column = f"stats/{column}/{name}"
                    if stats_column in next_df.columns:
                        next_df.at[row_idx, stats_column] = list(stats[name])
        next_df.to_parquet(path, index=False)


def convert_dataset(
    *,
    source_root: Path,
    dest_root: Path,
    overwrite: bool,
    columns: tuple[str, ...] = DEFAULT_COLUMNS,
    dest_repo_id: str = "",
) -> dict[str, Any]:
    import pandas as pd

    source_root = source_root.expanduser().resolve()
    dest_root = dest_root.expanduser().resolve()
    if not (source_root / "meta" / "info.json").exists():
        raise SystemExit(f"not a LeRobot dataset root: {source_root}")
    if dest_root.exists():
        if not overwrite:
            raise SystemExit(f"destination exists: {dest_root}; pass --overwrite")
        shutil.rmtree(dest_root)
    shutil.copytree(source_root, dest_root)

    info_path = dest_root / "meta" / "info.json"
    info = _read_json(info_path)
    if dest_repo_id:
        info["repo_id"] = dest_repo_id
    info["blupe_joint_convention"] = {
        "source": "lerobot_so101_v3_robot_degrees",
        "target": "molmoact2_so100_so101_v2_1_policy_degrees",
        "policy_to_robot_signs": DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS.tolist(),
        "policy_to_robot_offsets_deg": DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG.tolist(),
        "converted_columns": list(columns),
    }
    _write_json(info_path, info)

    arrays_by_column: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    arrays_by_column_by_episode: dict[str, dict[int, list[np.ndarray]]] = {column: {} for column in columns}
    frame_count = 0
    data_files = sorted((dest_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise SystemExit(f"no parquet files found under {dest_root / 'data'}")

    before_after: dict[str, dict[str, list[float]]] = {}
    for path in data_files:
        df = pd.read_parquet(path)
        next_df = df.copy()
        for column in columns:
            if column not in next_df.columns:
                raise SystemExit(f"missing column {column!r} in {path}")
            next_df, before, after = _replace_vector_column(next_df, column)
            arrays_by_column[column].append(after)
            for episode_index in sorted(set(next_df["episode_index"].astype(int).tolist())):
                mask = next_df["episode_index"].astype(int).to_numpy() == episode_index
                arrays_by_column_by_episode[column].setdefault(int(episode_index), []).append(after[mask])
            if column not in before_after:
                before_after[column] = {
                    "before_first": before[0].astype(float).tolist(),
                    "after_first": after[0].astype(float).tolist(),
                }
        frame_count += len(next_df)
        next_df.to_parquet(path, index=False)

    stacked_by_column = {
        column: np.concatenate(chunks, axis=0)
        for column, chunks in arrays_by_column.items()
    }
    episode_arrays = {
        column: {
            episode_index: np.concatenate(chunks, axis=0)
            for episode_index, chunks in episode_chunks.items()
        }
        for column, episode_chunks in arrays_by_column_by_episode.items()
    }
    _update_stats_json(dest_root, stacked_by_column)
    _update_episode_stats(dest_root, episode_arrays)

    summary = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "dest_repo_id": dest_repo_id,
        "data_files": len(data_files),
        "frames": frame_count,
        "columns": list(columns),
        "first_row": before_after,
        "stats": {
            column: {
                "min": np.round(arr.min(axis=0), 6).astype(float).tolist(),
                "max": np.round(arr.max(axis=0), 6).astype(float).tolist(),
                "mean": np.round(arr.mean(axis=0), 6).astype(float).tolist(),
            }
            for column, arr in stacked_by_column.items()
        },
    }
    _write_json(dest_root / "blupe_joint_convention_conversion.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Input LeRobot dataset root.")
    parser.add_argument("--dest-root", required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--dest-repo-id", default="", help="Repo id to record in meta/info.json.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = convert_dataset(
        source_root=Path(args.source_root),
        dest_root=Path(args.dest_root),
        overwrite=args.overwrite,
        dest_repo_id=args.dest_repo_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
