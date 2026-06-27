import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.convert_lerobot_joint_convention import convert_dataset


def _write_tiny_dataset(root: Path) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(exist_ok=True)

    rows = pd.DataFrame(
        {
            "observation.state": [
                np.array([0, -90, 70, 0, -45, 0], dtype=np.float32),
                np.array([1, -80, 75, 2, -44, 3], dtype=np.float32),
            ],
            "action": [
                np.array([0, -89, 71, 0, -45, 1], dtype=np.float32),
                np.array([2, -79, 76, 3, -43, 4], dtype=np.float32),
            ],
            "timestamp": np.array([0.0, 1 / 30], dtype=np.float32),
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "index": [0, 1],
            "task_index": [0, 0],
        }
    )
    rows.to_parquet(root / "data" / "chunk-000" / "file-000.parquet", index=False)

    stats = {
        "observation.state": {"mean": [0, -85, 72.5, 1, -44.5, 1.5]},
        "action": {"mean": [1, -84, 73.5, 1.5, -44, 2.5]},
    }
    (root / "meta" / "stats.json").write_text(json.dumps(stats) + "\n")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "so101_follower",
                "total_episodes": 1,
                "total_frames": 2,
                "fps": 30,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [6]},
                    "action": {"dtype": "float32", "shape": [6]},
                },
            }
        )
        + "\n"
    )
    episode_stats = pd.DataFrame(
        {
            "episode_index": [0],
            "length": [2],
            "stats/observation.state/mean": [np.zeros(6)],
            "stats/observation.state/count": [np.array([2])],
            "stats/action/mean": [np.zeros(6)],
            "stats/action/count": [np.array([2])],
        }
    )
    episode_stats.to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)


def test_convert_dataset_rewrites_state_action_and_stats(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _write_tiny_dataset(source)

    summary = convert_dataset(
        source_root=source,
        dest_root=dest,
        overwrite=False,
        dest_repo_id="local/converted",
    )

    converted = pd.read_parquet(dest / "data" / "chunk-000" / "file-000.parquet")
    np.testing.assert_allclose(converted["observation.state"].iloc[0], [0, 180, 160, 0, -45, 0])
    np.testing.assert_allclose(converted["action"].iloc[0], [0, 179, 161, 0, -45, 1])

    stats = json.loads((dest / "meta" / "stats.json").read_text())
    np.testing.assert_allclose(stats["observation.state"]["mean"], [0.5, 175, 162.5, 1, -44.5, 1.5])
    np.testing.assert_allclose(stats["observation.state"]["q01"], [0.01, 170.1, 160.05, 0.02, -44.99, 0.03])
    np.testing.assert_allclose(stats["observation.state"]["q99"], [0.99, 179.9, 164.95, 1.98, -44.01, 2.97])
    np.testing.assert_allclose(stats["action"]["mean"], [1, 174, 163.5, 1.5, -44, 2.5])
    np.testing.assert_allclose(stats["action"]["q01"], [0.02, 169.1, 161.05, 0.03, -44.98, 1.03])
    np.testing.assert_allclose(stats["action"]["q99"], [1.98, 178.9, 165.95, 2.97, -43.02, 3.97])

    episode_stats = pd.read_parquet(dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    np.testing.assert_allclose(episode_stats["stats/observation.state/mean"].iloc[0], [0.5, 175, 162.5, 1, -44.5, 1.5])
    np.testing.assert_allclose(episode_stats["stats/action/mean"].iloc[0], [1, 174, 163.5, 1.5, -44, 2.5])

    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["repo_id"] == "local/converted"
    assert info["blupe_joint_convention"]["target"] == "molmoact2_so100_so101_v2_1_policy_degrees"
    assert summary["frames"] == 2
