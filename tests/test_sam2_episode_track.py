import importlib.util
import json
from pathlib import Path


def _load_sam2_episode_track():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sam2_episode_track.py"
    spec = importlib.util.spec_from_file_location("run_sam2_episode_track", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sam2_tracker_accepts_single_sam3_seed_summary(tmp_path: Path) -> None:
    sam2_track = _load_sam2_episode_track()
    frames = []
    for idx in range(3):
        frame = tmp_path / f"frame_{idx:05d}.jpg"
        frame.write_bytes(b"fake jpg")
        frames.append(frame)

    seed_summary = tmp_path / "sam3_seed_summary.json"
    seed_summary.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "frame": "frame_00001.jpg",
                        "top_score": 0.92,
                        "top_box_xyxy": [100.0, 120.0, 150.0, 170.0],
                    }
                ]
            }
        )
    )

    seed_idx, seed_box, seed_meta = sam2_track._load_seed(
        frames=frames,
        seed_box=None,
        seed_frame=0,
        seed_summary=seed_summary,
        min_seed_score=0.5,
    )

    assert seed_idx == 1
    assert seed_box == [100.0, 120.0, 150.0, 170.0]
    assert seed_meta["source"] == "sam3_summary"
