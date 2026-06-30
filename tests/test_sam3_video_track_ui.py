import importlib.util
from pathlib import Path

import numpy as np


def _load_sam3_video_track_ui():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sam3_video_track_ui.py"
    spec = importlib.util.spec_from_file_location("sam3_video_track_ui", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detection_from_mask_matches_sam2_tracker_contract() -> None:
    sam3_video_track_ui = _load_sam3_video_track_ui()

    mask = np.zeros((8, 9), dtype=bool)
    mask[2:5, 3:7] = True
    detection = sam3_video_track_ui._detection_from_mask(mask, score=0.72)

    assert detection["tracked"] is True
    assert detection["score"] == 0.72
    assert detection["area"] == 12
    assert detection["box_xyxy"] == [3, 2, 6, 4]
    assert detection["mask_png_b64"]


def test_service_declares_sam3_video_mode_and_track_endpoint() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "sam3_video_track_ui.py").read_text()

    assert 'parsed.path != "/api/track_image"' in source
    assert '"mode": "sam3_video"' in source
    assert "init_video_session" in source
    assert "add_text_prompt" in source
