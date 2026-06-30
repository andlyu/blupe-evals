import base64
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _load_sam2_video_track_ui():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sam2_video_track_ui.py"
    spec = importlib.util.spec_from_file_location("sam2_video_track_ui", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_video_tracker_decode_mask_prefers_mask_box() -> None:
    sam2_video_track_ui = _load_sam2_video_track_ui()

    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[4:9, 6:12] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok

    decoded = sam2_video_track_ui._decode_mask(
        base64.b64encode(encoded.tobytes()).decode("ascii"),
        shape=(12, 16),
    )

    assert decoded is not None
    assert sam2_video_track_ui._mask_box(decoded) == [6, 4, 11, 8]


def test_video_tracker_detection_payload_contains_encoded_mask() -> None:
    sam2_video_track_ui = _load_sam2_video_track_ui()

    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:6] = True
    detection = sam2_video_track_ui._detection_from_mask(mask)

    assert detection["tracked"] is True
    assert detection["area"] == 9
    assert detection["box_xyxy"] == [3, 2, 5, 4]
    assert detection["mask_png_b64"]


def test_video_tracker_defaults_to_base_plus_model() -> None:
    sam2_video_track_ui = _load_sam2_video_track_ui()

    assert sam2_video_track_ui.DEFAULT_MODEL_ID == "facebook/sam2.1-hiera-base-plus"


def test_video_tracker_appends_live_frames_without_temp_jpeg_roundtrip() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "sam2_video_track_ui.py").read_text()

    assert "live-frame-" in source
    assert "latest.jpg" not in source
    assert "_load_img_as_tensor" not in source
