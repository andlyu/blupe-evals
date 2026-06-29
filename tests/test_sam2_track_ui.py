import base64
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _load_sam2_track_ui():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sam2_track_ui.py"
    spec = importlib.util.spec_from_file_location("sam2_track_ui", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_mask_prefers_mask_box_over_request_box() -> None:
    sam2_track_ui = _load_sam2_track_ui()

    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:8, 5:11] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok

    decoded = sam2_track_ui._decode_mask(
        base64.b64encode(encoded.tobytes()).decode("ascii"),
        shape=(12, 16),
    )

    assert decoded is not None
    assert sam2_track_ui._mask_box(decoded) == [5, 3, 10, 7]


def test_expand_box_clips_to_image_bounds() -> None:
    sam2_track_ui = _load_sam2_track_ui()

    assert sam2_track_ui._expand_box([2, 3, 7, 9], width=10, height=12, pad_px=5) == [
        0.0,
        0.0,
        9.0,
        11.0,
    ]


def test_detection_from_mask_contains_encoded_mask() -> None:
    sam2_track_ui = _load_sam2_track_ui()

    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:6] = True
    detection = sam2_track_ui._detection_from_mask(mask, score=0.5)

    assert detection["tracked"] is True
    assert detection["area"] == 9
    assert detection["box_xyxy"] == [3, 2, 5, 4]
    assert detection["mask_png_b64"]


def test_fast_defaults_use_tiny_model_and_downscale() -> None:
    sam2_track_ui = _load_sam2_track_ui()

    assert sam2_track_ui.DEFAULT_MODEL_ID == "facebook/sam2-hiera-tiny"
    assert sam2_track_ui.DEFAULT_RESIZE_MAX_SIDE == 384


def test_resize_image_and_box_scales_and_clips_mask_back() -> None:
    sam2_track_ui = _load_sam2_track_ui()
    from PIL import Image

    image = Image.new("RGB", (640, 360), "black")
    resized, box, scale = sam2_track_ui._resize_image_and_box(
        image,
        [100, 50, 200, 150],
        resize_max_side=320,
    )

    assert resized.size == (320, 180)
    assert scale == 0.5
    assert box == [50.0, 25.0, 100.0, 75.0]

    small_mask = np.zeros((180, 320), dtype=bool)
    small_mask[25:75, 50:100] = True
    full_mask = sam2_track_ui._resize_mask_to_shape(small_mask, (360, 640))
    assert full_mask.shape == (360, 640)
    assert sam2_track_ui._mask_box(full_mask) == [100, 50, 199, 149]


def test_choose_device_auto_prefers_mps_when_cuda_absent(monkeypatch) -> None:
    sam2_track_ui = _load_sam2_track_ui()
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert sam2_track_ui._choose_device("auto").type == "mps"
