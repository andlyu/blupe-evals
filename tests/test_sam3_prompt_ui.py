import importlib.util
import threading
from pathlib import Path

import numpy as np
from PIL import Image


def _load_sam3_prompt_ui():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sam3_prompt_ui.py"
    spec = importlib.util.spec_from_file_location("sam3_prompt_ui", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_image_returns_mask_only_when_requested():
    sam3_prompt_ui = _load_sam3_prompt_ui()

    class FakeSession(sam3_prompt_ui.Sam3Session):
        def __init__(self):
            self._lock = threading.Lock()

        def _processor_ready(self):
            return object()

        def _prompt(self, image, prompt, max_masks, include_masks, min_score):
            del image, max_masks, min_score
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:5, 3:7] = True
            detection = sam3_prompt_ui._detection_from_mask(
                mask,
                [3, 2, 6, 4],
                0.88,
                include_mask=include_masks,
            )
            return [detection], 1

    session = FakeSession()
    image = Image.new("RGB", (8, 8), "black")

    preview = session._detect_image(
        image=image,
        frame_label="test.jpg",
        frame_idx=0,
        prompts=["cardboard cylinder"],
        max_masks=1,
        alpha=0.65,
        include_masks=False,
    )
    live = session._detect_image(
        image=image,
        frame_label="uploaded",
        frame_idx=None,
        prompts=["cardboard cylinder"],
        max_masks=1,
        alpha=0.65,
        include_masks=True,
        min_score=0.15,
    )

    assert preview["top_mask"]["score"] == 0.88
    assert "mask_png_b64" not in preview["top_mask"]
    assert "mask_png_b64" not in preview["results"][0]["detections"][0]
    assert live["top_mask"]["score"] == 0.88
    assert "mask_png_b64" in live["top_mask"]
    assert "mask_png_b64" in live["results"][0]["detections"][0]


def test_sam3_service_exposes_health_route() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "sam3_prompt_ui.py").read_text()

    assert 'parsed.path == "/health"' in source
    assert '"ok": True' in source
    assert '"active_backend": self.session._active_backend' in source


def test_sam3_transformers_backend_uses_auto_classes() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "sam3_prompt_ui.py").read_text()

    assert "from transformers import AutoModel, AutoProcessor" in source
    assert "AutoProcessor.from_pretrained(self.model_id)" in source
    assert "AutoModel.from_pretrained(self.model_id)" in source


def test_sam3_service_logs_request_errors() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "sam3_prompt_ui.py").read_text()

    assert "[sam3-ui] request error:" in source
    assert 'self._json(400, {"error": str(exc)})' in source
