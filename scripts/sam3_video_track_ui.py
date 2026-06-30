#!/usr/bin/env python3
"""HTTP service for SAM3 Video text-prompt ball tracking.

The endpoint is intentionally compatible with ``sam2_track_ui.py``:
``POST /api/track_image`` receives one image and returns ``top_mask`` with a
PNG mask. That lets the SO101 eval UI switch trackers by changing
``SO101_SUCCESS_BALL_SAM2_URL`` while the implementation underneath is native
SAM3 Video tracking.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

DEFAULT_MODEL_ID = "facebook/sam3"
DEFAULT_PROMPT = "blue rubber ball"


def _choose_device(value: str):
    import torch

    if value != "auto":
        device = torch.device(value)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("requested --device mps, but torch.backends.mps.is_available() is false")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("requested --device cuda, but torch.cuda.is_available() is false")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        if getattr(value, "dtype", None) is not None and str(value.dtype).endswith("bfloat16"):
            value = value.float()
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _decode_image(payload: dict[str, Any]) -> Image.Image:
    image_b64 = payload.get("image_b64") or payload.get("image_jpeg_b64") or payload.get("image")
    if not image_b64:
        raise ValueError("image_b64 is required")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _encode_mask(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise ValueError("could not encode mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _mask_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _detection_from_mask(mask: np.ndarray, score: float | None = None) -> dict[str, Any]:
    box = _mask_box(mask)
    area = int(mask.sum())
    return {
        "tracked": box is not None and area > 0,
        "score": None if score is None else float(score),
        "area": area,
        "box_xyxy": box,
        "mask_png_b64": _encode_mask(mask) if box is not None and area > 0 else None,
    }


class Sam3VideoLiveSession:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        prompt: str = DEFAULT_PROMPT,
        device: str = "auto",
        max_session_frames: int = 900,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.prompt = prompt
        self.device = _choose_device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.max_session_frames = max(2, int(max_session_frames))
        self._lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if not hasattr(self.processor, "init_video_session"):
            raise SystemExit(f"{self.model_id} did not load a SAM3 Video processor")
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self.session = None
        self.session_id = ""
        self.frame_idx = -1
        self.frame_shape: tuple[int, int] | None = None
        self.last_source = ""
        self._reset_session("")

    def _reset_session(self, session_id: str) -> None:
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            inference_state_device=self.device,
            processing_device=self.device,
            video_storage_device=self.device,
            dtype=self.dtype,
        )
        self.session = self.processor.add_text_prompt(self.session, self.prompt)
        self.session_id = session_id
        self.frame_idx = -1
        self.frame_shape = None
        self.last_source = "reset"

    def _track_locked(self, image: Image.Image, session_id: str, reset: bool) -> dict[str, Any]:
        import torch

        session_id = session_id or "default"
        if (
            reset
            or self.session is None
            or self.session_id != session_id
            or self.frame_idx + 1 >= self.max_session_frames
        ):
            self._reset_session(session_id)

        shape = (image.height, image.width)
        if self.frame_shape is not None and self.frame_shape != shape:
            self._reset_session(session_id)
        self.frame_shape = shape

        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(self.session, frame=inputs["pixel_values"][0])
            results = self.processor.postprocess_outputs(
                self.session,
                outputs,
                original_sizes=inputs.get("original_sizes"),
            )

        self.frame_idx += 1
        masks = _to_numpy(results.get("masks", []))
        scores = _to_numpy(results.get("scores", []))
        if masks.ndim == 2:
            masks = masks[None, :, :]
        if len(masks) == 0:
            self.last_source = "no_mask"
            return {
                "tracked": False,
                "error": "SAM3 Video returned no mask",
                "source": "sam3_video",
                "session_id": self.session_id,
                "frame_idx": self.frame_idx,
            }

        best_idx = 0
        if len(scores) >= len(masks):
            best_idx = int(np.nanargmax(scores[: len(masks)]))
        score = None if len(scores) <= best_idx else float(scores[best_idx])
        mask = np.asarray(masks[best_idx]).astype(bool)
        detection = _detection_from_mask(mask, score=score)
        detection.update(
            {
                "source": "sam3_video",
                "session_id": self.session_id,
                "frame_idx": self.frame_idx,
            }
        )
        self.last_source = "track" if detection["tracked"] else "empty"
        return detection

    def track_uploaded(self, image: Image.Image, session_id: str, reset: bool) -> dict[str, Any]:
        started = time.monotonic()
        with self._lock:
            result = self._track_locked(image, session_id, reset)
        result["elapsed_s"] = round(time.monotonic() - started, 4)
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "model": self.model_id,
                "prompt": self.prompt,
                "device": str(self.device),
                "mode": "sam3_video",
                "session_id": self.session_id,
                "frame_idx": self.frame_idx,
                "seeded": self.session is not None,
                "last_source": self.last_source,
            }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    session: Sam3VideoLiveSession

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(200, self.session.status())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/track_image":
            self._json(404, {"error": "not found"})
            return
        started = time.monotonic()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image = _decode_image(payload)
            result = self.session.track_uploaded(
                image=image,
                session_id=str(payload.get("session_id") or "default"),
                reset=bool(payload.get("reset_session", False)),
            )
            response = {
                "tracked": bool(result.get("tracked")),
                "mode": "sam3_video",
                "elapsed_s": round(time.monotonic() - started, 4),
                "top_mask": result,
            }
            response.update(result)
            self._json(200, response)
        except Exception as exc:
            self._json(400, {"tracked": False, "mode": "sam3_video", "error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[sam3-video-track] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SAM3 Video text-prompt tracking over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8216)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-session-frames", type=int, default=900)
    args = parser.parse_args()

    Handler.session = Sam3VideoLiveSession(
        model_id=args.model_id,
        prompt=args.prompt,
        device=args.device,
        max_session_frames=args.max_session_frames,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[sam3-video-track] serving http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
