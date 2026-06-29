#!/usr/bin/env python3
"""Small HTTP service for live SAM2 single-object tracking.

The eval UI uses SAM3 to seed the first ball mask. This service then receives
each new frame plus the previous ball mask/box and runs SAM2 on the current
frame using that box as the prompt.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

DEFAULT_MODEL_ID = "facebook/sam2-hiera-tiny"
DEFAULT_RESIZE_MAX_SIDE = 384


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


def _mask_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _box_from_value(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(box)):
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    return box


def _expand_box(box: list[float], width: int, height: int, pad_px: float) -> list[float]:
    x0, y0, x1, y1 = box
    pad = max(0.0, float(pad_px))
    return [
        max(0.0, x0 - pad),
        max(0.0, y0 - pad),
        min(float(width - 1), x1 + pad),
        min(float(height - 1), y1 + pad),
    ]


def _resize_image_and_box(
    image: Image.Image,
    box: list[float],
    resize_max_side: int,
) -> tuple[Image.Image, list[float], float]:
    max_side = int(resize_max_side)
    if max_side <= 0:
        return image, box, 1.0
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image, box, 1.0
    scale = max_side / float(current_max)
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.BILINEAR,
    )
    scaled_box = [float(x) * scale for x in box]
    return resized, scaled_box, scale


def _resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    mask_u8 = cv2.resize(mask.astype(np.uint8) * 255, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_u8 > 0


def _decode_mask(mask_b64: str, shape: tuple[int, int]) -> np.ndarray | None:
    if "," in mask_b64:
        mask_b64 = mask_b64.split(",", 1)[1]
    try:
        mask_bytes = base64.b64decode(mask_b64)
    except Exception:
        return None
    mask_u8 = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask_u8 is None:
        return None
    if mask_u8.shape != shape:
        mask_u8 = cv2.resize(mask_u8, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_u8 > 0


def _encode_mask(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise ValueError("could not encode mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_image(payload: dict[str, Any]) -> Image.Image:
    image_b64 = payload.get("image_b64") or payload.get("image_jpeg_b64") or payload.get("image")
    if not image_b64:
        raise ValueError("image_b64 is required")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _detection_from_mask(mask: np.ndarray, score: float | None) -> dict[str, Any]:
    box = _mask_box(mask)
    area = int(mask.sum())
    return {
        "tracked": box is not None and area > 0,
        "score": None if score is None else float(score),
        "area": area,
        "box_xyxy": box,
        "mask_png_b64": _encode_mask(mask) if box is not None and area > 0 else None,
    }


class Sam2ImageSession:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        checkpoint: str = "",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "auto",
    ):
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.device = _choose_device(device)
        self._lock = threading.Lock()
        self.predictor = self._build_predictor()

    def _build_predictor(self):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if self.checkpoint:
                from sam2.build_sam import build_sam2

                model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
                return SAM2ImagePredictor(model)
            return SAM2ImagePredictor.from_pretrained(self.model_id, device=self.device)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("sam2"):
                raise SystemExit(
                    "sam2 is not installed. Install Meta SAM2 in the vision environment, e.g. "
                    "git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e ."
                ) from exc
            raise

    def track_uploaded(
        self,
        image: Image.Image,
        seed_box: list[float],
        min_area: int,
        max_area: int,
        multimask_output: bool = False,
        resize_max_side: int = DEFAULT_RESIZE_MAX_SIDE,
    ) -> dict[str, Any]:
        import torch

        original_shape = (image.height, image.width)
        inference_image, inference_box, scale = _resize_image_and_box(image.convert("RGB"), seed_box, resize_max_side)
        image_np = np.asarray(inference_image)
        box_np = np.asarray(inference_box, dtype=np.float32)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with self._lock, torch.inference_mode(), autocast:
            self.predictor.set_image(image_np)
            masks, scores, _ = self.predictor.predict(
                box=box_np,
                multimask_output=multimask_output,
            )

        masks_np = np.asarray(masks)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, :, :]
        scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores_np.size < masks_np.shape[0]:
            scores_np = np.pad(scores_np, (0, masks_np.shape[0] - scores_np.size), constant_values=np.nan)

        best_mask = None
        best_score = -float("inf")
        best_area = 0
        for idx, raw_mask in enumerate(masks_np):
            mask = _resize_mask_to_shape(raw_mask.astype(bool), original_shape)
            area = int(mask.sum())
            score = float(scores_np[idx]) if idx < scores_np.size and np.isfinite(scores_np[idx]) else 0.0
            if area < min_area or area > max_area:
                continue
            if score > best_score:
                best_mask = mask
                best_score = score
                best_area = area

        if best_mask is None:
            return {
                "tracked": False,
                "error": f"no_mask_in_area_range min={min_area} max={max_area}",
                "resize_scale": scale,
                "candidates": [
                    {
                        "score": float(scores_np[i]) if i < scores_np.size and np.isfinite(scores_np[i]) else None,
                        "area": int(_resize_mask_to_shape(raw.astype(bool), original_shape).sum()),
                        "box_xyxy": _mask_box(_resize_mask_to_shape(raw.astype(bool), original_shape)),
                    }
                    for i, raw in enumerate(masks_np)
                ],
            }

        top = _detection_from_mask(best_mask, best_score)
        top["area"] = best_area
        top["resize_scale"] = scale
        top["inference_size"] = [int(inference_image.width), int(inference_image.height)]
        return top


class Handler(BaseHTTPRequestHandler):
    session: Sam2ImageSession

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "model": self.session.checkpoint or self.session.model_id,
                    "device": str(self.session.device),
                },
            )
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
            payload = json.loads(self.rfile.read(length).decode())
            image = _decode_image(payload)
            image_np_shape = (image.height, image.width)

            seed_box = _box_from_value(payload.get("box_xyxy"))
            mask_b64 = payload.get("mask_png_b64")
            mask = _decode_mask(str(mask_b64), image_np_shape) if mask_b64 else None
            mask_box = _mask_box(mask) if mask is not None else None
            if mask_box is not None:
                seed_box = [float(x) for x in mask_box]
            if seed_box is None:
                raise ValueError("box_xyxy or mask_png_b64 is required")

            seed_box = _expand_box(
                seed_box,
                width=image.width,
                height=image.height,
                pad_px=float(payload.get("box_pad_px", 8)),
            )
            top = self.session.track_uploaded(
                image=image,
                seed_box=seed_box,
                min_area=max(1, int(payload.get("min_area", 1))),
                max_area=max(1, int(payload.get("max_area", image.width * image.height))),
                multimask_output=bool(payload.get("multimask_output", False)),
                resize_max_side=int(payload.get("resize_max_side", DEFAULT_RESIZE_MAX_SIDE)),
            )
            response = {
                "tracked": bool(top.get("tracked")),
                "elapsed_s": round(time.monotonic() - started, 4),
                "seed_box_xyxy": seed_box,
                "prompt_source": "mask_box" if mask_box is not None else "box",
                "top_mask": top,
            }
            response.update(top)
            self._json(200, response)
        except Exception as exc:
            self._json(400, {"tracked": False, "error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[sam2-track] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SAM2 live image tracking over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8214)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()

    Handler.session = Sam2ImageSession(
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[sam2-track] serving http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
