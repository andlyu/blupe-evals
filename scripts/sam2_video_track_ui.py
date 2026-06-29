#!/usr/bin/env python3
"""HTTP service for live-ish SAM2 video tracking.

This service keeps one SAM2 video predictor state alive. The caller sends a
SAM3-seeded ball mask/box on the first request for a session, then subsequent
requests append one frame and propagate SAM2 by one frame. The endpoint shape is
compatible with the older stateless ``sam2_track_ui.py`` so the SO101 eval UI can
switch services through ``SO101_SUCCESS_BALL_SAM2_URL``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import tempfile
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

DEFAULT_MODEL_ID = "facebook/sam2-hiera-tiny"
DEFAULT_OBJECT_ID = 1


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


def _decode_image(payload: dict[str, Any]) -> Image.Image:
    image_b64 = payload.get("image_b64") or payload.get("image_jpeg_b64") or payload.get("image")
    if not image_b64:
        raise ValueError("image_b64 is required")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


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


def _mask_to_numpy(mask_logits: Any) -> np.ndarray:
    mask = (mask_logits > 0.0).detach().cpu().numpy()
    return np.squeeze(mask).astype(bool)


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


class Sam2VideoLiveSession:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        checkpoint: str = "",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "auto",
        max_session_frames: int = 1500,
    ):
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.device = _choose_device(device)
        self.max_session_frames = max(2, int(max_session_frames))
        self._lock = threading.Lock()
        self._tmp_root = Path(tempfile.mkdtemp(prefix="sam2-video-live-"))
        self.predictor = self._build_predictor()
        self.inference_state: dict[str, Any] | None = None
        self.session_id = ""
        self.frame_idx = -1
        self.frame_shape: tuple[int, int] | None = None
        self.last_mask: np.ndarray | None = None
        self.last_source = ""

    def close(self) -> None:
        shutil.rmtree(self._tmp_root, ignore_errors=True)

    def _build_predictor(self):
        try:
            if self.checkpoint:
                from sam2.build_sam import build_sam2_video_predictor

                return build_sam2_video_predictor(
                    self.model_cfg,
                    self.checkpoint,
                    device=self.device,
                )

            from sam2.sam2_video_predictor import SAM2VideoPredictor

            return SAM2VideoPredictor.from_pretrained(self.model_id, device=self.device)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("sam2"):
                raise SystemExit(
                    "sam2 is not installed. Install Meta SAM2 in the vision environment, e.g. "
                    "git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e ."
                ) from exc
            raise

    def _write_seed_frame(self, image: Image.Image) -> Path:
        session_dir = self._tmp_root / "session"
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True)
        image.save(session_dir / "0.jpg", quality=95)
        return session_dir

    def _frame_to_model_tensor(self, image: Image.Image):
        import torch

        video_width, video_height = image.size
        resized = image.convert("RGB").resize(
            (int(self.predictor.image_size), int(self.predictor.image_size)),
            Image.Resampling.BILINEAR,
        )
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.frame_shape is not None and self.frame_shape != (video_height, video_width):
            raise ValueError(
                f"frame shape changed from {self.frame_shape} to {(video_height, video_width)}"
            )
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=tensor.device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=tensor.device)[:, None, None]
        tensor = (tensor - mean) / std
        if self.inference_state is not None and not self.inference_state.get("offload_video_to_cpu", False):
            tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    def _append_frame(self, image: Image.Image) -> int:
        import torch

        if self.inference_state is None:
            raise ValueError("session is not seeded")
        tensor = self._frame_to_model_tensor(image)
        images = self.inference_state["images"]
        if isinstance(images, torch.Tensor):
            self.inference_state["images"] = torch.cat([images, tensor.unsqueeze(0)], dim=0)
        elif hasattr(images, "images"):
            images.images.append(tensor)
            if hasattr(images, "img_paths"):
                images.img_paths.append(f"live-frame-{self.frame_idx + 1}.jpg")
        else:
            images.append(tensor)
        self.inference_state["num_frames"] += 1
        self.frame_idx = int(self.inference_state["num_frames"]) - 1
        return self.frame_idx

    def _extract_mask_for_object(self, obj_ids: Any, mask_logits: Any) -> np.ndarray | None:
        for i, obj_id in enumerate(obj_ids):
            if int(obj_id) == DEFAULT_OBJECT_ID:
                return _mask_to_numpy(mask_logits[i])
        return None

    def _seed_locked(
        self,
        image: Image.Image,
        seed_box: list[float],
        session_id: str,
    ) -> dict[str, Any]:
        import torch

        seed_dir = self._write_seed_frame(image)
        self.inference_state = self.predictor.init_state(video_path=str(seed_dir))
        images = self.inference_state.get("images")
        if hasattr(images, "shape"):
            self.inference_state["images"] = [images[0]]
        self.session_id = session_id
        self.frame_idx = 0
        self.frame_shape = (image.height, image.width)
        self.last_mask = None
        self.last_source = "seed"

        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=DEFAULT_OBJECT_ID,
                box=np.array(seed_box, dtype=np.float32),
            )
        mask = self._extract_mask_for_object(obj_ids, mask_logits)
        if mask is None:
            raise ValueError("SAM2 seed did not return object mask")
        self.last_mask = mask
        return {
            **_detection_from_mask(mask),
            "source": "sam2_video_seed",
            "session_id": self.session_id,
            "frame_idx": self.frame_idx,
        }

    def track_uploaded(self, image: Image.Image, seed_box: list[float], session_id: str, reset: bool) -> dict[str, Any]:
        import torch

        session_id = session_id or "default"
        started = time.monotonic()
        with self._lock:
            if (
                reset
                or self.inference_state is None
                or self.session_id != session_id
                or self.frame_idx + 1 >= self.max_session_frames
            ):
                result = self._seed_locked(image, seed_box, session_id)
                result["elapsed_s"] = round(time.monotonic() - started, 4)
                return result

            frame_idx = self._append_frame(image)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
            mask = None
            with torch.inference_mode(), autocast:
                for out_frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=1,
                    reverse=False,
                ):
                    if int(out_frame_idx) == frame_idx:
                        mask = self._extract_mask_for_object(obj_ids, mask_logits)
            if mask is None:
                return {
                    "tracked": False,
                    "error": "SAM2 video propagation returned no mask",
                    "source": "sam2_video_track",
                    "session_id": self.session_id,
                    "frame_idx": frame_idx,
                    "elapsed_s": round(time.monotonic() - started, 4),
                }
            self.last_mask = mask
            self.last_source = "track"
            result = _detection_from_mask(mask)
            result.update(
                {
                    "source": "sam2_video_track",
                    "session_id": self.session_id,
                    "frame_idx": frame_idx,
                    "elapsed_s": round(time.monotonic() - started, 4),
                }
            )
            return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "model": self.checkpoint or self.model_id,
                "device": str(self.device),
                "mode": "sam2_video",
                "session_id": self.session_id,
                "frame_idx": self.frame_idx,
                "seeded": self.inference_state is not None,
                "last_source": self.last_source,
            }


class Handler(BaseHTTPRequestHandler):
    session: Sam2VideoLiveSession

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
            self._json(200, self.session.status())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/track_image":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            image = _decode_image(payload)
            shape = (image.height, image.width)
            seed_box = _box_from_value(payload.get("box_xyxy"))
            mask_b64 = payload.get("mask_png_b64")
            mask = _decode_mask(str(mask_b64), shape) if mask_b64 else None
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
            result = self.session.track_uploaded(
                image=image,
                seed_box=seed_box,
                session_id=str(payload.get("session_id") or "default"),
                reset=bool(payload.get("reset_session", False)),
            )
            response = {
                "tracked": bool(result.get("tracked")),
                "mode": "sam2_video",
                "seed_box_xyxy": seed_box,
                "prompt_source": "mask_box" if mask_box is not None else "box",
                "top_mask": result,
            }
            response.update(result)
            self._json(200, response)
        except Exception as exc:
            self._json(400, {"tracked": False, "mode": "sam2_video", "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[sam2-video-track] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a stateful SAM2 video tracker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8214)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--max-session-frames", type=int, default=1500)
    args = parser.parse_args()

    session = Sam2VideoLiveSession(
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        max_session_frames=args.max_session_frames,
    )
    Handler.session = session
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[sam2-video-track] serving http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        session.close()


if __name__ == "__main__":
    main()
