#!/usr/bin/env python3
"""HTTP service for SAM3 Tracker Video mask-prompt ball tracking.

The endpoint is intentionally compatible with ``sam2_track_ui.py``:
``POST /api/track_image`` receives one image and returns ``top_mask`` with a
PNG mask. That lets the SO101 eval UI switch trackers by changing
``SO101_SUCCESS_BALL_SAM2_URL`` while the implementation underneath is native
SAM3 Tracker Video tracking.
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import threading
import time
import traceback
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

DEFAULT_MODEL_ID = "facebook/sam3"
DEFAULT_PROMPT = "light blue object"


def _cuda_memory_stats() -> dict[str, float]:
    try:
        import torch
    except Exception:
        return {}
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    scale = 1024 * 1024
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / scale, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / scale, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / scale, 1),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / scale, 1),
    }


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _decode_optional_seed_image(payload: dict[str, Any]) -> Image.Image | None:
    image_b64 = (
        payload.get("image_b64")
        or payload.get("image_jpeg_b64")
        or payload.get("seed_image_b64")
        or payload.get("seed_image_jpeg_b64")
        or payload.get("image")
    )
    if not image_b64:
        return None
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _encode_mask(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise ValueError("could not encode mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_mask_b64(mask_b64: str, shape: tuple[int, int]) -> np.ndarray:
    if "," in mask_b64:
        mask_b64 = mask_b64.split(",", 1)[1]
    mask_bytes = base64.b64decode(mask_b64)
    mask_u8 = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask_u8 is None:
        raise ValueError("could not decode mask_png_b64")
    if mask_u8.shape != shape:
        mask_u8 = cv2.resize(mask_u8, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask_u8 > 0
    if not mask.any():
        raise ValueError("mask_png_b64 is empty")
    return mask


def _decode_mask(payload: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    mask_b64 = payload.get("mask_png_b64") or payload.get("seed_mask_png_b64")
    if not mask_b64:
        return None
    return _decode_mask_b64(str(mask_b64), shape)


def _decode_seed_masks(payload: dict[str, Any], shape: tuple[int, int]) -> list[dict[str, Any]]:
    seed_masks: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("seed_masks") or []):
        if not isinstance(item, dict):
            continue
        mask_b64 = item.get("mask_png_b64") or item.get("seed_mask_png_b64")
        if not mask_b64:
            continue
        seed_image = _decode_optional_seed_image(item)
        seed_shape = (seed_image.height, seed_image.width) if seed_image is not None else shape
        source_frame_idx = item.get("source_frame_idx", item.get("frame_idx"))
        if source_frame_idx is not None:
            try:
                source_frame_idx = int(source_frame_idx)
            except (TypeError, ValueError):
                source_frame_idx = None
        seed_masks.append(
            {
                "mask": _decode_mask_b64(str(mask_b64), seed_shape),
                "image": seed_image,
                "seed_slot": item.get("seed_slot", item.get("slot", index + 1)),
                "object_id": int(item.get("object_id", payload.get("object_id", 1))),
                "source_frame_idx": source_frame_idx,
            }
        )

    if seed_masks:
        return seed_masks

    mask = _decode_mask(payload, shape)
    if mask is None:
        return []
    return [
        {
            "mask": mask,
            "image": None,
            "seed_slot": payload.get("seed_slot"),
            "object_id": int(payload.get("object_id", 1)),
        }
    ]


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
        inference_state_device: str = "cpu",
        video_storage_device: str = "cpu",
    ):
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        self.model_id = model_id
        self.prompt = prompt
        self.device = _choose_device(device)
        self.inference_state_device = _choose_device(inference_state_device)
        self.video_storage_device = _choose_device(video_storage_device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.max_session_frames = max(2, int(max_session_frames))
        self._lock = threading.Lock()
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_id)
        self.model = Sam3TrackerVideoModel.from_pretrained(self.model_id).to(self.device).eval()
        self.session = None
        self.session_id = ""
        self.frame_idx = -1
        self.frame_shape: tuple[int, int] | None = None
        self.last_source = ""
        self.last_mask: np.ndarray | None = None
        self.last_score: float | None = None
        self.last_box_xyxy: list[int] | None = None
        self.last_valid_frame_idx: int | None = None
        self.fallback_count = 0
        self.reset_count = 0
        self.last_reset_reason = ""
        self.last_reset_monotonic = 0.0
        self.seed_object_id: int | None = None
        self.seed_slot: int | None = None
        self.seed_count = 0
        self.last_seed_frame_idx: int | None = None
        self.last_seed_area: int | None = None
        self._reset_session("", reason="startup")

    def _reset_session(self, session_id: str, reason: str = "request") -> None:
        self.session = None
        _cleanup_cuda()
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            inference_state_device=self.inference_state_device,
            processing_device=self.device,
            video_storage_device=self.video_storage_device,
            dtype=self.dtype,
        )
        self.session_id = session_id
        self.frame_idx = -1
        self.frame_shape = None
        self.last_mask = None
        self.last_score = None
        self.last_box_xyxy = None
        self.last_valid_frame_idx = None
        self.fallback_count = 0
        self.reset_count += 1
        self.last_reset_reason = reason
        self.last_reset_monotonic = time.monotonic()
        self.seed_object_id = None
        self.seed_slot = None
        self.last_seed_frame_idx = None
        self.last_seed_area = None
        self.last_source = "reset"

    def _apply_seed_mask(self, mask: np.ndarray, frame_idx: int, object_id: int, seed_slot: int | None) -> None:
        if self.session is None:
            raise RuntimeError("SAM3 Tracker Video session is not initialized")
        obj_id_int = int(object_id)
        frame_idx_int = int(frame_idx)
        self.session.video_height = int(mask.shape[0])
        self.session.video_width = int(mask.shape[1])
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=frame_idx_int,
            obj_ids=obj_id_int,
            input_masks=mask.astype(bool),
        )
        self.seed_object_id = obj_id_int
        self.seed_slot = None if seed_slot is None else int(seed_slot)
        self.seed_count += 1
        self.last_seed_frame_idx = frame_idx_int
        self.last_seed_area = int(mask.sum())

    def _remember_detection(self, mask: np.ndarray, score: float | None) -> None:
        mask_bool = np.asarray(mask).astype(bool)
        if not mask_bool.any():
            return
        self.last_mask = mask_bool.copy()
        self.last_score = score
        self.last_box_xyxy = _mask_box(mask_bool)
        self.last_valid_frame_idx = int(self.frame_idx)

    def _track_locked(
        self,
        image: Image.Image,
        session_id: str,
        reset: bool,
        seed_mask: np.ndarray | None = None,
        seed_masks: list[dict[str, Any]] | None = None,
        seed_slot: int | None = None,
        object_id: int = 1,
    ) -> dict[str, Any]:
        import torch

        session_id = session_id or "default"
        decoded_seed_masks = list(seed_masks or [])
        if seed_mask is not None:
            decoded_seed_masks.append(
                {
                    "mask": seed_mask,
                    "seed_slot": seed_slot,
                    "object_id": object_id,
                }
            )
        seeded_this_frame = bool(decoded_seed_masks)

        reset_reason = ""
        if reset:
            reset_reason = "request"
        elif self.session is None:
            reset_reason = "no_session"
        elif self.frame_idx + 1 >= self.max_session_frames:
            if seeded_this_frame:
                reset_reason = "max_session_frames"
            else:
                self.last_source = "reset_required"
                return {
                    "tracked": False,
                    "error": "session frame limit reached; seed_masks required to reset",
                    "source": "sam3_video",
                    "session_id": self.session_id,
                    "frame_idx": self.frame_idx,
                    "session_reset": False,
                    "reset_required": True,
                    "reset_reason": "max_session_frames",
                    "reset_count": self.reset_count,
                    "reset_frame_idx": None,
                    "max_session_frames": self.max_session_frames,
                    "seeded_this_frame": False,
                    "seed_object_id": self.seed_object_id,
                    "seed_slot": self.seed_slot,
                }
        if reset_reason:
            self._reset_session(session_id, reason=reset_reason)
        elif not self.session_id:
            self.session_id = session_id

        shape = (image.height, image.width)
        if self.frame_shape is not None and self.frame_shape != shape:
            self.last_source = "frame_shape_change"
            return {
                "tracked": False,
                "error": "frame shape changed; manual reset_session is required",
                "source": "sam3_video",
                "session_id": self.session_id,
                "frame_idx": self.frame_idx,
                "session_reset": False,
                "reset_reason": None,
                "reset_count": self.reset_count,
                "reset_frame_idx": None,
                "reset_required": False,
                "max_session_frames": self.max_session_frames,
                "seeded_this_frame": False,
                "seed_object_id": self.seed_object_id,
                "seed_slot": self.seed_slot,
            }
        self.frame_shape = shape

        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            image_seed_masks = [seed for seed in decoded_seed_masks if seed.get("image") is not None]
            image_seed_masks.sort(
                key=lambda seed: (
                    seed.get("source_frame_idx") is None,
                    int(seed.get("source_frame_idx") or 0),
                    int(seed.get("seed_slot") or 0),
                )
            )
            current_frame_seed_masks = [seed for seed in decoded_seed_masks if seed.get("image") is None]
            for seed in image_seed_masks:
                seed_image = seed["image"]
                if (seed_image.height, seed_image.width) != shape:
                    self.last_source = "seed_frame_shape_change"
                    return {
                        "tracked": False,
                        "error": "seed frame shape differs from current frame",
                        "source": "sam3_video",
                        "session_id": self.session_id,
                        "frame_idx": self.frame_idx,
                        "session_reset": bool(reset_reason),
                        "reset_reason": reset_reason or None,
                        "reset_count": self.reset_count,
                        "reset_frame_idx": None,
                        "reset_required": False,
                        "max_session_frames": self.max_session_frames,
                        "seeded_this_frame": True,
                        "seed_object_id": self.seed_object_id,
                        "seed_slot": self.seed_slot,
                    }
                seed_inputs = self.processor(images=seed_image, return_tensors="pt").to(self.device)
                seed_frame_idx = self.session.add_new_frame(seed_inputs["pixel_values"][0])
                self._apply_seed_mask(
                    np.asarray(seed["mask"]).astype(bool),
                    seed_frame_idx,
                    object_id=int(seed.get("object_id", object_id)),
                    seed_slot=None if seed.get("seed_slot") in (None, "") else int(seed.get("seed_slot")),
                )
                self.model(self.session, frame_idx=seed_frame_idx)

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            if current_frame_seed_masks:
                model_frame_idx = self.session.add_new_frame(inputs["pixel_values"][0])
                for seed in current_frame_seed_masks:
                    self._apply_seed_mask(
                        np.asarray(seed["mask"]).astype(bool),
                        model_frame_idx,
                        object_id=int(seed.get("object_id", object_id)),
                        seed_slot=None if seed.get("seed_slot") in (None, "") else int(seed.get("seed_slot")),
                    )
                outputs = self.model(self.session, frame_idx=model_frame_idx)
            else:
                outputs = self.model(self.session, frame=inputs["pixel_values"][0])

        self.frame_idx = int(getattr(outputs, "frame_idx", self.frame_idx + 1))
        pred_masks = getattr(outputs, "pred_masks", None)
        if pred_masks is None:
            masks = np.asarray([])
        else:
            processed_masks = self.processor.post_process_masks(
                [pred_masks],
                original_sizes=[[image.height, image.width]],
                binarize=True,
            )[0]
            masks = _to_numpy(processed_masks)
        scores = np.asarray([])
        score_logits = getattr(outputs, "object_score_logits", None)
        if score_logits is not None:
            scores = np.ravel(_to_numpy(torch.sigmoid(score_logits)))
        object_ids = np.ravel(np.asarray(getattr(outputs, "object_ids", []) or []))
        while masks.ndim > 3 and masks.shape[0] == 1:
            masks = masks[0]
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
                "session_reset": bool(reset_reason),
                "reset_reason": reset_reason or None,
                "reset_count": self.reset_count,
                "reset_frame_idx": self.frame_idx if reset_reason else None,
                "reset_required": False,
                "max_session_frames": self.max_session_frames,
                "seeded_this_frame": bool(seeded_this_frame),
                "seed_frame_count": len(image_seed_masks),
                "seed_object_id": self.seed_object_id,
                "seed_slot": self.seed_slot,
            }

        best_idx = 0
        preferred_object_id = self.seed_object_id
        preferred_found = False
        if preferred_object_id is not None and len(object_ids) >= len(masks):
            matches = np.where(object_ids[: len(masks)].astype(int) == int(preferred_object_id))[0]
            if len(matches) > 0:
                best_idx = int(matches[0])
                preferred_found = True
        if not preferred_found and len(scores) >= len(masks):
            best_idx = int(np.nanargmax(scores[: len(masks)]))
        score = None if len(scores) <= best_idx else float(scores[best_idx])
        mask = np.asarray(masks[best_idx]).astype(bool)
        selected_object_id = None
        if len(object_ids) > best_idx:
            selected_object_id = int(object_ids[best_idx])
        detection = _detection_from_mask(mask, score=score)
        detection.update(
            {
                "source": "sam3_video",
                "session_id": self.session_id,
                "frame_idx": self.frame_idx,
                "object_id": selected_object_id,
                "session_reset": bool(reset_reason),
                "reset_reason": reset_reason or None,
                "reset_count": self.reset_count,
                "reset_frame_idx": self.frame_idx if reset_reason else None,
                "reset_required": False,
                "max_session_frames": self.max_session_frames,
                "seeded_this_frame": bool(seeded_this_frame),
                "seed_frame_count": len(image_seed_masks),
                "seed_object_id": self.seed_object_id,
                "seed_slot": self.seed_slot,
                "seed_count": self.seed_count,
                "last_seed_frame_idx": self.last_seed_frame_idx,
                "last_seed_area": self.last_seed_area,
            }
        )
        if detection["tracked"]:
            self._remember_detection(mask, score)
        self.last_source = "track" if detection["tracked"] else "empty"
        return detection

    def track_uploaded(
        self,
        image: Image.Image,
        session_id: str,
        reset: bool,
        seed_mask: np.ndarray | None = None,
        seed_masks: list[dict[str, Any]] | None = None,
        seed_slot: int | None = None,
        object_id: int = 1,
    ) -> dict[str, Any]:
        started = time.monotonic()
        memory_before = _cuda_memory_stats()
        with self._lock:
            result = self._track_locked(
                image=image,
                session_id=session_id,
                reset=reset,
                seed_mask=seed_mask,
                seed_masks=seed_masks,
                seed_slot=seed_slot,
                object_id=object_id,
            )
        result["elapsed_s"] = round(time.monotonic() - started, 4)
        result["cuda_memory_before"] = memory_before
        result["cuda_memory_after"] = _cuda_memory_stats()
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "model": self.model_id,
                "prompt": self.prompt,
                "device": str(self.device),
                "inference_state_device": str(self.inference_state_device),
                "video_storage_device": str(self.video_storage_device),
                "mode": "sam3_video",
                "session_id": self.session_id,
                "frame_idx": self.frame_idx,
                "seeded": self.session is not None,
                "last_source": self.last_source,
                "last_valid_frame_idx": self.last_valid_frame_idx,
                "fallback_count": self.fallback_count,
                "reset_count": self.reset_count,
                "last_reset_reason": self.last_reset_reason,
                "frames_since_reset": None if self.frame_idx < 0 else self.frame_idx,
                "max_session_frames": self.max_session_frames,
                "frames_until_reset": None if self.frame_idx < 0 else max(0, self.max_session_frames - self.frame_idx - 1),
                "seed_object_id": self.seed_object_id,
                "seed_slot": self.seed_slot,
                "seed_count": self.seed_count,
                "last_seed_frame_idx": self.last_seed_frame_idx,
                "last_seed_area": self.last_seed_area,
                "cuda_memory": _cuda_memory_stats(),
            }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    session: Sam3VideoLiveSession
    request_log = False

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if self.headers.get("Connection", "").lower() == "close":
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
            self.close_connection = False
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
            seed_masks = _decode_seed_masks(payload, (image.height, image.width))
            seed_slot = payload.get("seed_slot")
            seed_slot_int = None if seed_slot in (None, "") else int(seed_slot)
            object_id = int(payload.get("object_id", 1))
            result = self.session.track_uploaded(
                image=image,
                session_id=str(payload.get("session_id") or "default"),
                reset=bool(payload.get("reset_session", False)),
                seed_masks=seed_masks,
                seed_slot=seed_slot_int,
                object_id=object_id,
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
            _cleanup_cuda()
            traceback.print_exc()
            self._json(
                400,
                {
                    "tracked": False,
                    "mode": "sam3_video",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "cuda_memory": _cuda_memory_stats(),
                },
            )

    def log_message(self, fmt, *args):
        if self.request_log:
            print(f"[sam3-video-track] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SAM3 Video text-prompt tracking over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8216)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-state-device", default="cpu")
    parser.add_argument("--video-storage-device", default="cpu")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-session-frames", type=int, default=900)
    parser.add_argument("--request-log", action="store_true")
    args = parser.parse_args()

    Handler.session = Sam3VideoLiveSession(
        model_id=args.model_id,
        prompt=args.prompt,
        device=args.device,
        max_session_frames=args.max_session_frames,
        inference_state_device=args.inference_state_device,
        video_storage_device=args.video_storage_device,
    )
    Handler.request_log = bool(args.request_log)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[sam3-video-track] serving http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
