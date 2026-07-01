#!/usr/bin/env python3
"""HTTP service for live YOLO segmentation ball tracking."""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image


def _choose_device(value: str) -> str:
    import torch

    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mask_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] <= 0:
        return None
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def _box_center(box: list[float] | None) -> tuple[float, float] | None:
    if box is None:
        return None
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _box_from_value(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(box)):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _decode_mask(mask_b64: str, shape: tuple[int, int]) -> np.ndarray | None:
    if "," in mask_b64:
        mask_b64 = mask_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(mask_b64)
    except Exception:
        return None
    mask_u8 = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
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


class YoloBallSession:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        continuity_weight: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = model
        self.device = _choose_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.continuity_weight = float(continuity_weight)
        self._lock = threading.Lock()
        self.model = YOLO(model)

    def track_uploaded(
        self,
        image: Image.Image,
        seed_box: list[float] | None,
        min_area: int,
        max_area: int,
        conf: float | None = None,
    ) -> dict[str, Any]:
        rgb = np.asarray(image.convert("RGB"))
        original_shape = rgb.shape[:2]
        seed_center = _box_center(seed_box)
        with self._lock:
            results = self.model.predict(
                source=rgb,
                imgsz=self.imgsz,
                conf=self.conf if conf is None else float(conf),
                device=self.device,
                verbose=False,
            )
        if not results:
            return {"tracked": False, "error": "no_result"}
        result = results[0]
        if result.masks is None or result.boxes is None:
            return {"tracked": False, "error": "no_masks"}

        masks_np = result.masks.data.detach().cpu().numpy()
        boxes_np = result.boxes.xyxy.detach().cpu().numpy()
        confs_np = result.boxes.conf.detach().cpu().numpy()
        classes_np = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else np.zeros(len(confs_np))

        best_mask: np.ndarray | None = None
        best_score = -float("inf")
        best_raw_conf: float | None = None
        candidates: list[dict[str, Any]] = []
        for idx, raw_mask in enumerate(masks_np):
            if int(classes_np[idx]) != 0:
                continue
            mask = raw_mask > 0.5
            if mask.shape != original_shape:
                mask = cv2.resize(mask.astype(np.uint8) * 255, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            area = int(mask.sum())
            box = _mask_box(mask)
            raw_conf = float(confs_np[idx])
            candidate = {
                "score": raw_conf,
                "area": area,
                "box_xyxy": box,
                "model_box_xyxy": [float(x) for x in boxes_np[idx].tolist()],
            }
            if area < min_area or area > max_area or box is None:
                candidate["rejected"] = "area"
                candidates.append(candidate)
                continue
            score = raw_conf
            center = _mask_center(mask)
            if seed_center is not None and center is not None:
                diag = float(np.hypot(original_shape[1], original_shape[0]))
                dist = float(np.hypot(center[0] - seed_center[0], center[1] - seed_center[1]))
                score -= self.continuity_weight * min(1.0, dist / max(1.0, diag))
                candidate["seed_distance_px"] = dist
            candidate["selection_score"] = score
            candidates.append(candidate)
            if score > best_score:
                best_score = score
                best_mask = mask
                best_raw_conf = raw_conf

        if best_mask is None:
            return {"tracked": False, "error": "no_mask_in_area_range", "candidates": candidates[:10]}
        top = _detection_from_mask(best_mask, best_raw_conf)
        top["selection_score"] = best_score
        top["candidates"] = candidates[:10]
        return top


class Handler(BaseHTTPRequestHandler):
    session: YoloBallSession
    protocol_version = "HTTP/1.1"
    request_log = False

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode()
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
            self._json(
                200,
                {
                    "ok": True,
                    "model": self.session.model_path,
                    "device": self.session.device,
                    "mode": "yolo_seg",
                    "imgsz": self.session.imgsz,
                    "conf": self.session.conf,
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
            shape = (image.height, image.width)
            seed_box = _box_from_value(payload.get("box_xyxy"))
            mask_b64 = payload.get("mask_png_b64")
            seed_mask = _decode_mask(str(mask_b64), shape) if mask_b64 else None
            seed_mask_box = _mask_box(seed_mask) if seed_mask is not None else None
            if seed_mask_box is not None:
                seed_box = [float(x) for x in seed_mask_box]
            top = self.session.track_uploaded(
                image=image,
                seed_box=seed_box,
                min_area=max(1, int(payload.get("min_area", 1))),
                max_area=max(1, int(payload.get("max_area", image.width * image.height))),
                conf=None if payload.get("conf") is None else float(payload.get("conf")),
            )
            response = {
                "tracked": bool(top.get("tracked")),
                "mode": "yolo_seg",
                "elapsed_s": round(time.monotonic() - started, 4),
                "seed_box_xyxy": seed_box,
                "top_mask": top,
            }
            response.update(top)
            self._json(200, response)
        except Exception as exc:
            self._json(400, {"tracked": False, "mode": "yolo_seg", "error": str(exc)})

    def log_message(self, fmt, *args):
        if self.request_log:
            print(f"[yolo-ball-track] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve YOLO ball segmentation over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8214)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--continuity-weight", type=float, default=0.25)
    parser.add_argument("--request-log", action="store_true")
    args = parser.parse_args()

    Handler.session = YoloBallSession(
        model=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        continuity_weight=args.continuity_weight,
    )
    Handler.request_log = bool(args.request_log)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[yolo-ball-track] serving http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
