#!/usr/bin/env python3
"""Benchmark SAM2 ball tracking clips against sampled SAM3 ball masks.

This is intentionally separate from the SO101 eval UI. It talks to the running
SAM3 and SAM2 HTTP services and reports only:
  A) SAM2-vs-SAM3 mask alignment on SAM3 checkpoint frames.
  B) SAM2 inference throughput against a target Hz.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image


DEFAULT_VIDEO_URL = (
    "https://huggingface.co/datasets/andlyu/pick_up_ball_v21_pt2/resolve/main/"
    "videos/observation.images.wrist/chunk-000/file-000.mp4"
)


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("failed to encode frame as JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _mask_to_png_b64(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_from_png_b64(mask_b64: str, shape: tuple[int, int]) -> np.ndarray | None:
    if not mask_b64:
        return None
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


def _mask_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def _download(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        tmp = output.with_suffix(output.suffix + ".partial")
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"download {pct:5.1f}% {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="\r", flush=True)
        tmp.replace(output)
    print()
    return output


def _top_mask(data: dict[str, Any]) -> dict[str, Any]:
    top = data.get("top_mask")
    if isinstance(top, dict):
        return top
    return {}


def _call_sam3(
    *,
    url: str,
    image_b64: str,
    shape: tuple[int, int],
    prompt: str,
    min_score: float,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    resp = requests.post(
        url,
        json={
            "image_b64": image_b64,
            "prompts": [prompt],
            "max_masks": 1,
            "min_score": min_score,
            "alpha": 0.65,
        },
        timeout=timeout_s,
    )
    elapsed_s = time.monotonic() - started
    resp.raise_for_status()
    data = resp.json()
    top = _top_mask(data)
    mask = _mask_from_png_b64(str(top.get("mask_png_b64") or ""), shape)
    box = top.get("box_xyxy")
    if mask is not None and not isinstance(box, list):
        box = _mask_box(mask)
    return {
        "ok": mask is not None and bool(mask.any()),
        "mask": mask,
        "box_xyxy": [float(x) for x in box] if isinstance(box, list) and len(box) == 4 else None,
        "score": None if top.get("score") is None else float(top.get("score")),
        "area": None if mask is None else int(mask.sum()),
        "elapsed_s": elapsed_s,
    }


def _call_sam2(
    *,
    url: str,
    image_b64: str,
    shape: tuple[int, int],
    session_id: str,
    seed_box: list[float],
    reset_session: bool,
    seed_mask: np.ndarray | None,
    resize_max_side: int,
    box_pad_px: float,
    timeout_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_b64": image_b64,
        "box_xyxy": seed_box,
        "session_id": session_id,
        "reset_session": reset_session,
        "min_area": 1,
        "max_area": int(shape[0] * shape[1]),
        "multimask_output": False,
        "resize_max_side": resize_max_side,
        "box_pad_px": box_pad_px,
    }
    if seed_mask is not None:
        payload["mask_png_b64"] = _mask_to_png_b64(seed_mask)

    started = time.monotonic()
    resp = requests.post(url, json=payload, timeout=timeout_s)
    wall_elapsed_s = time.monotonic() - started
    resp.raise_for_status()
    data = resp.json()
    top = data.get("top_mask") if isinstance(data.get("top_mask"), dict) else data
    mask = _mask_from_png_b64(str(top.get("mask_png_b64") or ""), shape)
    box = top.get("box_xyxy") if isinstance(top, dict) else None
    if mask is not None and not isinstance(box, list):
        box = _mask_box(mask)
    return {
        "ok": mask is not None and bool(mask.any()) and bool(data.get("tracked", top.get("tracked", True))),
        "mask": mask,
        "box_xyxy": [float(x) for x in box] if isinstance(box, list) and len(box) == 4 else None,
        "area": None if mask is None else int(mask.sum()),
        "wall_elapsed_s": wall_elapsed_s,
        "service_elapsed_s": None if data.get("elapsed_s") is None else float(data.get("elapsed_s")),
        "mode": data.get("mode"),
        "error": data.get("error") or top.get("error") if isinstance(top, dict) else None,
    }


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    return cap


def run(args: argparse.Namespace) -> dict[str, Any]:
    video_path = Path(args.video) if args.video else _download(args.video_url, Path(args.download_path))
    cap = _open_video(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    session_id = f"sam-ball-bench-{int(time.time())}"
    processed = 0
    source_idx = -1
    last_mask: np.ndarray | None = None
    last_box: list[float] | None = None
    last_anchor_processed_idx: int | None = None

    sam2_wall_times: list[float] = []
    sam2_service_times: list[float] = []
    sam3_times: list[float] = []
    ious: list[float] = []
    alignment_rows: list[dict[str, Any]] = []
    sam2_calls = 0
    sam2_ok = 0
    sam3_calls = 0
    sam3_ok = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        source_idx += 1
        if source_idx < args.start_frame:
            continue
        if (source_idx - args.start_frame) % args.frame_stride != 0:
            continue
        if args.max_frames and processed >= args.max_frames:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        shape = rgb.shape[:2]
        image_b64 = _encode_jpeg_b64(rgb, quality=args.jpeg_quality)

        sam2_result: dict[str, Any] | None = None
        if last_mask is not None and last_box is not None:
            sam2_result = _call_sam2(
                url=args.sam2_url,
                image_b64=image_b64,
                shape=shape,
                session_id=session_id,
                seed_box=last_box,
                reset_session=False,
                seed_mask=None,
                resize_max_side=args.sam2_resize_max_side,
                box_pad_px=args.sam2_box_pad_px,
                timeout_s=args.sam2_timeout_s,
            )
            sam2_calls += 1
            sam2_wall_times.append(float(sam2_result["wall_elapsed_s"]))
            if sam2_result.get("service_elapsed_s") is not None:
                sam2_service_times.append(float(sam2_result["service_elapsed_s"]))
            if sam2_result["ok"]:
                sam2_ok += 1
                last_mask = sam2_result["mask"]
                last_box = sam2_result["box_xyxy"]

        needs_sam3 = last_mask is None or (
            args.sam3_every_n_frames > 0
            and last_anchor_processed_idx is not None
            and processed - last_anchor_processed_idx >= args.sam3_every_n_frames
        )
        if last_anchor_processed_idx is None:
            needs_sam3 = True

        if needs_sam3:
            sam3_result = _call_sam3(
                url=args.sam3_url,
                image_b64=image_b64,
                shape=shape,
                prompt=args.prompt,
                min_score=args.sam3_min_score,
                timeout_s=args.sam3_timeout_s,
            )
            sam3_calls += 1
            sam3_times.append(float(sam3_result["elapsed_s"]))
            if sam3_result["ok"]:
                sam3_ok += 1

            if sam3_result["ok"] and sam2_result is not None and sam2_result.get("ok"):
                iou = _mask_iou(sam2_result["mask"], sam3_result["mask"])
                if iou is not None:
                    ious.append(iou)
                    alignment_rows.append(
                        {
                            "processed_idx": processed,
                            "source_frame_idx": source_idx,
                            "iou": iou,
                            "sam2_area": sam2_result.get("area"),
                            "sam3_area": sam3_result.get("area"),
                            "sam3_score": sam3_result.get("score"),
                        }
                    )

            if sam3_result["ok"] and sam3_result["box_xyxy"] is not None:
                reset_result = _call_sam2(
                    url=args.sam2_url,
                    image_b64=image_b64,
                    shape=shape,
                    session_id=session_id,
                    seed_box=sam3_result["box_xyxy"],
                    reset_session=True,
                    seed_mask=sam3_result["mask"],
                    resize_max_side=args.sam2_resize_max_side,
                    box_pad_px=args.sam2_box_pad_px,
                    timeout_s=args.sam2_timeout_s,
                )
                sam2_calls += 1
                sam2_wall_times.append(float(reset_result["wall_elapsed_s"]))
                if reset_result.get("service_elapsed_s") is not None:
                    sam2_service_times.append(float(reset_result["service_elapsed_s"]))
                if reset_result["ok"]:
                    sam2_ok += 1
                    last_mask = reset_result["mask"]
                    last_box = reset_result["box_xyxy"]
                else:
                    last_mask = sam3_result["mask"]
                    last_box = sam3_result["box_xyxy"]
                last_anchor_processed_idx = processed

        processed += 1

    cap.release()

    sam2_wall_sum = sum(sam2_wall_times)
    sam2_service_sum = sum(sam2_service_times)
    sam2_wall_hz = None if sam2_wall_sum <= 0 else sam2_calls / sam2_wall_sum
    sam2_service_hz = None if sam2_service_sum <= 0 else len(sam2_service_times) / sam2_service_sum
    summary = {
        "video": str(video_path),
        "source_frame_count": frame_count,
        "source_fps": fps,
        "start_frame": args.start_frame,
        "frame_stride": args.frame_stride,
        "processed_frames": processed,
        "sam3_every_n_processed_frames": args.sam3_every_n_frames,
        "alignment_samples": len(ious),
        "alignment_iou_mean": None if not ious else float(statistics.fmean(ious)),
        "alignment_iou_median": None if not ious else float(statistics.median(ious)),
        "alignment_iou_min": None if not ious else float(min(ious)),
        "alignment_iou_p10": _percentile(ious, 10),
        "alignment_iou_p90": _percentile(ious, 90),
        "sam2_calls": sam2_calls,
        "sam2_tracked_calls": sam2_ok,
        "sam2_wall_hz": sam2_wall_hz,
        "sam2_service_hz": sam2_service_hz,
        "speed_target_hz": args.speed_target_hz,
        "speed_pass": bool(sam2_wall_hz is not None and sam2_wall_hz >= args.speed_target_hz),
        "sam3_calls": sam3_calls,
        "sam3_valid_calls": sam3_ok,
        "sam3_mean_elapsed_s": None if not sam3_times else float(statistics.fmean(sam3_times)),
        "sam2_mean_wall_elapsed_s": None if not sam2_wall_times else float(statistics.fmean(sam2_wall_times)),
        "sam2_mean_service_elapsed_s": None if not sam2_service_times else float(statistics.fmean(sam2_service_times)),
        "alignment_rows": alignment_rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SAM2 tracking against sampled SAM3 masks.")
    parser.add_argument("--video", default="", help="Local video path. If omitted, --video-url is downloaded.")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--download-path", default="/tmp/so101-sam-ball-benchmark/file-000.mp4")
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8213/api/detect_image")
    parser.add_argument("--sam2-url", default="http://127.0.0.1:8214/api/track_image")
    parser.add_argument("--prompt", default="blue rubber ball")
    parser.add_argument("--sam3-min-score", type=float, default=0.25)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--sam3-every-n-frames", type=int, default=10)
    parser.add_argument("--speed-target-hz", type=float, default=3.0)
    parser.add_argument("--sam2-resize-max-side", type=int, default=384)
    parser.add_argument("--sam2-box-pad-px", type=float, default=2.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--sam2-timeout-s", type=float, default=10.0)
    parser.add_argument("--sam3-timeout-s", type=float, default=30.0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if args.sam3_every_n_frames < 1:
        raise SystemExit("--sam3-every-n-frames must be >= 1")

    summary = run(args)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
    printable = dict(summary)
    printable.pop("alignment_rows", None)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
