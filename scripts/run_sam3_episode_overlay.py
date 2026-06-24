from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def _to_numpy(value):
    if hasattr(value, "detach"):
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _find_bpe_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        Path("/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("could not find SAM3 bpe_simple_vocab_16e6.txt.gz")


def _overlay_top_detection(
    image: Image.Image,
    mask: np.ndarray | None,
    box: list[float] | None,
    score: float,
    prompt: str,
    memory_age: int = 0,
) -> Image.Image:
    base = np.array(image.convert("RGB")).astype(np.float32)
    overlay = base.copy()
    if mask is not None:
        mask_bool = np.squeeze(mask) > 0
        color = np.array([32, 170, 255], dtype=np.float32)
        overlay[mask_bool] = 0.35 * overlay[mask_bool] + 0.65 * color

    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    if box is not None:
        x0, y0, x1, y1 = box
        color = (0, 210, 255) if memory_age == 0 else (255, 190, 0)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"{prompt} {score:.2f}"
        if memory_age:
            label += f" held {memory_age}"
        font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = right - left
        text_h = bottom - top
        label_y = max(0, y0 - text_h - 6)
        draw.rectangle((x0, label_y, x0 + text_w + 8, label_y + text_h + 6), fill=(0, 0, 0))
        draw.text((x0 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", default="blue ball")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process every frame")
    parser.add_argument(
        "--memory-frames",
        type=int,
        default=0,
        help="Reuse the last good detection for this many missed frames.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Only detections at or above this score update memory.",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlay_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("*.jpg"))
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise SystemExit(f"no jpg frames in {frames_dir}")

    model = build_sam3_image_model(bpe_path=str(_find_bpe_path()))
    processor = Sam3Processor(model)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else torch.autocast("cpu", enabled=False)
    )

    detections = []
    last_mask = None
    last_box = None
    last_score = 0.0
    last_center = None
    missed_since_hit = 0
    with torch.inference_mode(), autocast:
        for idx, frame_path in enumerate(frames):
            image = Image.open(frame_path).convert("RGB")
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=args.prompt)
            masks = _to_numpy(output["masks"])
            boxes = _to_numpy(output["boxes"])
            scores = _to_numpy(output["scores"])

            raw_score = float(scores[0]) if len(scores) else 0.0
            raw_box = boxes[0].tolist() if len(boxes) else None
            center = None
            if raw_box is not None:
                center = [(raw_box[0] + raw_box[2]) / 2.0, (raw_box[1] + raw_box[3]) / 2.0]
            raw_mask = masks[0] if len(masks) else None

            has_fresh_detection = raw_mask is not None and raw_box is not None and raw_score >= args.min_score
            if has_fresh_detection:
                last_mask = raw_mask.copy()
                last_box = list(raw_box)
                last_score = raw_score
                last_center = list(center) if center is not None else None
                missed_since_hit = 0
                display_mask = raw_mask
                display_box = raw_box
                display_score = raw_score
                display_center = center
                display_source = "sam3"
                memory_age = 0
            else:
                missed_since_hit += 1
                if last_box is not None and missed_since_hit <= args.memory_frames:
                    display_mask = last_mask
                    display_box = last_box
                    display_score = last_score
                    display_center = last_center
                    display_source = "memory"
                    memory_age = missed_since_hit
                else:
                    display_mask = None
                    display_box = None
                    display_score = 0.0
                    display_center = None
                    display_source = "none"
                    memory_age = 0

            _overlay_top_detection(
                image,
                display_mask,
                display_box,
                display_score,
                args.prompt,
                memory_age=memory_age,
            ).save(
                overlay_dir / frame_path.name
            )
            detections.append(
                {
                    "frame": frame_path.name,
                    "num_masks": int(len(masks)),
                    "top_score": raw_score,
                    "top_box_xyxy": raw_box,
                    "top_center_xy": center,
                    "display_source": display_source,
                    "display_score": display_score,
                    "display_box_xyxy": display_box,
                    "display_center_xy": display_center,
                    "memory_age": memory_age,
                }
            )
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "frame": frame_path.name,
                        "num_masks": int(len(masks)),
                        "top_score": round(raw_score, 4),
                        "display_source": display_source,
                        "memory_age": memory_age,
                    }
                ),
                flush=True,
            )

    hits = [d for d in detections if d["num_masks"] > 0]
    scores = [d["top_score"] for d in hits]
    memory_filled = [d for d in detections if d["display_source"] == "memory"]
    displayed = [d for d in detections if d["display_source"] != "none"]
    summary = {
        "frames_dir": str(frames_dir),
        "prompt": args.prompt,
        "memory_frames": args.memory_frames,
        "score_threshold": args.min_score,
        "frames": len(detections),
        "hits": len(hits),
        "hit_rate": len(hits) / len(detections),
        "memory_filled": len(memory_filled),
        "displayed": len(displayed),
        "display_rate": len(displayed) / len(detections),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "min_detection_score": float(np.min(scores)) if scores else 0.0,
        "overlay_frames_dir": str(overlay_dir),
        "detections": detections,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "detections"}, indent=2))


if __name__ == "__main__":
    main()
