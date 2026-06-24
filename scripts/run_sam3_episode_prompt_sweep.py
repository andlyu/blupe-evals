from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def _to_numpy(value):
    if hasattr(value, "detach"):
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _overlay(image: Image.Image, masks: np.ndarray) -> Image.Image:
    base = np.array(image.convert("RGB")).astype(np.float32)
    overlay = base.copy()
    colors = np.array(
        [[255, 32, 96], [32, 200, 255], [64, 255, 96], [255, 180, 32]],
        dtype=np.float32,
    )
    for i, mask in enumerate(masks):
        mask = np.squeeze(mask) > 0
        color = colors[i % len(colors)]
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _sample_frames(paths: list[Path], max_frames: int) -> list[Path]:
    if len(paths) <= max_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    return [paths[i] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-frames", type=int, default=15)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _sample_frames(sorted(frames_dir.glob("*.jpg")), args.max_frames)
    if not frames:
        raise SystemExit(f"no jpg frames in {frames_dir}")

    bpe_path = Path("/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    processor = Sam3Processor(model)

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else torch.autocast("cpu", enabled=False)
    )

    results: dict[str, list[dict]] = {prompt: [] for prompt in args.prompt}
    best_prompt = None
    best_score_tuple = (-1, -1.0)

    with torch.inference_mode(), autocast:
        for frame_idx, frame_path in enumerate(frames):
            image = Image.open(frame_path).convert("RGB")
            state = processor.set_image(image)
            for prompt in args.prompt:
                output = processor.set_text_prompt(state=state, prompt=prompt)
                masks = _to_numpy(output["masks"])
                boxes = _to_numpy(output["boxes"])
                scores = _to_numpy(output["scores"])
                top_score = float(scores[0]) if len(scores) else 0.0
                box = boxes[0].tolist() if len(boxes) else None
                center = None
                if box is not None:
                    center = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
                record = {
                    "frame": frame_path.name,
                    "num_masks": int(len(masks)),
                    "top_score": top_score,
                    "top_box_xyxy": box,
                    "top_center_xy": center,
                }
                results[prompt].append(record)

                if len(masks):
                    slug = prompt.replace(" ", "_")
                    if frame_idx in {0, len(frames) // 2, len(frames) - 1}:
                        _overlay(image, masks[:1]).save(
                            out_dir / f"{slug}_{frame_idx:03d}_{frame_path.stem}.jpg"
                        )

    summary = []
    for prompt, records in results.items():
        hits = [r for r in records if r["num_masks"] > 0]
        scores = [r["top_score"] for r in hits]
        item = {
            "prompt": prompt,
            "frames": len(records),
            "hits": len(hits),
            "hit_rate": len(hits) / len(records),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
        }
        score_tuple = (item["hits"], item["mean_score"])
        if score_tuple > best_score_tuple:
            best_prompt = prompt
            best_score_tuple = score_tuple
        summary.append(item)

    payload = {
        "frames_dir": str(frames_dir),
        "sampled_frames": [p.name for p in frames],
        "summary": summary,
        "best_prompt": best_prompt,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    shutil.copy2(frames[0], out_dir / "first_frame.jpg")
    print(json.dumps({"summary": summary, "best_prompt": best_prompt}, indent=2))


if __name__ == "__main__":
    main()
