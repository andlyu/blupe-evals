from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    bpe_path = Path(__file__).resolve().parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        bpe_path = Path("/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    processor = Sam3Processor(model)

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else torch.autocast("cpu", enabled=False)
    )
    with torch.inference_mode(), autocast:
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks = _to_numpy(output["masks"])
    boxes = _to_numpy(output["boxes"])
    scores = _to_numpy(output["scores"])

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "image": args.image,
        "prompt": args.prompt,
        "num_masks": int(len(masks)),
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    if len(masks) > 0:
        base = np.array(image).astype(np.float32)
        overlay = base.copy()
        colors = np.array(
            [
                [255, 32, 96],
                [32, 200, 255],
                [64, 255, 96],
                [255, 180, 32],
                [180, 64, 255],
            ],
            dtype=np.float32,
        )
        for i, mask in enumerate(masks):
            mask = np.squeeze(mask) > 0
            color = colors[i % len(colors)]
            overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
            prefix.with_suffix(".overlay.jpg")
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
