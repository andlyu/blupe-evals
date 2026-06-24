from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _prepare_sam2_video_dir(frames: list[Path], out_dir: Path) -> Path:
    """SAM2's loader expects frame names whose stems are plain integers."""
    video_dir = out_dir / "sam2_numeric_frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    for old in video_dir.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()
    for idx, frame in enumerate(frames):
        target = video_dir / f"{idx}.jpg"
        try:
            target.symlink_to(frame.resolve())
        except OSError:
            import shutil

            shutil.copy2(frame, target)
    return video_dir


def _parse_box(value: str) -> list[float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("box must satisfy x1>x0 and y1>y0")
    return parts


def _load_seed(
    *,
    frames: list[Path],
    seed_box: list[float] | None,
    seed_frame: int,
    seed_summary: Path | None,
    min_seed_score: float,
) -> tuple[int, list[float], dict]:
    if seed_box is not None:
        if seed_frame < 0 or seed_frame >= len(frames):
            raise SystemExit(f"--seed-frame {seed_frame} outside 0..{len(frames) - 1}")
        return seed_frame, seed_box, {"source": "manual"}

    if seed_summary is None:
        raise SystemExit("provide either --seed-box or --seed-summary")

    summary = json.loads(seed_summary.read_text())
    by_name = {p.name: i for i, p in enumerate(frames)}
    for detection in summary.get("detections", []):
        box = detection.get("top_box_xyxy")
        score = float(detection.get("top_score") or 0.0)
        if not box or score < min_seed_score:
            continue
        frame_name = detection.get("frame")
        if frame_name not in by_name:
            continue
        return (
            by_name[frame_name],
            [float(v) for v in box],
            {
                "source": "sam3_summary",
                "summary": str(seed_summary),
                "frame": frame_name,
                "score": score,
            },
        )

    raise SystemExit(
        f"no seed detection in {seed_summary} at score >= {min_seed_score}; "
        "lower --min-seed-score or pass --seed-box"
    )


def _build_predictor(args, device):
    try:
        if args.checkpoint:
            from sam2.build_sam import build_sam2_video_predictor

            return build_sam2_video_predictor(
                args.model_cfg,
                args.checkpoint,
                device=device,
                vos_optimized=args.vos_optimized,
            )

        from sam2.sam2_video_predictor import SAM2VideoPredictor

        return SAM2VideoPredictor.from_pretrained(
            args.hf_model,
            device=device,
            vos_optimized=args.vos_optimized,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("sam2"):
            raise SystemExit(
                "sam2 is not installed in this Python environment. Install Meta SAM2 in the "
                "environment you use for vision inference, then rerun this script. Example: "
                "git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e ."
            ) from exc
        raise


def _choose_device(value: str):
    import torch

    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mask_to_numpy(mask_logits):
    import numpy as np

    mask = (mask_logits > 0.0).detach().cpu().numpy()
    return np.squeeze(mask).astype(bool)


def _mask_box(mask) -> list[int] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _overlay(
    image,
    mask,
    box: list[int] | None,
    label: str,
    alpha: float,
):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    base = np.array(image.convert("RGB")).astype(np.float32)
    out_arr = base.copy()
    if mask is not None and mask.any():
        color = np.array([32, 170, 255], dtype=np.float32)
        out_arr[mask] = (1.0 - alpha) * out_arr[mask] + alpha * color

    out = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))
    if box is None:
        return out

    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), outline=(0, 210, 255), width=3)
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w = right - left
    text_h = bottom - top
    label_y = max(0, y0 - text_h - 6)
    draw.rectangle((x0, label_y, x0 + text_w + 8, label_y + text_h + 6), fill=(0, 0, 0))
    draw.text((x0 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
    return out


def _write_video(overlay_dir: Path, video_out: Path, fps: float) -> None:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit("opencv-python is required to write --video-out") from exc

    frames = _frame_paths(overlay_dir)
    if not frames:
        raise SystemExit(f"no overlay frames in {overlay_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise SystemExit(f"could not read {frames[0]}")
    height, width = first.shape[:2]
    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            image = cv2.imread(str(frame))
            if image is None:
                raise SystemExit(f"could not read {frame}")
            writer.write(image)
    finally:
        writer.release()


def _write_mask(mask, path: Path) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Meta SAM2 from one detection, then track that object through an episode."
    )
    parser.add_argument("--frames-dir", required=True, help="Directory of frame_*.jpg files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", default="blue ball")
    parser.add_argument("--seed-summary", help="SAM3 overlay summary.json to seed from.")
    parser.add_argument("--seed-box", type=_parse_box, help="Manual seed box: x0,y0,x1,y1.")
    parser.add_argument("--seed-frame", type=int, default=0, help="Frame index for --seed-box.")
    parser.add_argument("--min-seed-score", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Local SAM2 checkpoint. If omitted, --hf-model is used.",
    )
    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path, used with --checkpoint.",
    )
    parser.add_argument("--hf-model", default="facebook/sam2-hiera-large")
    parser.add_argument("--vos-optimized", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--async-loading-frames", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--video-out", default="", help="Optional MP4 output path.")
    parser.add_argument(
        "--mask-dir",
        default="",
        help="Optional directory for binary mask PNGs. Defaults to <out-dir>/mask_frames.",
    )
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"missing Python dependency {exc.name!r}; run this in the SAM2 vision environment"
        ) from exc

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlay_frames"
    mask_dir = Path(args.mask_dir) if args.mask_dir else out_dir / "mask_frames"
    seed_summary = Path(args.seed_summary) if args.seed_summary else None
    frames = _frame_paths(frames_dir)
    if not frames:
        raise SystemExit(f"no image frames found in {frames_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sam2_video_dir = _prepare_sam2_video_dir(frames, out_dir)

    seed_idx, seed_box, seed_meta = _load_seed(
        frames=frames,
        seed_box=args.seed_box,
        seed_frame=args.seed_frame,
        seed_summary=seed_summary,
        min_seed_score=args.min_seed_score,
    )

    device = _choose_device(args.device)
    predictor = _build_predictor(args, device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    segments: dict[int, np.ndarray] = {}
    with torch.inference_mode(), autocast:
        state = predictor.init_state(
            video_path=str(sam2_video_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=args.async_loading_frames,
        )
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_idx,
            obj_id=1,
            box=np.array(seed_box, dtype=np.float32),
        )

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state,
            start_frame_idx=seed_idx,
            reverse=False,
        ):
            for i, obj_id in enumerate(obj_ids):
                if obj_id == 1:
                    segments[int(frame_idx)] = _mask_to_numpy(mask_logits[i])

        if seed_idx > 0:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=seed_idx,
                reverse=True,
            ):
                for i, obj_id in enumerate(obj_ids):
                    if obj_id == 1:
                        segments[int(frame_idx)] = _mask_to_numpy(mask_logits[i])

    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    detections = []
    for idx, frame_path in enumerate(frames):
        image = Image.open(frame_path).convert("RGB")
        mask = segments.get(idx)
        box = _mask_box(mask) if mask is not None else None
        area_px = int(mask.sum()) if mask is not None else 0
        label = f"{args.prompt} SAM2"
        _overlay(image, mask, box, label, args.alpha).save(overlay_dir / frame_path.name)
        mask_path = None
        if mask is not None and area_px > 0:
            mask_path = mask_dir / f"{frame_path.stem}.png"
            _write_mask(mask, mask_path)
            try:
                mask_path_value = str(mask_path.relative_to(out_dir))
            except ValueError:
                mask_path_value = str(mask_path)
        else:
            mask_path_value = None
        detections.append(
            {
                "frame": frame_path.name,
                "tracked": mask is not None and area_px > 0,
                "area_px": area_px,
                "box_xyxy": box,
                "mask_path": mask_path_value,
            }
        )

    tracked = [d for d in detections if d["tracked"]]
    summary = {
        "frames_dir": str(frames_dir),
        "sam2_video_dir": str(sam2_video_dir),
        "prompt": args.prompt,
        "seed_frame_idx": seed_idx,
        "seed_frame": frames[seed_idx].name,
        "seed_box_xyxy": seed_box,
        "seed": seed_meta,
        "model": args.checkpoint or args.hf_model,
        "frames": len(detections),
        "tracked": len(tracked),
        "track_rate": len(tracked) / len(detections),
        "overlay_frames_dir": str(overlay_dir),
        "mask_frames_dir": str(mask_dir),
        "detections": detections,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.video_out:
        _write_video(overlay_dir, Path(args.video_out), args.fps)
        summary["video_out"] = args.video_out
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "detections"}, indent=2))


if __name__ == "__main__":
    main()
