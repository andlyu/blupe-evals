from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_value(cls, value: Any) -> "Box | None":
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        x0, y0, x1, y1 = (float(v) for v in value)
        if x1 <= x0 or y1 <= y0:
            return None
        return cls(x0=x0, y0=y0, x1=x1, y1=y1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def region(
        self,
        *,
        x_pad_frac: float,
        y_pad_frac: float,
        y0_frac: float,
        y1_frac: float,
    ) -> "Box":
        x_pad = self.width * x_pad_frac
        y_pad = self.height * y_pad_frac
        return Box(
            x0=self.x0 - x_pad,
            y0=self.y0 + self.height * y0_frac - y_pad,
            x1=self.x1 + x_pad,
            y1=self.y0 + self.height * y1_frac + y_pad,
        )


def _frame_sort_key(frame_name: str) -> tuple[str, int]:
    stem = Path(frame_name).stem
    suffix = stem.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return (stem[: -len(suffix)], int(suffix))
    return (stem, -1)


def _frame_idx(frame_name: str, fallback: int) -> int:
    suffix = Path(frame_name).stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"summary not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON summary: {path}: {exc}") from exc


def _extract_box(record: dict[str, Any]) -> Box | None:
    for key in ("box_xyxy", "display_box_xyxy", "top_box_xyxy"):
        box = Box.from_value(record.get(key))
        if box is not None:
            return box
    return None


def _track_by_frame(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    track: dict[str, dict[str, Any]] = {}
    for fallback, record in enumerate(summary.get("detections", [])):
        frame = record.get("frame")
        if not isinstance(frame, str):
            continue
        track[frame] = {
            "frame_idx": _frame_idx(frame, fallback),
            "box": _extract_box(record),
            "mask_path": record.get("mask_path") if isinstance(record.get("mask_path"), str) else None,
            "raw": record,
        }
    return track


def _median_box(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return Box(
        x0=float(median(box.x0 for box in boxes)),
        y0=float(median(box.y0 for box in boxes)),
        x1=float(median(box.x1 for box in boxes)),
        y1=float(median(box.y1 for box in boxes)),
    )


def _intersection_area(a: Box, b: Box) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _box_overlap_fraction(ball_box: Box | None, container_region: Box | None) -> float | None:
    if ball_box is None or container_region is None:
        return None
    if ball_box.area <= 0:
        return None
    return _intersection_area(ball_box, container_region) / ball_box.area


def _resolve_mask_path(summary_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return summary_path.parent / path


def _load_mask(path: Path | None):
    if path is None or not path.exists():
        return None
    import numpy as np
    from PIL import Image

    return np.array(Image.open(path).convert("L")) > 0


def _filled_contour_mask(mask):
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("opencv-python is required for --container-mask-mode filled-contour") from exc

    src = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(src)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled > 0


def _convex_hull_mask(mask):
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("opencv-python is required for --container-mask-mode convex-hull") from exc

    ys, xs = np.where(mask)
    hull_mask = np.zeros(mask.shape, dtype=np.uint8)
    if len(xs) < 3:
        return hull_mask > 0
    points = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    return hull_mask > 0


def _mask_overlap_fraction(ball_mask, container_mask) -> float | None:
    if ball_mask is None or container_mask is None:
        return None
    if ball_mask.shape != container_mask.shape:
        raise SystemExit(f"mask shape mismatch: ball={ball_mask.shape} container={container_mask.shape}")
    ball_area = int(ball_mask.sum())
    if ball_area <= 0:
        return None
    import numpy as np

    return float(np.logical_and(ball_mask, container_mask).sum() / ball_area)


def _true_runs(events: list[dict[str, Any]], key: str) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    cur_len = 0
    cur_start = 0
    for idx, event in enumerate(events):
        if event[key]:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
        else:
            if cur_len:
                runs.append((cur_len, cur_start, idx - 1))
            cur_len = 0
    if cur_len:
        runs.append((cur_len, cur_start, len(events) - 1))
    return runs


def _write_overlays(
    *,
    frames_dir: Path,
    overlay_dir: Path,
    events: list[dict[str, Any]],
    over_frame: str | None,
    leave_frame: str | None,
    successes: list[dict[str, Any]],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise SystemExit("Pillow is required for --overlay-dir") from exc

    overlay_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    event_by_frame = {event["frame"]: event for event in events}
    success_by_frame: dict[str, dict[str, Any]] = {}
    for success_idx, success in enumerate(successes, start=1):
        start_idx = success["over_run_start_event_idx"]
        end_idx = success["over_run_end_event_idx"]
        for event in events[start_idx : end_idx + 1]:
            success_by_frame[event["frame"]] = {"idx": success_idx, "role": "over"}
        success_by_frame[success["first_over_frame"]] = {"idx": success_idx, "role": "over_start"}
        success_by_frame[success["leave_frame"]] = {"idx": success_idx, "role": "leave"}

    frames = sorted(
        (path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: _frame_sort_key(path.name),
    )
    for frame_path in frames:
        event = event_by_frame.get(frame_path.name)
        if event is None:
            continue
        image = Image.open(frame_path).convert("RGB")
        try:
            import numpy as np

            arr = np.array(image).astype(np.float32)
            container_mask = _load_mask(Path(event["container_mask_resolved_path"])) if event.get("container_mask_resolved_path") else None
            ball_mask = _load_mask(Path(event["ball_mask_resolved_path"])) if event.get("ball_mask_resolved_path") else None
            if container_mask is not None:
                mode = event.get("container_mask_mode", "raw")
                if mode == "filled-contour":
                    container_mask = _filled_contour_mask(container_mask)
                elif mode == "convex-hull":
                    container_mask = _convex_hull_mask(container_mask)
                arr[container_mask] = 0.65 * arr[container_mask] + 0.35 * np.array([255, 190, 0])
            if ball_mask is not None:
                color = np.array([255, 80, 80]) if frame_path.name == leave_frame else np.array([64, 255, 96])
                arr[ball_mask] = 0.45 * arr[ball_mask] + 0.55 * color
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        except ModuleNotFoundError:
            pass
        draw = ImageDraw.Draw(image)
        cup_box = Box.from_value(event["container_box_xyxy"])
        cup_region = Box.from_value(event["container_region_xyxy"])
        ball_box = Box.from_value(event["ball_box_xyxy"])
        success = success_by_frame.get(frame_path.name)

        if cup_box is not None:
            draw.rectangle(cup_box.as_list(), outline=(255, 190, 0), width=3)
        if cup_region is not None:
            draw.rectangle(cup_region.as_list(), outline=(64, 255, 96), width=3)
        if ball_box is not None:
            if success and success["role"] == "leave":
                color = (255, 80, 80)
            elif success and success["role"] in {"over", "over_start"}:
                color = (64, 255, 96)
            elif event["over_container"]:
                color = (255, 220, 80)
            else:
                color = (32, 170, 255)
            draw.rectangle(ball_box.as_list(), outline=color, width=3)
            cx, cy = ball_box.center
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)

        if success and success["role"] == "over_start":
            label = f"SUCCESS {success['idx']}: >=90% OVER CUP"
        elif success and success["role"] == "over":
            overlap = event["ball_overlap_container_region"]
            label = f"SUCCESS {success['idx']}: over={overlap:.2f}"
        elif success and success["role"] == "leave":
            overlap = event["ball_overlap_container_region"]
            label = f"SUCCESS {success['idx']}: LEFT CUP ({overlap:.2f})"
        elif frame_path.name == over_frame:
            label = ">=90% OVER CUP"
        elif frame_path.name == leave_frame:
            label = "LEFT CUP"
        else:
            overlap = event["ball_overlap_container_region"]
            label = "missing" if overlap is None else f"overlap={overlap:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((8, 8, 16 + right - left, 18 + bottom - top), fill=(0, 0, 0))
        draw.text((12, 12), label, fill=(255, 255, 255), font=font)
        image.save(overlay_dir / frame_path.name)


def _text_report(payload: dict[str, Any]) -> str:
    result = payload["result"]
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        f"{status}: ball leaves cylindrical cup after >=90% overlap",
        f"first >= threshold frame: {result['first_over_frame']}",
        f"first >= threshold frame idx: {result['first_over_frame_idx']}",
        f"longest >= threshold run: {result['longest_over_run']}",
        f"leave frame: {result['leave_frame']}",
        f"leave frame idx: {result['leave_frame_idx']}",
        f"leave overlap: {result['leave_overlap']}",
        f"successes: {payload['success_count']}",
        f"reason: {result['reason']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the win condition: the ball is >=90% over the cardboard cylindrical cup, "
            "then later visibly leaves that cup region."
        )
    )
    parser.add_argument("--ball-summary", required=True)
    parser.add_argument("--container-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-out", default="")
    parser.add_argument("--frames-dir", default="", help="Required only with --overlay-dir.")
    parser.add_argument("--overlay-dir", default="")
    parser.add_argument("--overlap-threshold", type=float, default=0.9)
    parser.add_argument("--leave-overlap-threshold", type=float, default=0.1)
    parser.add_argument("--min-over-frames", type=int, default=1)
    parser.add_argument(
        "--overlap-source",
        choices=("auto", "masks", "boxes"),
        default="auto",
        help="Use mask overlap when available; boxes are kept only as a fallback/debug path.",
    )
    parser.add_argument(
        "--container-mask-mode",
        choices=("raw", "filled-contour", "convex-hull"),
        default="convex-hull",
        help="convex-hull turns the cup mask silhouette into a projected cup area.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--select-over-run",
        choices=("first", "last", "longest"),
        default="longest",
        help="Which >=threshold run to use before looking for a later leave event.",
    )
    parser.add_argument("--container-region-y0-frac", type=float, default=0.0)
    parser.add_argument("--container-region-y1-frac", type=float, default=0.62)
    parser.add_argument("--container-region-x-pad-frac", type=float, default=0.08)
    parser.add_argument("--container-region-y-pad-frac", type=float, default=0.04)
    args = parser.parse_args()

    if args.overlay_dir and not args.frames_dir:
        raise SystemExit("--frames-dir is required when --overlay-dir is set")
    if not 0.0 <= args.container_region_y0_frac < args.container_region_y1_frac <= 1.0:
        raise SystemExit("container region y fractions must satisfy 0 <= y0 < y1 <= 1")
    if args.min_over_frames < 1:
        raise SystemExit("--min-over-frames must be >= 1")
    if args.leave_overlap_threshold >= args.overlap_threshold:
        raise SystemExit("--leave-overlap-threshold must be below --overlap-threshold")

    ball_summary_path = Path(args.ball_summary)
    container_summary_path = Path(args.container_summary)
    ball_track = _track_by_frame(_load_summary(ball_summary_path))
    container_track = _track_by_frame(_load_summary(container_summary_path))
    if not ball_track:
        raise SystemExit(f"no ball detections in {ball_summary_path}")
    if not container_track:
        raise SystemExit(f"no container detections in {container_summary_path}")

    median_container_box = _median_box(
        [item["box"] for item in container_track.values() if item["box"] is not None]
    )
    if median_container_box is None:
        raise SystemExit(f"no valid container boxes in {container_summary_path}")
    container_region = median_container_box.region(
        x_pad_frac=args.container_region_x_pad_frac,
        y_pad_frac=args.container_region_y_pad_frac,
        y0_frac=args.container_region_y0_frac,
        y1_frac=args.container_region_y1_frac,
    )

    frames = sorted(set(ball_track) | set(container_track), key=_frame_sort_key)
    events: list[dict[str, Any]] = []
    mask_cache: dict[Path, Any] = {}

    def load_cached_mask(path: Path | None):
        if path is None:
            return None
        if path not in mask_cache:
            mask_cache[path] = _load_mask(path)
        return mask_cache[path]

    for fallback, frame in enumerate(frames):
        frame_idx = _frame_idx(frame, fallback)
        if frame_idx < args.start_frame:
            continue
        if args.end_frame is not None and frame_idx > args.end_frame:
            continue
        ball_item = ball_track.get(frame, {})
        container_item = container_track.get(frame, {})
        ball_box = ball_item.get("box")
        container_box = container_item.get("box") or median_container_box
        ball_mask_path = _resolve_mask_path(ball_summary_path, ball_item.get("mask_path"))
        container_mask_path = _resolve_mask_path(container_summary_path, container_item.get("mask_path"))
        ball_mask = load_cached_mask(ball_mask_path)
        container_mask = load_cached_mask(container_mask_path)
        if container_mask is not None and args.container_mask_mode == "filled-contour":
            cache_key = Path(str(container_mask_path) + "#filled")
            if cache_key not in mask_cache:
                mask_cache[cache_key] = _filled_contour_mask(container_mask)
            container_mask = mask_cache[cache_key]
        elif container_mask is not None and args.container_mask_mode == "convex-hull":
            cache_key = Path(str(container_mask_path) + "#hull")
            if cache_key not in mask_cache:
                mask_cache[cache_key] = _convex_hull_mask(container_mask)
            container_mask = mask_cache[cache_key]

        mask_overlap = _mask_overlap_fraction(ball_mask, container_mask)
        box_overlap = _box_overlap_fraction(ball_box, container_region)
        if args.overlap_source == "masks":
            overlap = mask_overlap
            overlap_source = "masks"
        elif args.overlap_source == "boxes":
            overlap = box_overlap
            overlap_source = "boxes"
        elif mask_overlap is not None:
            overlap = mask_overlap
            overlap_source = "masks"
        else:
            overlap = box_overlap
            overlap_source = "boxes"
        over_container = overlap is not None and overlap >= args.overlap_threshold
        left_container = overlap is not None and overlap <= args.leave_overlap_threshold
        events.append(
            {
                "frame": frame,
                "frame_idx": frame_idx,
                "ball_box_xyxy": ball_box.as_list() if ball_box is not None else None,
                "ball_center_xy": list(ball_box.center) if ball_box is not None else None,
                "ball_mask_path": ball_item.get("mask_path"),
                "ball_mask_resolved_path": str(ball_mask_path) if ball_mask_path is not None else None,
                "container_box_xyxy": container_box.as_list() if container_box is not None else None,
                "container_mask_path": container_item.get("mask_path"),
                "container_mask_resolved_path": (
                    str(container_mask_path) if container_mask_path is not None else None
                ),
                "container_region_xyxy": container_region.as_list(),
                "container_mask_mode": args.container_mask_mode,
                "overlap_source": overlap_source,
                "ball_overlap_container_region": overlap,
                "ball_box_overlap_container_region": box_overlap,
                "ball_mask_overlap_container_region": mask_overlap,
                "over_container": over_container,
                "left_container": left_container,
            }
        )

    runs = _true_runs(events, "over_container")
    valid_runs = [run for run in runs if run[0] >= args.min_over_frames]
    longest_len, longest_start, longest_end = (
        max(runs, key=lambda run: (run[0], run[1])) if runs else (0, None, None)
    )

    successes: list[dict[str, Any]] = []
    for run_len, run_start, run_end in valid_runs:
        run_leave_event = None
        run_leave_event_idx = None
        for event_idx, event in enumerate(events[run_end + 1 :], start=run_end + 1):
            if event["left_container"]:
                run_leave_event = event
                run_leave_event_idx = event_idx
                break
        if run_leave_event is None or run_leave_event_idx is None:
            continue
        successes.append(
            {
                "over_run_start_event_idx": run_start,
                "over_run_end_event_idx": run_end,
                "over_run_len": run_len,
                "first_over_frame": events[run_start]["frame"],
                "first_over_frame_idx": events[run_start]["frame_idx"],
                "last_over_frame": events[run_end]["frame"],
                "last_over_frame_idx": events[run_end]["frame_idx"],
                "leave_event_idx": run_leave_event_idx,
                "leave_frame": run_leave_event["frame"],
                "leave_frame_idx": run_leave_event["frame_idx"],
                "leave_overlap": run_leave_event["ball_overlap_container_region"],
            }
        )

    accepted_over_start: int | None = None
    accepted_over_end: int | None = None
    if valid_runs:
        if args.select_over_run == "first":
            selected_run = valid_runs[0]
        elif args.select_over_run == "last":
            selected_run = valid_runs[-1]
        else:
            selected_run = max(valid_runs, key=lambda run: (run[0], run[1]))
        _, accepted_over_start, accepted_over_end = selected_run

    leave_event: dict[str, Any] | None = None
    if accepted_over_end is not None:
        for event in events[accepted_over_end + 1 :]:
            if event["left_container"]:
                leave_event = event
                break

    first_over_event = events[accepted_over_start] if accepted_over_start is not None else None
    passed = first_over_event is not None and leave_event is not None
    if passed:
        reason = "ball reached >=90% overlap with the cup region, then later visibly left it"
    elif first_over_event is None:
        reason = "ball never reached the required overlap with the cup region"
    else:
        reason = "ball reached the cup overlap threshold but no later visible leave frame was found"

    result = {
        "passed": passed,
        "reason": reason,
        "first_over_frame": first_over_event["frame"] if first_over_event else None,
        "first_over_frame_idx": first_over_event["frame_idx"] if first_over_event else None,
        "accepted_over_run_end_frame": (
            events[accepted_over_end]["frame"] if accepted_over_end is not None else None
        ),
        "accepted_over_run_end_frame_idx": (
            events[accepted_over_end]["frame_idx"] if accepted_over_end is not None else None
        ),
        "selected_over_run": args.select_over_run,
        "longest_over_run": longest_len,
        "longest_over_run_start": (
            events[longest_start]["frame"] if longest_start is not None else None
        ),
        "longest_over_run_end": events[longest_end]["frame"] if longest_end is not None else None,
        "leave_frame": leave_event["frame"] if leave_event else None,
        "leave_frame_idx": leave_event["frame_idx"] if leave_event else None,
        "leave_overlap": leave_event["ball_overlap_container_region"] if leave_event else None,
    }
    payload = {
        "ball_summary": str(ball_summary_path),
        "container_summary": str(container_summary_path),
        "criteria": {
            "overlap_threshold": args.overlap_threshold,
            "leave_overlap_threshold": args.leave_overlap_threshold,
            "min_over_frames": args.min_over_frames,
            "overlap_source": args.overlap_source,
            "container_mask_mode": args.container_mask_mode,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "select_over_run": args.select_over_run,
            "container_region_y0_frac": args.container_region_y0_frac,
            "container_region_y1_frac": args.container_region_y1_frac,
            "container_region_x_pad_frac": args.container_region_x_pad_frac,
            "container_region_y_pad_frac": args.container_region_y_pad_frac,
        },
        "container": {
            "median_box_xyxy": median_container_box.as_list(),
            "region_xyxy": container_region.as_list(),
        },
        "frames": len(events),
        "visible_ball_frames": sum(1 for event in events if event["ball_box_xyxy"] is not None),
        "over_container_frames": sum(1 for event in events if event["over_container"]),
        "left_container_frames": sum(1 for event in events if event["left_container"]),
        "success_count": len(successes),
        "successes": successes,
        "result": result,
        "events": events,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    text_report = _text_report(payload)
    if args.text_out:
        text_out = Path(args.text_out)
        text_out.parent.mkdir(parents=True, exist_ok=True)
        text_out.write_text(text_report + "\n")

    if args.overlay_dir:
        _write_overlays(
            frames_dir=Path(args.frames_dir),
            overlay_dir=Path(args.overlay_dir),
            events=events,
            over_frame=result["first_over_frame"],
            leave_frame=result["leave_frame"],
            successes=successes,
        )

    print(text_report)
    print(json.dumps({k: v for k, v in payload.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
