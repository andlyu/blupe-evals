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
    tracks: dict[str, dict[str, Any]] = {}
    for record in summary.get("detections", []):
        frame = record.get("frame")
        if not isinstance(frame, str):
            continue
        box = _extract_box(record)
        tracks[frame] = {
            "box": box,
            "raw": record,
        }
    return tracks


def _median_box(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return Box(
        x0=float(median(box.x0 for box in boxes)),
        y0=float(median(box.y0 for box in boxes)),
        x1=float(median(box.x1 for box in boxes)),
        y1=float(median(box.y1 for box in boxes)),
    )


def _contains_point(box: Box, point: tuple[float, float]) -> bool:
    x, y = point
    return box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1


def _intersection_area(a: Box, b: Box) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _format_frame_idx(frame_name: str, fallback: int) -> int:
    suffix = Path(frame_name).stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


def _run_lengths(events: list[dict[str, Any]]) -> tuple[int, int | None, int | None]:
    best_len = 0
    best_start: int | None = None
    best_end: int | None = None
    cur_len = 0
    cur_start = 0
    for idx, event in enumerate(events):
        if event["in_cup"]:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = idx
        else:
            cur_len = 0
    return best_len, best_start, best_end


def _write_overlays(
    *,
    frames_dir: Path,
    overlay_dir: Path,
    events: list[dict[str, Any]],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise SystemExit("Pillow is required for --overlay-dir") from exc

    overlay_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    event_by_frame = {event["frame"]: event for event in events}
    frames = sorted(
        (p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: _frame_sort_key(p.name),
    )
    for frame_path in frames:
        event = event_by_frame.get(frame_path.name)
        if event is None:
            continue
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        cup_box = Box.from_value(event["cup_box_xyxy"])
        cup_region = Box.from_value(event["cup_region_xyxy"])
        ball_box = Box.from_value(event["ball_box_xyxy"])

        if cup_box is not None:
            draw.rectangle(cup_box.as_list(), outline=(255, 190, 0), width=3)
        if cup_region is not None:
            draw.rectangle(cup_region.as_list(), outline=(64, 255, 96), width=3)
        if ball_box is not None:
            color = (64, 255, 96) if event["in_cup"] else (32, 170, 255)
            draw.rectangle(ball_box.as_list(), outline=color, width=3)
            cx, cy = ball_box.center
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)

        label = "IN CUP" if event["in_cup"] else "tracking"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((8, 8, 16 + right - left, 18 + bottom - top), fill=(0, 0, 0))
        draw.text((12, 12), label, fill=(255, 255, 255), font=font)
        image.save(overlay_dir / frame_path.name)


def _text_report(payload: dict[str, Any]) -> str:
    result = payload["result"]
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        f"{status}: ball-in-cup verification",
        f"frames: {payload['frames']}",
        f"ball visible frames: {payload['ball_visible_frames']}",
        f"in-cup frames: {payload['in_cup_frames']}",
        f"cup median box: {payload['cup_location']['median_box_xyxy']}",
        f"first in-cup frame: {result['first_in_cup_frame']}",
        f"last in-cup frame: {result['last_in_cup_frame']}",
        f"longest in-cup run: {result['longest_in_cup_run']}",
        f"longest run span: {result['longest_in_cup_run_start']} -> {result['longest_in_cup_run_end']}",
        f"last visible ball frame: {result['last_visible_ball_frame']}",
        f"last visible ball in cup: {result['last_visible_ball_in_cup']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether a tracked ball entered a tracked cup using SAM2/SAM3 summary.json files."
        )
    )
    parser.add_argument("--ball-summary", required=True, help="SAM2 or SAM3 summary for the ball.")
    parser.add_argument("--cup-summary", required=True, help="SAM2 or SAM3 summary for the cup.")
    parser.add_argument("--out", required=True, help="Output JSON verdict path.")
    parser.add_argument(
        "--text-out",
        default="",
        help="Optional terminal-friendly text verdict path.",
    )
    parser.add_argument(
        "--frames-dir",
        default="",
        help="Optional source frames directory. Required only with --overlay-dir.",
    )
    parser.add_argument(
        "--overlay-dir",
        default="",
        help="Optional directory for diagnostic overlay frames.",
    )
    parser.add_argument(
        "--cup-box-mode",
        choices=("per-frame", "median"),
        default="median",
        help="Use one robust median cup box for the whole video, or per-frame cup boxes.",
    )
    parser.add_argument(
        "--cup-memory-frames",
        type=int,
        default=-1,
        help="How many missed cup frames to fill from the last known cup box. -1 means indefinitely.",
    )
    parser.add_argument(
        "--cup-region-y0-frac",
        type=float,
        default=0.0,
        help="Top of accepted cup region as a fraction of cup box height.",
    )
    parser.add_argument(
        "--cup-region-y1-frac",
        type=float,
        default=1.0,
        help="Bottom of accepted cup region as a fraction of cup box height.",
    )
    parser.add_argument(
        "--cup-region-x-pad-frac",
        type=float,
        default=0.08,
        help="Horizontal padding added to accepted cup region, relative to cup box width.",
    )
    parser.add_argument(
        "--cup-region-y-pad-frac",
        type=float,
        default=0.04,
        help="Vertical padding added to accepted cup region, relative to cup box height.",
    )
    parser.add_argument(
        "--min-ball-overlap",
        type=float,
        default=0.15,
        help="Accept a frame when at least this fraction of the ball box overlaps the cup region.",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=3,
        help="Number of consecutive accepted frames required for pass=True.",
    )
    args = parser.parse_args()

    if args.overlay_dir and not args.frames_dir:
        raise SystemExit("--frames-dir is required when --overlay-dir is set")
    if not 0.0 <= args.cup_region_y0_frac < args.cup_region_y1_frac <= 1.0:
        raise SystemExit("cup region y fractions must satisfy 0 <= y0 < y1 <= 1")
    if args.min_consecutive_frames < 1:
        raise SystemExit("--min-consecutive-frames must be >= 1")
    if args.min_ball_overlap < 0.0:
        raise SystemExit("--min-ball-overlap must be >= 0")

    ball_summary_path = Path(args.ball_summary)
    cup_summary_path = Path(args.cup_summary)
    ball_summary = _load_summary(ball_summary_path)
    cup_summary = _load_summary(cup_summary_path)
    ball_track = _track_by_frame(ball_summary)
    cup_track = _track_by_frame(cup_summary)
    if not ball_track:
        raise SystemExit(f"no ball detections in {ball_summary_path}")
    if not cup_track:
        raise SystemExit(f"no cup detections in {cup_summary_path}")

    frames = sorted(set(ball_track) | set(cup_track), key=_frame_sort_key)
    median_cup_box = _median_box([item["box"] for item in cup_track.values() if item["box"]])
    if args.cup_box_mode == "median" and median_cup_box is None:
        raise SystemExit(f"no valid cup boxes in {cup_summary_path}")

    events: list[dict[str, Any]] = []
    last_cup_box: Box | None = None
    missed_cup_frames = 0
    for idx, frame in enumerate(frames):
        ball_box = ball_track.get(frame, {}).get("box")
        current_cup_box = cup_track.get(frame, {}).get("box")

        if args.cup_box_mode == "median":
            cup_box = median_cup_box
        elif current_cup_box is not None:
            cup_box = current_cup_box
            last_cup_box = current_cup_box
            missed_cup_frames = 0
        else:
            missed_cup_frames += 1
            if last_cup_box is not None and (
                args.cup_memory_frames < 0 or missed_cup_frames <= args.cup_memory_frames
            ):
                cup_box = last_cup_box
            else:
                cup_box = None

        cup_region = (
            cup_box.region(
                x_pad_frac=args.cup_region_x_pad_frac,
                y_pad_frac=args.cup_region_y_pad_frac,
                y0_frac=args.cup_region_y0_frac,
                y1_frac=args.cup_region_y1_frac,
            )
            if cup_box is not None
            else None
        )

        ball_center = ball_box.center if ball_box is not None else None
        center_in_cup = (
            ball_center is not None and cup_region is not None and _contains_point(cup_region, ball_center)
        )
        overlap_frac = 0.0
        if ball_box is not None and cup_region is not None and ball_box.area > 0:
            overlap_frac = _intersection_area(ball_box, cup_region) / ball_box.area
        in_cup = bool(center_in_cup or overlap_frac >= args.min_ball_overlap)

        events.append(
            {
                "frame": frame,
                "frame_idx": _format_frame_idx(frame, idx),
                "ball_box_xyxy": ball_box.as_list() if ball_box is not None else None,
                "ball_center_xy": list(ball_center) if ball_center is not None else None,
                "cup_box_xyxy": cup_box.as_list() if cup_box is not None else None,
                "cup_region_xyxy": cup_region.as_list() if cup_region is not None else None,
                "cup_box_source": (
                    "median"
                    if args.cup_box_mode == "median"
                    else "detection"
                    if current_cup_box is not None
                    else "memory"
                    if cup_box is not None
                    else "none"
                ),
                "ball_overlap_cup_region": overlap_frac,
                "ball_center_in_cup_region": center_in_cup,
                "in_cup": in_cup,
            }
        )

    longest_run, run_start, run_end = _run_lengths(events)
    in_cup_events = [event for event in events if event["in_cup"]]
    first_in_cup = in_cup_events[0] if in_cup_events else None
    last_in_cup = in_cup_events[-1] if in_cup_events else None
    visible_ball_events = [event for event in events if event["ball_box_xyxy"] is not None]
    last_visible_ball = visible_ball_events[-1] if visible_ball_events else None
    had_outside_before_entry = False
    if first_in_cup is not None:
        first_idx = events.index(first_in_cup)
        had_outside_before_entry = any(
            event["ball_box_xyxy"] is not None and not event["in_cup"] for event in events[:first_idx]
        )

    result = {
        "passed": longest_run >= args.min_consecutive_frames,
        "first_in_cup_frame": first_in_cup["frame"] if first_in_cup else None,
        "first_in_cup_frame_idx": first_in_cup["frame_idx"] if first_in_cup else None,
        "last_in_cup_frame": last_in_cup["frame"] if last_in_cup else None,
        "last_in_cup_frame_idx": last_in_cup["frame_idx"] if last_in_cup else None,
        "longest_in_cup_run": longest_run,
        "longest_in_cup_run_start": events[run_start]["frame"] if run_start is not None else None,
        "longest_in_cup_run_end": events[run_end]["frame"] if run_end is not None else None,
        "had_outside_before_entry": had_outside_before_entry,
        "last_visible_ball_frame": last_visible_ball["frame"] if last_visible_ball else None,
        "last_visible_ball_in_cup": bool(last_visible_ball and last_visible_ball["in_cup"]),
    }

    payload = {
        "ball_summary": str(ball_summary_path),
        "cup_summary": str(cup_summary_path),
        "criteria": {
            "cup_box_mode": args.cup_box_mode,
            "cup_memory_frames": args.cup_memory_frames,
            "cup_region_y0_frac": args.cup_region_y0_frac,
            "cup_region_y1_frac": args.cup_region_y1_frac,
            "cup_region_x_pad_frac": args.cup_region_x_pad_frac,
            "cup_region_y_pad_frac": args.cup_region_y_pad_frac,
            "min_ball_overlap": args.min_ball_overlap,
            "min_consecutive_frames": args.min_consecutive_frames,
        },
        "cup_location": {
            "median_box_xyxy": median_cup_box.as_list() if median_cup_box is not None else None,
            "valid_box_frames": sum(1 for item in cup_track.values() if item["box"] is not None),
        },
        "frames": len(events),
        "ball_visible_frames": len(visible_ball_events),
        "in_cup_frames": len(in_cup_events),
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
        )

    print(text_report)
    print(json.dumps({k: v for k, v in payload.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
