from __future__ import annotations

import argparse
import json
import math
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


@dataclass(frozen=True)
class TrackEvent:
    frame: str
    frame_idx: int
    box: Box | None
    raw: dict[str, Any]


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


def _events_from_summary(summary: dict[str, Any]) -> list[TrackEvent]:
    events: list[TrackEvent] = []
    for fallback, record in enumerate(summary.get("detections", [])):
        frame = record.get("frame")
        if not isinstance(frame, str):
            continue
        events.append(
            TrackEvent(
                frame=frame,
                frame_idx=_frame_idx(frame, fallback),
                box=_extract_box(record),
                raw=record,
            )
        )
    return sorted(events, key=lambda event: (event.frame_idx, _frame_sort_key(event.frame)))


def _event_by_idx(events: list[TrackEvent]) -> dict[int, TrackEvent]:
    return {event.frame_idx: event for event in events}


def _median_box(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return Box(
        x0=float(median(box.x0 for box in boxes)),
        y0=float(median(box.y0 for box in boxes)),
        x1=float(median(box.x1 for box in boxes)),
        y1=float(median(box.y1 for box in boxes)),
    )


def _parse_box(value: str) -> Box:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x0,y0,x1,y1")
    box = Box.from_value(parts)
    if box is None:
        raise argparse.ArgumentTypeError("box must satisfy x1>x0 and y1>y0")
    return box


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


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _persistent_release_candidate(
    events_by_idx: dict[int, TrackEvent],
    *,
    start_idx: int,
    candidate_idx: int,
    held_region: Box | None,
    held_center: tuple[float, float] | None,
    release_distance_px: float,
    persist_frames: int,
    persist_window: int,
) -> bool:
    evidence = 0
    for idx in range(candidate_idx, candidate_idx + persist_window + 1):
        event = events_by_idx.get(idx)
        if event is None or event.box is None:
            evidence += 1
            continue
        center = event.box.center
        outside_held_region = held_region is not None and not _contains_point(held_region, center)
        away_from_held_center = (
            held_center is not None and _distance(center, held_center) >= release_distance_px
        )
        if outside_held_region or away_from_held_center:
            evidence += 1
    return candidate_idx >= start_idx and evidence >= persist_frames


def _infer_release_frame(
    wrist_events: list[TrackEvent],
    *,
    held_region: Box | None,
    held_sample_frames: int,
    release_distance_px: float,
    release_min_frame: int,
    persist_frames: int,
    persist_window: int,
) -> dict[str, Any]:
    events_by_idx = _event_by_idx(wrist_events)
    visible = [event for event in wrist_events if event.box is not None]
    if not visible:
        raise SystemExit("wrist summary has no visible ball boxes")

    held_samples = [event for event in visible if event.frame_idx >= release_min_frame][
        :held_sample_frames
    ]
    if len(held_samples) < max(2, min(held_sample_frames, 4)):
        held_samples = visible[:held_sample_frames]
    if not held_samples:
        raise SystemExit("not enough wrist ball boxes to infer a held position")

    held_center = (
        float(median(event.box.center[0] for event in held_samples if event.box is not None)),
        float(median(event.box.center[1] for event in held_samples if event.box is not None)),
    )
    start_idx = max(release_min_frame, held_samples[-1].frame_idx + 1)
    max_idx = max(event.frame_idx for event in wrist_events)
    for idx in range(start_idx, max_idx + 1):
        if _persistent_release_candidate(
            events_by_idx,
            start_idx=start_idx,
            candidate_idx=idx,
            held_region=held_region,
            held_center=held_center if held_region is None else None,
            release_distance_px=release_distance_px,
            persist_frames=persist_frames,
            persist_window=persist_window,
        ):
            return {
                "source": "wrist_ball_summary",
                "frame_idx": idx,
                "held_center_xy": list(held_center),
                "held_region_xyxy": held_region.as_list() if held_region is not None else None,
                "held_sample_frames": [event.frame for event in held_samples],
                "release_distance_px": release_distance_px,
                "persist_frames": persist_frames,
                "persist_window": persist_window,
            }

    raise SystemExit("could not infer a release frame from the wrist ball track")


def _infer_release_frame_from_wrist_drop(
    wrist_events: list[TrackEvent],
    *,
    drop_y_threshold: float | None,
    release_min_frame: int,
    persist_frames: int,
    persist_window: int,
) -> dict[str, Any]:
    if not wrist_events:
        raise SystemExit("wrist summary has no events")
    events_by_idx = _event_by_idx(wrist_events)
    max_idx = max(event.frame_idx for event in wrist_events)
    for candidate_idx in range(release_min_frame, max_idx + 1):
        candidate = events_by_idx.get(candidate_idx)
        candidate_missing = candidate is None or candidate.box is None
        candidate_below_threshold = (
            candidate is not None
            and candidate.box is not None
            and drop_y_threshold is not None
            and candidate.box.center[1] >= drop_y_threshold
        )
        if not (candidate_missing or candidate_below_threshold):
            continue

        evidence = 0
        for idx in range(candidate_idx, candidate_idx + persist_window + 1):
            event = events_by_idx.get(idx)
            missing = event is None or event.box is None
            below_threshold = (
                event is not None
                and event.box is not None
                and drop_y_threshold is not None
                and event.box.center[1] >= drop_y_threshold
            )
            if missing or below_threshold:
                evidence += 1
        if evidence >= persist_frames:
            return {
                "source": "wrist_drop_threshold_or_missing",
                "frame_idx": candidate_idx,
                "drop_y_threshold": drop_y_threshold,
                "persist_frames": persist_frames,
                "persist_window": persist_window,
            }

    raise SystemExit("could not infer a drop/missing release frame from the wrist ball track")


def _container_box_for_frame(
    *,
    frame_idx: int,
    container_by_idx: dict[int, TrackEvent],
    median_container_box: Box | None,
    manual_container_box: Box | None,
    mode: str,
    memory_frames: int,
) -> tuple[Box | None, str]:
    if manual_container_box is not None:
        return manual_container_box, "manual"
    if mode == "median":
        return median_container_box, "median"

    current = container_by_idx.get(frame_idx)
    if current is not None and current.box is not None:
        return current.box, "detection"

    if memory_frames == 0:
        return None, "none"
    lower = frame_idx - memory_frames if memory_frames > 0 else min(container_by_idx) if container_by_idx else frame_idx
    candidates = [
        event for idx, event in container_by_idx.items() if lower <= idx < frame_idx and event.box is not None
    ]
    if not candidates:
        return None, "none"
    latest = max(candidates, key=lambda event: event.frame_idx)
    return latest.box, "memory"


def _score_ball_against_container(ball_box: Box | None, container_region: Box | None) -> dict[str, Any]:
    if ball_box is None or container_region is None:
        return {
            "ball_center_in_container_region": False,
            "ball_overlap_container_region": 0.0,
            "inside_container_region": False,
        }
    center_in_region = _contains_point(container_region, ball_box.center)
    overlap = _intersection_area(ball_box, container_region) / ball_box.area if ball_box.area > 0 else 0.0
    return {
        "ball_center_in_container_region": center_in_region,
        "ball_overlap_container_region": overlap,
        "inside_container_region": center_in_region,
    }


def _inside_container_region(score: dict[str, Any], *, min_overlap: float, mode: str) -> bool:
    if mode == "overlap":
        return bool(score["ball_overlap_container_region"] >= min_overlap)
    return bool(
        score["ball_center_in_container_region"]
        or score["ball_overlap_container_region"] >= min_overlap
    )


def _pre_drop_above_event(
    *,
    fixed_ball_by_idx: dict[int, TrackEvent],
    container_by_idx: dict[int, TrackEvent],
    median_container_box: Box | None,
    manual_container_box: Box | None,
    mode: str,
    memory_frames: int,
    window_start: int,
    lookback_frames: int,
    container_region_y0_frac: float,
    container_region_y1_frac: float,
    container_region_x_pad_frac: float,
    container_region_y_pad_frac: float,
    above_mode: str,
) -> dict[str, Any] | None:
    for frame_idx in range(window_start - 1, window_start - lookback_frames - 1, -1):
        ball_event = fixed_ball_by_idx.get(frame_idx)
        if ball_event is None or ball_event.box is None:
            continue
        container_box, container_source = _container_box_for_frame(
            frame_idx=frame_idx,
            container_by_idx=container_by_idx,
            median_container_box=median_container_box,
            manual_container_box=manual_container_box,
            mode=mode,
            memory_frames=memory_frames,
        )
        if container_box is None:
            continue
        container_region = container_box.region(
            x_pad_frac=container_region_x_pad_frac,
            y_pad_frac=container_region_y_pad_frac,
            y0_frac=container_region_y0_frac,
            y1_frac=container_region_y1_frac,
        )
        cx, cy = ball_event.box.center
        in_column = container_region.x0 <= cx <= container_region.x1
        above_edge = in_column and cy < container_region.y0
        in_region = _contains_point(container_region, (cx, cy))
        if above_mode == "column-or-region":
            above = bool(above_edge or in_region)
        else:
            above = above_edge
        return {
            "fixed_frame_idx": frame_idx,
            "fixed_frame": ball_event.frame,
            "ball_box_xyxy": ball_event.box.as_list(),
            "ball_center_xy": [cx, cy],
            "container_region_xyxy": container_region.as_list(),
            "container_box_source": container_source,
            "above_mode": above_mode,
            "in_container_column": in_column,
            "in_container_region": in_region,
            "above_cylinder_area": above,
        }
    return None


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
    events_by_frame = {event["fixed_frame"]: event for event in events}
    frames = sorted(
        (path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: _frame_sort_key(path.name),
    )
    for frame_path in frames:
        event = events_by_frame.get(frame_path.name)
        if event is None:
            continue
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        container_box = Box.from_value(event["container_box_xyxy"])
        container_region = Box.from_value(event["container_region_xyxy"])
        ball_box = Box.from_value(event["ball_box_xyxy"])
        if container_box is not None:
            draw.rectangle(container_box.as_list(), outline=(255, 190, 0), width=3)
        if container_region is not None:
            draw.rectangle(container_region.as_list(), outline=(64, 255, 96), width=3)
        if ball_box is not None:
            color = (64, 255, 96) if event["inside_container_region"] else (32, 170, 255)
            draw.rectangle(ball_box.as_list(), outline=color, width=3)
            cx, cy = ball_box.center
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)

        label = "CHECK IN" if event["is_selected_check_frame"] else "drop window"
        if event["inside_container_region"]:
            label += " PASS"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((8, 8, 16 + right - left, 18 + bottom - top), fill=(0, 0, 0))
        draw.text((12, 12), label, fill=(255, 255, 255), font=font)
        image.save(overlay_dir / frame_path.name)


def _text_report(payload: dict[str, Any]) -> str:
    result = payload["result"]
    if result["status"] == "passed":
        status = "PASS"
    elif result["status"] == "failed":
        status = "FAIL"
    else:
        status = "UNKNOWN"
    lines = [
        f"{status}: drop-into-container verification",
        f"release frame idx: {payload['release']['frame_idx']}",
        f"fixed check window: {payload['fixed_check']['window_start_idx']} -> {payload['fixed_check']['window_end_idx']}",
        f"selected fixed frame: {result['selected_fixed_frame']}",
        f"selected fixed frame idx: {result['selected_fixed_frame_idx']}",
        f"container box source: {result['container_box_source']}",
        f"container region: {result['container_region_xyxy']}",
        f"ball box: {result['ball_box_xyxy']}",
        f"ball center in region: {result['ball_center_in_container_region']}",
        f"ball overlap region: {result['ball_overlap_container_region']:.3f}",
        f"reason: {result['reason']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use a wrist-camera release time and a fixed-camera ball/container view to verify "
            "whether the dropped ball first appears inside the container opening."
        )
    )
    parser.add_argument("--fixed-ball-summary", required=True, help="Ball track summary for fixed camera.")
    parser.add_argument("--out", required=True, help="Output JSON verdict path.")
    parser.add_argument("--text-out", default="", help="Optional text verdict path.")
    parser.add_argument("--wrist-ball-summary", default="", help="Ball track summary for wrist camera.")
    parser.add_argument("--release-frame", type=int, default=None, help="Manual release frame index.")
    parser.add_argument(
        "--wrist-release-mode",
        choices=("held-motion", "drop-threshold"),
        default="held-motion",
        help=(
            "How to infer release from --wrist-ball-summary. drop-threshold fires when the "
            "wrist ball center is below --wrist-drop-y-threshold or missing."
        ),
    )
    parser.add_argument(
        "--wrist-drop-y-threshold",
        type=float,
        default=None,
        help="Image y threshold for drop-threshold mode. Larger y is lower in the image.",
    )
    parser.add_argument("--wrist-held-region", type=_parse_box, help="Wrist image region where held ball center should remain.")
    parser.add_argument("--held-sample-frames", type=int, default=12)
    parser.add_argument("--release-distance-px", type=float, default=70.0)
    parser.add_argument("--release-min-frame", type=int, default=0)
    parser.add_argument("--release-persist-frames", type=int, default=3)
    parser.add_argument("--release-persist-window", type=int, default=5)
    parser.add_argument("--container-summary", default="", help="Container track summary for fixed camera.")
    parser.add_argument("--container-box", type=_parse_box, help="Manual fixed-camera container box x0,y0,x1,y1.")
    parser.add_argument(
        "--container-box-mode",
        choices=("median", "per-frame"),
        default="median",
        help="How to use --container-summary when no manual --container-box is provided.",
    )
    parser.add_argument(
        "--container-memory-frames",
        type=int,
        default=-1,
        help="For per-frame container mode, fill misses from prior box. -1 means indefinitely.",
    )
    parser.add_argument("--container-region-y0-frac", type=float, default=0.0)
    parser.add_argument("--container-region-y1-frac", type=float, default=0.62)
    parser.add_argument("--container-region-x-pad-frac", type=float, default=0.08)
    parser.add_argument("--container-region-y-pad-frac", type=float, default=0.04)
    parser.add_argument("--min-ball-overlap", type=float, default=0.15)
    parser.add_argument(
        "--containment-mode",
        choices=("center-or-overlap", "overlap"),
        default="center-or-overlap",
        help="Use overlap mode with --min-ball-overlap 0.9 for the strict 90%% containment criterion.",
    )
    parser.add_argument(
        "--require-pre-drop-above",
        action="store_true",
        help="Require the most recent fixed-camera ball before release to be above the container column.",
    )
    parser.add_argument(
        "--above-mode",
        choices=("above-edge", "column-or-region"),
        default="above-edge",
        help=(
            "above-edge requires the ball center to be vertically above the container region. "
            "column-or-region also accepts the ball already being over/in the opening region."
        ),
    )
    parser.add_argument("--above-window-before", type=int, default=8)
    parser.add_argument("--sync-offset-fixed-minus-wrist", type=int, default=0)
    parser.add_argument("--fixed-delay-frames", type=int, default=0)
    parser.add_argument("--fixed-window-after", type=int, default=8)
    parser.add_argument(
        "--select-frame",
        choices=("first-visible", "first-frame"),
        default="first-visible",
        help="Use the first visible fixed-camera ball in the window, or the first frame even if ball is missing.",
    )
    parser.add_argument("--fixed-frames-dir", default="", help="Required only with --overlay-dir.")
    parser.add_argument("--overlay-dir", default="", help="Optional fixed-camera diagnostic overlays.")
    args = parser.parse_args()

    if args.release_frame is None and not args.wrist_ball_summary:
        raise SystemExit("provide either --release-frame or --wrist-ball-summary")
    if args.container_box is None and not args.container_summary:
        raise SystemExit("provide either --container-box or --container-summary")
    if args.overlay_dir and not args.fixed_frames_dir:
        raise SystemExit("--fixed-frames-dir is required with --overlay-dir")
    if not 0.0 <= args.container_region_y0_frac < args.container_region_y1_frac <= 1.0:
        raise SystemExit("container region y fractions must satisfy 0 <= y0 < y1 <= 1")
    if args.release_persist_frames < 1:
        raise SystemExit("--release-persist-frames must be >= 1")
    if args.fixed_window_after < 0:
        raise SystemExit("--fixed-window-after must be >= 0")

    fixed_ball_summary_path = Path(args.fixed_ball_summary)
    fixed_ball_events = _events_from_summary(_load_summary(fixed_ball_summary_path))
    if not fixed_ball_events:
        raise SystemExit(f"no detections in fixed ball summary: {fixed_ball_summary_path}")
    fixed_ball_by_idx = _event_by_idx(fixed_ball_events)

    if args.release_frame is not None:
        release = {"source": "manual", "frame_idx": args.release_frame}
    else:
        wrist_events = _events_from_summary(_load_summary(Path(args.wrist_ball_summary)))
        if args.wrist_release_mode == "drop-threshold":
            release = _infer_release_frame_from_wrist_drop(
                wrist_events,
                drop_y_threshold=args.wrist_drop_y_threshold,
                release_min_frame=args.release_min_frame,
                persist_frames=args.release_persist_frames,
                persist_window=args.release_persist_window,
            )
        else:
            release = _infer_release_frame(
                wrist_events,
                held_region=args.wrist_held_region,
                held_sample_frames=args.held_sample_frames,
                release_distance_px=args.release_distance_px,
                release_min_frame=args.release_min_frame,
                persist_frames=args.release_persist_frames,
                persist_window=args.release_persist_window,
            )

    container_by_idx: dict[int, TrackEvent] = {}
    median_container_box: Box | None = None
    if args.container_summary:
        container_events = _events_from_summary(_load_summary(Path(args.container_summary)))
        container_by_idx = _event_by_idx(container_events)
        median_container_box = _median_box([event.box for event in container_events if event.box is not None])
        if args.container_box is None and median_container_box is None:
            raise SystemExit(f"no valid container boxes in {args.container_summary}")

    window_start = (
        int(release["frame_idx"]) + args.sync_offset_fixed_minus_wrist + args.fixed_delay_frames
    )
    window_end = window_start + args.fixed_window_after
    pre_drop_event = _pre_drop_above_event(
        fixed_ball_by_idx=fixed_ball_by_idx,
        container_by_idx=container_by_idx,
        median_container_box=median_container_box,
        manual_container_box=args.container_box,
        mode=args.container_box_mode,
        memory_frames=args.container_memory_frames,
        window_start=window_start,
        lookback_frames=args.above_window_before,
        container_region_y0_frac=args.container_region_y0_frac,
        container_region_y1_frac=args.container_region_y1_frac,
        container_region_x_pad_frac=args.container_region_x_pad_frac,
        container_region_y_pad_frac=args.container_region_y_pad_frac,
        above_mode=args.above_mode,
    )
    window_events: list[dict[str, Any]] = []
    selected_event: dict[str, Any] | None = None

    for frame_idx in range(window_start, window_end + 1):
        ball_event = fixed_ball_by_idx.get(frame_idx)
        ball_box = ball_event.box if ball_event is not None else None
        container_box, container_source = _container_box_for_frame(
            frame_idx=frame_idx,
            container_by_idx=container_by_idx,
            median_container_box=median_container_box,
            manual_container_box=args.container_box,
            mode=args.container_box_mode,
            memory_frames=args.container_memory_frames,
        )
        container_region = (
            container_box.region(
                x_pad_frac=args.container_region_x_pad_frac,
                y_pad_frac=args.container_region_y_pad_frac,
                y0_frac=args.container_region_y0_frac,
                y1_frac=args.container_region_y1_frac,
            )
            if container_box is not None
            else None
        )
        score = _score_ball_against_container(ball_box, container_region)
        inside = _inside_container_region(
            score,
            min_overlap=args.min_ball_overlap,
            mode=args.containment_mode,
        )
        event = {
            "fixed_frame_idx": frame_idx,
            "fixed_frame": ball_event.frame if ball_event is not None else f"frame_{frame_idx:05d}.jpg",
            "ball_box_xyxy": ball_box.as_list() if ball_box is not None else None,
            "ball_center_xy": list(ball_box.center) if ball_box is not None else None,
            "container_box_xyxy": container_box.as_list() if container_box is not None else None,
            "container_region_xyxy": container_region.as_list() if container_region is not None else None,
            "container_box_source": container_source,
            "ball_center_in_container_region": score["ball_center_in_container_region"],
            "ball_overlap_container_region": score["ball_overlap_container_region"],
            "inside_container_region": inside,
            "is_selected_check_frame": False,
        }
        window_events.append(event)
        if selected_event is None:
            if args.select_frame == "first-frame" or ball_box is not None:
                selected_event = event

    if selected_event is None:
        result = {
            "status": "unknown",
            "reason": "no fixed-camera ball observation in the check window",
            "selected_fixed_frame": None,
            "selected_fixed_frame_idx": None,
            "container_box_source": None,
            "container_region_xyxy": None,
            "ball_box_xyxy": None,
            "ball_center_in_container_region": False,
            "ball_overlap_container_region": 0.0,
        }
    else:
        selected_event["is_selected_check_frame"] = True
        has_ball = selected_event["ball_box_xyxy"] is not None
        has_container = selected_event["container_region_xyxy"] is not None
        containment_passed = bool(
            has_ball and has_container and selected_event["inside_container_region"]
        )
        pre_drop_above = bool(pre_drop_event and pre_drop_event["above_cylinder_area"])
        if args.require_pre_drop_above and pre_drop_event is None:
            reason = "no fixed-camera ball observation before release to verify above-container precondition"
            status = "unknown"
        elif args.require_pre_drop_above and not pre_drop_above:
            reason = "pre-drop fixed-camera ball observation is not above the container area"
            status = "failed"
        elif containment_passed:
            reason = "first fixed-camera ball observation after release is inside the container region"
            status = "passed"
        elif not has_ball:
            reason = "selected fixed-camera frame has no ball box"
            status = "unknown"
        elif not has_container:
            reason = "selected fixed-camera frame has no container region"
            status = "unknown"
        else:
            reason = "first fixed-camera ball observation after release is outside the container region"
            status = "failed"
        result = {
            "status": status,
            "reason": reason,
            "pre_drop": pre_drop_event,
            "pre_drop_above_cylinder_area": pre_drop_above,
            "selected_fixed_frame": selected_event["fixed_frame"],
            "selected_fixed_frame_idx": selected_event["fixed_frame_idx"],
            "container_box_source": selected_event["container_box_source"],
            "container_region_xyxy": selected_event["container_region_xyxy"],
            "ball_box_xyxy": selected_event["ball_box_xyxy"],
            "ball_center_in_container_region": selected_event["ball_center_in_container_region"],
            "ball_overlap_container_region": selected_event["ball_overlap_container_region"],
        }

    payload = {
        "fixed_ball_summary": str(fixed_ball_summary_path),
        "wrist_ball_summary": args.wrist_ball_summary or None,
        "container_summary": args.container_summary or None,
        "release": release,
        "fixed_check": {
            "sync_offset_fixed_minus_wrist": args.sync_offset_fixed_minus_wrist,
            "fixed_delay_frames": args.fixed_delay_frames,
            "window_start_idx": window_start,
            "window_end_idx": window_end,
            "select_frame": args.select_frame,
        },
        "criteria": {
            "container_box_mode": args.container_box_mode,
            "container_memory_frames": args.container_memory_frames,
            "container_region_y0_frac": args.container_region_y0_frac,
            "container_region_y1_frac": args.container_region_y1_frac,
            "container_region_x_pad_frac": args.container_region_x_pad_frac,
            "container_region_y_pad_frac": args.container_region_y_pad_frac,
            "min_ball_overlap": args.min_ball_overlap,
            "containment_mode": args.containment_mode,
            "require_pre_drop_above": args.require_pre_drop_above,
            "above_mode": args.above_mode,
            "above_window_before": args.above_window_before,
        },
        "result": result,
        "window_events": window_events,
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
            frames_dir=Path(args.fixed_frames_dir),
            overlay_dir=Path(args.overlay_dir),
            events=window_events,
        )

    print(text_report)
    print(json.dumps({k: v for k, v in payload.items() if k != "window_events"}, indent=2))


if __name__ == "__main__":
    main()
