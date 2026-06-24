#!/usr/bin/env python3
"""Plot first-frame blue-ball positions for successful vs failed SO-101 episodes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BLUE_LOWER_HSV = np.array([85, 45, 35], dtype=np.uint8)
BLUE_UPPER_HSV = np.array([140, 255, 255], dtype=np.uint8)
MIN_BALL_AREA = 80
MAX_BALL_AREA = 12000
SUCCESS_COLOR = (80, 220, 80)
FAILURE_COLOR = (80, 80, 255)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return rows


def _episode_dirs(root: Path, include_failures: bool) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        if not path.is_dir() or not (path / "episode_meta.json").exists():
            continue
        result = _read_json(path / "episode_result.json")
        outcome = str(result.get("outcome") or _read_json(path / "episode_meta.json").get("outcome") or "")
        if outcome == "success" or (include_failures and outcome == "failure"):
            out.append(path)
    return out


def _detect_blue_ball(image_bgr: np.ndarray) -> dict[str, Any] | None:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER_HSV, BLUE_UPPER_HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_label = 0
    best_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if MIN_BALL_AREA <= area <= MAX_BALL_AREA and area > best_area:
            best_label = label
            best_area = area
    if best_label == 0:
        return None
    x, y = centroids[best_label]
    left = int(stats[best_label, cv2.CC_STAT_LEFT])
    top = int(stats[best_label, cv2.CC_STAT_TOP])
    width = int(stats[best_label, cv2.CC_STAT_WIDTH])
    height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    return {
        "x": float(x),
        "y": float(y),
        "area": best_area,
        "box_xyxy": [left, top, left + width, top + height],
    }


def _first_frame(episodes: list[Path], camera: str) -> np.ndarray | None:
    for episode in episodes:
        rows = _read_jsonl(episode / camera / "frames.jsonl")
        for row in rows:
            frame = row.get("frame")
            if not frame:
                continue
            image = cv2.imread(str(episode / camera / str(frame)))
            if image is not None:
                return image
    return None


def _draw_legend(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (12, 12), (430, 96), (0, 0, 0), thickness=-1)
    cv2.putText(image, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image, (34, 62), 6, SUCCESS_COLOR, thickness=-1)
    cv2.putText(image, "success first frame", (52, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.circle(image, (34, 84), 6, FAILURE_COLOR, thickness=-1)
    cv2.putText(image, "failure first frame", (52, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)


def _draw_points(
    background: np.ndarray,
    points: list[dict[str, Any]],
    title: str,
) -> np.ndarray:
    canvas = cv2.addWeighted(background, 0.52, np.zeros_like(background), 0.48, 0)
    ordered_points = sorted(points, key=lambda point: point["outcome"] == "failure")
    for point in ordered_points:
        color = SUCCESS_COLOR if point["outcome"] == "success" else FAILURE_COLOR
        center = (round(point["x"]), round(point["y"]))
        cv2.circle(canvas, center, 10, (255, 255, 255), thickness=2)
        cv2.circle(canvas, center, 7, color, thickness=-1)
        label = f"{'S' if point['outcome'] == 'success' else 'F'}{point['episode_num']}"
        cv2.putText(canvas, label, (center[0] + 12, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    _draw_legend(canvas, title)
    return canvas


def _write_html(output_dir: Path, cameras: list[str], summary: dict[str, Any]) -> None:
    items = []
    for camera in cameras:
        items.append(
            f"""
            <section>
              <h2>{camera}</h2>
              <img src="{camera}_first_frame_map.jpg" alt="{camera} first-frame map">
            </section>
            """
        )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SO101 Ball Success/Failure Map</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; background:#111; color:#eee; }}
main {{ padding: 16px; display: grid; gap: 18px; }}
section {{ display: grid; grid-template-columns: 1fr; gap: 12px; align-items: start; }}
h1, h2 {{ grid-column: 1 / -1; margin: 0; }}
img {{ width: 100%; border: 1px solid #333; border-radius: 6px; background:#000; }}
pre {{ background:#0b0b0b; border:1px solid #333; border-radius:6px; padding:12px; overflow:auto; }}
@media (max-width: 900px) {{ section {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>SO101 Ball Success/Failure Map</h1>
<pre>{json.dumps(summary, indent=2)}</pre>
{''.join(items)}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", default="episodes")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--camera", action="append", default=[], help="Camera to analyze. Default: cam0 and cam1.")
    parser.add_argument("--include-failures", action="store_true", default=True)
    args = parser.parse_args()

    episodes_root = Path(args.episodes_root)
    cameras = args.camera or ["cam0", "cam1"]
    output_dir = Path(args.output_dir) if args.output_dir else Path("reports") / "so101_ball_maps" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _episode_dirs(episodes_root, include_failures=args.include_failures)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "episodes_root": str(episodes_root),
        "output_dir": str(output_dir),
        "episode_count": len(episodes),
        "cameras": cameras,
        "outcomes": {"success": 0, "failure": 0},
        "detections": {},
    }
    for episode_num, episode in enumerate(episodes, start=1):
        result = _read_json(episode / "episode_result.json")
        outcome = str(result.get("outcome") or _read_json(episode / "episode_meta.json").get("outcome") or "")
        if outcome not in {"success", "failure"}:
            continue
        summary["outcomes"][outcome] += 1
        for camera in cameras:
            frames = _read_jsonl(episode / camera / "frames.jsonl")
            if not frames:
                continue
            row = frames[0]
            frame_name = str(row.get("frame") or "")
            if not frame_name:
                continue
            image = cv2.imread(str(episode / camera / frame_name))
            if image is None:
                continue
            detection = _detect_blue_ball(image)
            if detection is None:
                continue
            all_rows.append(
                {
                    "episode": episode.name,
                    "episode_num": episode_num,
                    "outcome": outcome,
                    "camera": camera,
                    "frame": frame_name,
                    "frame_idx": int(row.get("frame_idx") or 0),
                    "timestamp_s": row.get("timestamp_s"),
                    **detection,
                }
            )

    with (output_dir / "ball_positions.csv").open("w", newline="") as f:
        fieldnames = ["episode", "episode_num", "outcome", "camera", "frame", "frame_idx", "timestamp_s", "x", "y", "area", "box_xyxy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    for camera in cameras:
        points = [row for row in all_rows if row["camera"] == camera]
        background = _first_frame(episodes, camera)
        if background is None:
            background = np.zeros((480, 640, 3), dtype=np.uint8)
        summary["detections"][camera] = {
            "points": len(points),
            "success_points": sum(1 for row in points if row["outcome"] == "success"),
            "failure_points": sum(1 for row in points if row["outcome"] == "failure"),
        }
        cv2.imwrite(
            str(output_dir / f"{camera}_first_frame_map.jpg"),
            _draw_points(background, points, f"{camera}: first-frame ball positions"),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_html(output_dir, cameras, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
