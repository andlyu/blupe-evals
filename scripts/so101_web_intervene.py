#!/usr/bin/env python3
"""SO101 web control surface with policy stop and manual joint intervention.

Runs on the Jetson. It uses the existing camera MJPEG server on :8080 and the
Jetson-local MolmoAct tunnel on :8202.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blupe_evals.station import CameraConfig, HttpPolicyClient, default_camera_configs, parse_camera_config_specs
from blupe_evals.station.joint_conventions import (
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
    load_joint_array_env,
    policy_action_to_robot_target as convert_policy_action_to_robot_target,
    robot_state_to_policy_state as convert_robot_state_to_policy_state,
    validate_policy_to_robot_signs,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SEMANTIC_CAMERA_NAMES = ("front", "side", "wrist")
DEFAULT_POLICY_CAMERA_NAMES = tuple(
    name.strip()
    for name in os.environ.get("SO101_POLICY_CAMERAS", "front,side").split(",")
    if name.strip()
)
DEFAULT_INSTRUCTION = "Move to blue ball, grab it, and place it in the cardboard cylinder"
DEFAULT_DURATION_S = 300.0
DEFAULT_EXEC_STEPS = 30
DEFAULT_MAX_STEP_DEG = 1.0
POLICY_HARD_MAX_STEP_DEG = float(os.environ.get("SO101_POLICY_HARD_MAX_STEP_DEG", "2.0"))
DEFAULT_HZ = 30.0
MOLMO_TIMEOUT_S = 75.0
RECORD_ROOT = REPO_ROOT / "episodes"
RAW_RECORD_ROOT = REPO_ROOT / "recordings"
DEFAULT_LEROBOT_DATASET_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_RECORD_DURATION_S = 120.0
DEFAULT_RECORD_FPS = 30.0
DEFAULT_EVAL_RUN_DURATION_S = 3600.0
DEFAULT_EVAL_ATTEMPT_DURATION_S = 60.0
DEFAULT_EVAL_MAX_CONSECUTIVE_FAILURES = 4
DEFAULT_EVAL_RECORD_FPS = 30.0
DEFAULT_EVAL_HOME_TIMEOUT_S = 90.0
DEFAULT_EVAL_HOME_S = 5.0
DEFAULT_HOME_POSE_DEG = np.array([0.0, -90.0, 70.0, 0.0, -45.0, 0.0], dtype=np.float32)
HOME_POSE_PATH = REPO_ROOT / "config" / "so101_home.json"
HOME_EPS_DEG = 1.0
HOME_STEP_DEG = 5.0
DEFAULT_SUCCESS_FPS = 10.0
DEFAULT_STATUS_LOG_LIMIT = 60
MAX_STATUS_LOG_LIMIT = 150
TELEOP_LEASE_TIMEOUT_S = 30.0
SUCCESS_OVERLAP_THRESHOLD = 0.9
SUCCESS_LEAVE_THRESHOLD = 0.1
SUCCESS_MIN_OVER_FRAMES = 1
SUCCESS_MIN_BALL_AREA = 80
SUCCESS_MAX_BALL_AREA = 12000
SUCCESS_BLUE_LOWER_HSV = np.array([85, 45, 35], dtype=np.uint8)
SUCCESS_BLUE_UPPER_HSV = np.array([140, 255, 255], dtype=np.uint8)
SUCCESS_CUP_SEARCH_PAD_PX = 72
SUCCESS_CUP_MIN_MASK_AREA = 1500
SUCCESS_CUP_MAX_MASK_AREA_MULT = 4.5
SUCCESS_CONTAINER_SAM3_URL = os.environ.get("SO101_SUCCESS_SAM3_URL", "http://127.0.0.1:8213/api/detect_image")
SUCCESS_CONTAINER_SAM3_PROMPT = os.environ.get(
    "SO101_SUCCESS_SAM3_PROMPT",
    "cardboard cylinder along with the insides",
)
SUCCESS_CONTAINER_SAM3_MIN_SCORE = float(os.environ.get("SO101_SUCCESS_SAM3_MIN_SCORE", "0.15"))
SUCCESS_CONTAINER_SAM3_TIMEOUT_S = float(os.environ.get("SO101_SUCCESS_SAM3_TIMEOUT_S", "15"))
DEFAULT_CUP_POLYGON = np.array(
    [
        [464, 288],
        [488, 324],
        [480, 374],
        [439, 414],
        [420, 420],
        [391, 418],
        [373, 406],
        [365, 349],
        [376, 304],
        [391, 287],
        [430, 276],
    ],
    dtype=np.int32,
)


DEFAULT_CAMERA_CONFIGS = default_camera_configs(SEMANTIC_CAMERA_NAMES)


POLICY_TO_ROBOT_JOINT_SIGNS = load_joint_array_env(
    "SO101_POLICY_TO_ROBOT_JOINT_SIGNS",
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
    joint_count=len(JOINTS),
)
POLICY_TO_ROBOT_JOINT_OFFSETS_DEG = load_joint_array_env(
    "SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG",
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    joint_count=len(JOINTS),
)
validate_policy_to_robot_signs(
    POLICY_TO_ROBOT_JOINT_SIGNS,
    env_name="SO101_POLICY_TO_ROBOT_JOINT_SIGNS",
)


def policy_step_limit(requested_max_step_deg: float) -> tuple[float, bool]:
    requested = float(requested_max_step_deg)
    if not np.isfinite(requested) or requested <= 0:
        requested = DEFAULT_MAX_STEP_DEG
    hard_cap = max(0.0, float(POLICY_HARD_MAX_STEP_DEG))
    if hard_cap <= 0:
        return requested, False
    return min(requested, hard_cap), requested > hard_cap


def robot_state_to_policy_state(state: np.ndarray) -> np.ndarray:
    return convert_robot_state_to_policy_state(
        state,
        policy_to_robot_signs=POLICY_TO_ROBOT_JOINT_SIGNS,
        policy_to_robot_offsets_deg=POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    )


def policy_action_to_robot_target(action: np.ndarray) -> np.ndarray:
    return convert_policy_action_to_robot_target(
        action,
        policy_to_robot_signs=POLICY_TO_ROBOT_JOINT_SIGNS,
        policy_to_robot_offsets_deg=POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    )


def _fmt(values: np.ndarray) -> str:
    return "  ".join(f"{x:8.2f}" for x in values)


def load_home_pose() -> np.ndarray:
    if not HOME_POSE_PATH.exists():
        return DEFAULT_HOME_POSE_DEG.copy()
    data = json.loads(HOME_POSE_PATH.read_text())
    raw = data.get("home", data) if isinstance(data, dict) else data
    pose = np.asarray(raw, dtype=np.float32).reshape(-1)
    if pose.shape != (len(JOINTS),):
        raise ValueError(f"{HOME_POSE_PATH} must contain {len(JOINTS)} home values, got {pose.shape}")
    return pose


def save_home_pose(pose: np.ndarray) -> None:
    pose = np.asarray(pose, dtype=np.float32).reshape(len(JOINTS))
    HOME_POSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joints": JOINTS,
        "home": [float(x) for x in pose],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    HOME_POSE_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def read_mjpeg_jpeg(url: str, timeout_s: float = 5.0) -> bytes:
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    with requests.get(url, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                buf.extend(chunk)
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
                if start >= 0 and end >= 0:
                    return bytes(buf[start : end + 2])
            if time.monotonic() > deadline:
                break
    raise TimeoutError(f"no MJPEG frame from {url}")


def read_mjpeg_frame(url: str, timeout_s: float = 5.0) -> np.ndarray:
    jpg = np.frombuffer(read_mjpeg_jpeg(url, timeout_s=timeout_s), dtype=np.uint8)
    bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to decode JPEG from {url}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


class LiveCupSuccessTracker:
    def __init__(
        self,
        cup_polygon: np.ndarray = DEFAULT_CUP_POLYGON,
        overlap_threshold: float = SUCCESS_OVERLAP_THRESHOLD,
        leave_threshold: float = SUCCESS_LEAVE_THRESHOLD,
        min_over_frames: int = SUCCESS_MIN_OVER_FRAMES,
    ):
        self.cup_polygon = np.asarray(cup_polygon, dtype=np.int32)
        self.overlap_threshold = float(overlap_threshold)
        self.leave_threshold = float(leave_threshold)
        self.min_over_frames = int(min_over_frames)
        self.container_sam3_url = SUCCESS_CONTAINER_SAM3_URL
        self.container_sam3_prompt = SUCCESS_CONTAINER_SAM3_PROMPT
        self.container_sam3_min_score = SUCCESS_CONTAINER_SAM3_MIN_SCORE
        self.container_sam3_timeout_s = SUCCESS_CONTAINER_SAM3_TIMEOUT_S
        self.cup_mask: np.ndarray | None = None
        self.cup_mask_source = "pending"
        self.cup_mask_generation = 0
        self.cup_mask_area = 0
        self.cup_mask_box_xyxy: list[int] | None = None
        self.cup_mask_calculated_at_frame: int | None = None
        self.cup_mask_prompt = self.container_sam3_prompt
        self.cup_mask_score: float | None = None
        self.cup_mask_min_score = self.container_sam3_min_score
        self.cup_mask_sam3_status = "not_run"
        self.cup_mask_sam3_error = ""
        self.cup_mask_sam3_raw_score: float | None = None
        self.cup_mask_sam3_raw_area: int | None = None
        self.cup_mask_sam3_box_xyxy: list[float] | None = None
        self.reset()

    def configure_sam3(self, prompt: str | None = None, min_score: float | None = None) -> None:
        if prompt is not None:
            prompt = prompt.strip()
            if not prompt:
                raise ValueError("SAM3 prompt cannot be empty")
            self.container_sam3_prompt = prompt
            self.cup_mask_prompt = prompt
        if min_score is not None:
            if min_score < 0.0 or min_score > 1.0:
                raise ValueError("SAM3 confidence must be between 0 and 1")
            self.container_sam3_min_score = float(min_score)
            self.cup_mask_min_score = float(min_score)

    def reset(self, recalculate_container: bool = False, container_reason: str = "new_run") -> None:
        self.frame_idx = 0
        self.state = "waiting"
        self.success_count = 0
        self.current_overlap: float | None = None
        self.ball_area = 0
        self.over_active = False
        self.over_start_frame: int | None = None
        self.over_len = 0
        self.last_success: dict[str, Any] | None = None
        self.last_event = "reset"
        if recalculate_container:
            self.invalidate_container_mask(container_reason)

    def invalidate_container_mask(self, reason: str = "manual") -> None:
        self.cup_mask = None
        self.cup_mask_source = f"pending:{reason}"
        self.cup_mask_area = 0
        self.cup_mask_box_xyxy = None
        self.cup_mask_calculated_at_frame = None
        self.cup_mask_score = None
        self.cup_mask_sam3_status = f"pending:{reason}"
        self.cup_mask_sam3_error = ""
        self.cup_mask_sam3_raw_score = None
        self.cup_mask_sam3_raw_area = None
        self.cup_mask_sam3_box_xyxy = None
        self.cup_mask_generation += 1
        self.last_event = f"container mask pending gen={self.cup_mask_generation}"

    def _default_cup_mask_for(self, shape: tuple[int, int, int]) -> np.ndarray:
        height, width = shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.cup_polygon.reshape(-1, 1, 2)], 255)
        return mask > 0

    @staticmethod
    def _mask_box(mask: np.ndarray) -> list[int] | None:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _set_sam3_mask_status(self, status: str, error: str = "") -> None:
        self.cup_mask_sam3_status = status
        self.cup_mask_sam3_error = error

    def _calculate_sam3_cup_mask(self, rgb: np.ndarray, default_area: int) -> tuple[np.ndarray, str] | None:
        if not self.container_sam3_url or not self.container_sam3_prompt:
            self._set_sam3_mask_status("disabled")
            return None
        self._set_sam3_mask_status("requesting")
        self.cup_mask_sam3_raw_score = None
        self.cup_mask_sam3_raw_area = None
        self.cup_mask_sam3_box_xyxy = None

        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            self._set_sam3_mask_status("encode_failed")
            return None

        payload = {
            "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "prompts": [self.container_sam3_prompt],
            "max_masks": 1,
            "min_score": self.container_sam3_min_score,
            "alpha": 0.65,
        }
        try:
            resp = requests.post(
                self.container_sam3_url,
                json=payload,
                timeout=self.container_sam3_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            top = data.get("top_mask") or {}
            if not top:
                self._set_sam3_mask_status("no_top_mask")
                return None
            score = float(top.get("score", 0.0))
            self.cup_mask_sam3_raw_score = score
            box = top.get("box_xyxy")
            if isinstance(box, list):
                self.cup_mask_sam3_box_xyxy = [float(x) for x in box]
            if score < self.container_sam3_min_score:
                self._set_sam3_mask_status(
                    "low_score",
                    f"score={score:.3f} min={self.container_sam3_min_score:.3f}",
                )
                return None

            mask_b64 = top.get("mask_png_b64")
            if not mask_b64:
                self._set_sam3_mask_status("no_mask_payload", f"score={score:.3f}")
                return None
            if "," in mask_b64:
                mask_b64 = mask_b64.split(",", 1)[1]
            mask_bytes = base64.b64decode(mask_b64)
            mask_u8 = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                self._set_sam3_mask_status("mask_decode_failed", f"score={score:.3f}")
                return None
            if mask_u8.shape != rgb.shape[:2]:
                mask_u8 = cv2.resize(mask_u8, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_bool = mask_u8 > 0
            area = int(mask_bool.sum())
            self.cup_mask_sam3_raw_area = area
            max_area = int(max(default_area, 1) * SUCCESS_CUP_MAX_MASK_AREA_MULT)
            if area < SUCCESS_CUP_MIN_MASK_AREA:
                self._set_sam3_mask_status(
                    "area_too_small",
                    f"area={area} min={SUCCESS_CUP_MIN_MASK_AREA} score={score:.3f}",
                )
                return None
            if area > max_area:
                self._set_sam3_mask_status(
                    "area_too_large",
                    f"area={area} max={max_area} score={score:.3f}",
                )
                return None

            prompt = str(top.get("prompt") or self.container_sam3_prompt)
            self.cup_mask_prompt = prompt
            self.cup_mask_score = score
            self._set_sam3_mask_status("accepted", f"area={area} score={score:.3f}")
            return mask_bool, f"sam3:{prompt} score={score:.2f}"
        except Exception as exc:
            self._set_sam3_mask_status("request_failed", str(exc))
            return None

    def _cup_mask_from_candidate(
        self,
        candidate: np.ndarray,
        source: str,
        x0: int,
        y0: int,
        height: int,
        width: int,
        default_area: int,
    ) -> tuple[np.ndarray, str] | None:
        candidate_u8 = candidate.astype(np.uint8) * 255
        if int(candidate_u8.sum()) == 0:
            return None

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, close_kernel)
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, open_kernel)
        candidate_u8 = cv2.dilate(candidate_u8, open_kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)

        selected = np.zeros(candidate_u8.shape, dtype=np.uint8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 35:
                continue
            selected[labels == label] = 255

        ys, xs = np.where(selected > 0)
        if len(xs) == 0:
            return None

        points = np.column_stack([xs + x0, ys + y0]).astype(np.int32)
        hull = cv2.convexHull(points)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        mask_bool = mask > 0
        area = int(mask_bool.sum())
        max_area = int(max(default_area, 1) * SUCCESS_CUP_MAX_MASK_AREA_MULT)
        if area < SUCCESS_CUP_MIN_MASK_AREA or area > max_area:
            return None

        return mask_bool, source

    def _calculate_cup_mask(self, rgb: np.ndarray) -> tuple[np.ndarray, str]:
        default_mask = self._default_cup_mask_for(rgb.shape)
        default_area = int(default_mask.sum())
        sam3_result = self._calculate_sam3_cup_mask(rgb, default_area)
        if sam3_result is not None:
            return sam3_result

        height, width = rgb.shape[:2]
        x, y, w, h = cv2.boundingRect(self.cup_polygon.reshape(-1, 1, 2))
        pad = SUCCESS_CUP_SEARCH_PAD_PX
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return default_mask, "default_polygon_bad_roi"

        prior = cv2.dilate(
            default_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
        )[y0:y1, x0:x1] > 0
        crop = rgb[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        crop_i = crop.astype(np.int16)
        red = crop_i[:, :, 0]
        green = crop_i[:, :, 1]
        blue = crop_i[:, :, 2]
        warm_rgb = (red >= blue - 4) & (green >= blue - 18) & ((red + green) >= (blue * 2 - 20))
        not_blue = ~((blue > red + 18) & (blue > green + 18))
        not_highlight = ~((sat <= 10) & (val >= 225))

        # The cup is cardboard/brown and can be partly occluded by the blue ball.
        # Try strict hue first, then relax for dim/low-saturation frames.
        # Each pass detects visible cardboard/rim pixels and fills their hull so
        # overlap uses the inferred container area, not just the visible rim.
        candidates = [
            (
                "live_cardboard_hull_strict",
                (((hue <= 42) | (hue >= 165)) & (sat >= 12) & (val >= 18) & (val <= 252) & prior),
            ),
            (
                "live_cardboard_hull_relaxed",
                (((hue <= 55) | (hue >= 155)) & (sat >= 6) & (val >= 16) & not_blue & not_highlight & prior),
            ),
            (
                "live_cardboard_hull_rgb",
                ((val >= 18) & (val <= 252) & warm_rgb & not_blue & not_highlight & prior),
            ),
        ]
        for source, candidate in candidates:
            result = self._cup_mask_from_candidate(candidate, source, x0, y0, height, width, default_area)
            if result is not None:
                return result

        return default_mask, "default_polygon_no_cardboard"

    def _cup_mask_for(self, rgb: np.ndarray) -> np.ndarray:
        if self.cup_mask is None or self.cup_mask.shape != rgb.shape[:2]:
            self.cup_mask, self.cup_mask_source = self._calculate_cup_mask(rgb)
            self.cup_mask_area = int(self.cup_mask.sum())
            self.cup_mask_box_xyxy = self._mask_box(self.cup_mask)
            self.cup_mask_calculated_at_frame = self.frame_idx
            sam3_suffix = ""
            if not str(self.cup_mask_source).startswith("sam3:"):
                sam3_suffix = f" sam3={self.cup_mask_sam3_status}"
                if self.cup_mask_sam3_error:
                    sam3_suffix += f" ({self.cup_mask_sam3_error})"
            self.last_event = (
                f"container mask gen={self.cup_mask_generation} "
                f"source={self.cup_mask_source} area={self.cup_mask_area}{sam3_suffix}"
            )
        return self.cup_mask

    def _ball_mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, SUCCESS_BLUE_LOWER_HSV, SUCCESS_BLUE_UPPER_HSV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best_label = 0
        best_area = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if SUCCESS_MIN_BALL_AREA <= area <= SUCCESS_MAX_BALL_AREA and area > best_area:
                best_area = area
                best_label = label
        if best_label == 0:
            return np.zeros(mask.shape, dtype=bool)
        return labels == best_label

    def update(self, rgb: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        self.frame_idx += 1
        cup_mask = self._cup_mask_for(rgb)
        ball_mask = self._ball_mask(rgb)
        self.ball_area = int(ball_mask.sum())
        visible = self.ball_area > 0
        overlap = None
        if visible:
            overlap = float(np.logical_and(ball_mask, cup_mask).sum() / self.ball_area)
        self.current_overlap = overlap

        success_this_frame = False
        if visible and overlap is not None and overlap >= self.overlap_threshold:
            if not self.over_active:
                self.over_active = True
                self.over_start_frame = self.frame_idx
                self.over_len = 0
                self.last_event = f"over start frame={self.frame_idx}"
            self.over_len += 1
            self.state = "over_cup"
        elif (
            self.over_active
            and visible
            and overlap is not None
            and overlap <= self.leave_threshold
            and self.over_len >= self.min_over_frames
        ):
            self.success_count += 1
            success_this_frame = True
            self.last_success = {
                "success": self.success_count,
                "over_start_frame": self.over_start_frame,
                "over_len": self.over_len,
                "leave_frame": self.frame_idx,
                "leave_overlap": overlap,
                "time": time.strftime("%H:%M:%S"),
            }
            self.last_event = (
                f"SUCCESS {self.success_count}: over_start={self.over_start_frame} "
                f"len={self.over_len} leave={self.frame_idx} overlap={overlap:.3f}"
            )
            self.over_active = False
            self.over_start_frame = None
            self.over_len = 0
            self.state = "success"
        elif self.over_active:
            self.state = "watch_leave" if visible else "over_missing"
        else:
            self.state = "tracking" if visible else "missing_ball"

        status = self.status()
        status["visible"] = visible
        status["success_this_frame"] = success_this_frame
        return status, self.overlay(rgb, cup_mask, ball_mask, success_this_frame)

    def overlay(
        self,
        rgb: np.ndarray,
        cup_mask: np.ndarray,
        ball_mask: np.ndarray,
        success_this_frame: bool,
    ) -> np.ndarray:
        out = rgb.copy().astype(np.float32)
        out[cup_mask] = 0.68 * out[cup_mask] + 0.32 * np.array([255, 190, 0], dtype=np.float32)
        if ball_mask.any():
            if success_this_frame:
                ball_color = np.array([255, 80, 80], dtype=np.float32)
            elif self.over_active or self.state == "over_cup":
                ball_color = np.array([64, 255, 96], dtype=np.float32)
            else:
                ball_color = np.array([32, 170, 255], dtype=np.float32)
            out[ball_mask] = 0.45 * out[ball_mask] + 0.55 * ball_color

        bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if ball_mask.any():
            contours, _ = cv2.findContours(ball_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (80, 80, 255) if success_this_frame else (96, 255, 64) if self.over_active else (255, 170, 32)
            cv2.drawContours(bgr, contours, -1, color, thickness=2)
        overlap_text = "missing" if self.current_overlap is None else f"overlap={self.current_overlap:.2f}"
        if success_this_frame:
            label = f"SUCCESS {self.success_count}: LEFT CUP"
        elif self.over_active:
            label = f"OVER CUP {self.over_len}f {overlap_text}"
        else:
            label = f"{self.state} {overlap_text}"
        cv2.rectangle(bgr, (8, 8), (430, 76), (0, 0, 0), thickness=-1)
        cv2.putText(bgr, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            f"successes={self.success_count} ball_area={self.ball_area}",
            (14, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        mask_source = str(self.cup_mask_source)[:42]
        cv2.putText(
            bgr,
            f"mask={mask_source} area={self.cup_mask_area}",
            (14, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cup_contours, _ = cv2.findContours(cup_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cup_contours, -1, (0, 210, 255), thickness=2)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "frame_idx": self.frame_idx,
            "success_count": self.success_count,
            "current_overlap": self.current_overlap,
            "ball_area": self.ball_area,
            "over_active": self.over_active,
            "over_start_frame": self.over_start_frame,
            "over_len": self.over_len,
            "last_success": self.last_success,
            "last_event": self.last_event,
            "overlap_threshold": self.overlap_threshold,
            "leave_threshold": self.leave_threshold,
            "min_over_frames": self.min_over_frames,
            "container_mask_source": self.cup_mask_source,
            "container_mask_generation": self.cup_mask_generation,
            "container_mask_area": self.cup_mask_area,
            "container_mask_box_xyxy": self.cup_mask_box_xyxy,
            "container_mask_calculated_at_frame": self.cup_mask_calculated_at_frame,
            "container_mask_prompt": self.cup_mask_prompt,
            "container_mask_score": self.cup_mask_score,
            "container_mask_min_score": self.cup_mask_min_score,
            "container_mask_sam3_status": self.cup_mask_sam3_status,
            "container_mask_sam3_error": self.cup_mask_sam3_error,
            "container_mask_sam3_raw_score": self.cup_mask_sam3_raw_score,
            "container_mask_sam3_raw_area": self.cup_mask_sam3_raw_area,
            "container_mask_sam3_box_xyxy": self.cup_mask_sam3_box_xyxy,
        }


class SO101Controller:
    def __init__(
        self,
        port: str,
        robot_id: str,
        molmo_url: str,
        leader_port: str = "/dev/ttyACM1",
        leader_id: str = "blupe_leader",
        camera_configs: list[CameraConfig] | None = None,
        policy_camera_names: list[str] | tuple[str, ...] | None = None,
        success_enabled: bool = True,
        success_fps: float = DEFAULT_SUCCESS_FPS,
    ):
        self.port = port
        self.robot_id = robot_id
        self.leader_port = leader_port
        self.leader_id = leader_id
        self.molmo_url = molmo_url
        self.camera_configs = list(camera_configs or DEFAULT_CAMERA_CONFIGS)
        self.cameras_by_name = {cam.name: cam for cam in self.camera_configs}
        self.cameras_by_id = {cam.id: cam for cam in self.camera_configs}
        self.policy_camera_names = tuple(policy_camera_names or DEFAULT_POLICY_CAMERA_NAMES)
        self.success_enabled = success_enabled
        self.success_fps = success_fps
        self.robot: SO101Follower | None = None
        self.leader: SO101Leader | None = None
        self.robot_lock = threading.RLock()
        self.leader_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.policy_thread: threading.Thread | None = None
        self.success_stop_event = threading.Event()
        self.success_thread: threading.Thread | None = None
        self.success_tracker = LiveCupSuccessTracker()
        self.success_tracker_lock = threading.Lock()
        self.success_condition = threading.Condition()
        self.success_overlay_jpeg: bytes | None = None
        self.success_status: dict[str, Any] = self.success_tracker.status()
        self.success_error = ""
        self.status_lock = threading.Lock()
        self.mode = "idle"
        self.stage = "idle"
        self.last_error = ""
        self.last_state: list[float] | None = None
        self.last_leader_state: list[float] | None = None
        self.last_query_s: float | None = None
        self.started_at: float | None = None
        self.steps = 0
        self.chunks = 0
        self.logs: deque[str] = deque(maxlen=200)
        self.home_pose = load_home_pose()
        self.policy_client = HttpPolicyClient(molmo_url, timeout_s=MOLMO_TIMEOUT_S)
        self.record_stop_event = threading.Event()
        self.record_thread: threading.Thread | None = None
        self.record_started_at: float | None = None
        self.record_duration_s: float | None = None
        self.record_fps: float | None = None
        self.record_dir: Path | None = None
        self.record_cameras: list[str] = []
        self.record_counts: dict[str, int] = {}
        self.record_error = ""
        self.record_capture_mode = "policy_execute"
        self.last_action: list[float] | None = None
        self.last_action_at_mono: float | None = None
        self.eval_stop_event = threading.Event()
        self.eval_thread: threading.Thread | None = None
        self.eval_history: list[dict[str, Any]] = []
        self.eval_started_at_mono: float | None = None
        self.eval_completed_at_mono: float | None = None
        self.eval_deadline_mono: float | None = None
        self.eval_attempt_started_at_mono: float | None = None
        self.eval_attempt_deadline_mono: float | None = None
        self.eval_summary_path: Path | None = None
        self.eval_config: dict[str, Any] = {}
        self.eval_status: dict[str, Any] = self._new_eval_status()
        self.teleop_lease: dict[str, Any] | None = None
        self.dataset_thread: threading.Thread | None = None
        self.dataset_status: dict[str, Any] = self._new_dataset_status()
        self.hf_teleop_process: subprocess.Popen[str] | None = None
        self.hf_teleop_thread: threading.Thread | None = None
        self.hf_teleop_status: dict[str, Any] = self._new_hf_teleop_status()

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.status_lock:
            self.logs.append(line)

    def set_mode(self, mode: str) -> None:
        with self.status_lock:
            self.mode = mode

    def set_stage(self, stage: str) -> None:
        with self.status_lock:
            self.stage = stage

    def set_error(self, exc: BaseException | str) -> None:
        msg = str(exc)
        with self.status_lock:
            self.last_error = msg
        self.log(f"ERROR {msg}")

    def _new_eval_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "state": "idle",
            "started_at": None,
            "resumed_at": None,
            "completed_at": None,
            "run_duration_s": None,
            "attempt_duration_s": None,
            "no_success_timeout_s": None,
            "attempt": 0,
            "successes": 0,
            "failures": 0,
            "consecutive_failures": 0,
            "max_consecutive_failures": DEFAULT_EVAL_MAX_CONSECUTIVE_FAILURES,
            "allow_teleop": True,
            "record_episodes": True,
            "waiting_for_intervention": False,
            "current_attempt": None,
            "last_attempt": None,
            "last_success": None,
            "last_success_at": None,
            "recording_dir": None,
            "summary_path": None,
            "stop_reason": "",
            "error": "",
        }

    def _new_dataset_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "state": "idle",
            "started_at": None,
            "completed_at": None,
            "episode_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "uncompressed_success_count": 0,
            "dataset_root": "",
            "repo_id": "",
            "episodes_root": "",
            "upload": False,
            "uploaded": False,
            "private": False,
            "returncode": None,
            "output_tail": [],
            "error": "",
        }

    def _new_hf_teleop_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "state": "idle",
            "started_at": None,
            "completed_at": None,
            "duration_s": None,
            "fps": None,
            "pid": None,
            "returncode": None,
            "command": [],
            "output_tail": [],
            "error": "",
        }

    def _eval_running(self) -> bool:
        return self.eval_thread is not None and self.eval_thread.is_alive()

    def _dataset_running(self) -> bool:
        return self.dataset_thread is not None and self.dataset_thread.is_alive()

    @staticmethod
    def _episode_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _dataset_episode_counts(self, root: Path | None = None) -> dict[str, int]:
        counts = {
            "episode_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "uncompressed_success_count": 0,
        }
        root = root or RECORD_ROOT
        if not root.exists():
            return counts
        for path in root.iterdir():
            if not path.is_dir() or not (path / "episode_meta.json").exists():
                continue
            result = self._episode_json(path / "episode_result.json")
            meta = self._episode_json(path / "episode_meta.json")
            outcome = str(result.get("outcome") or meta.get("outcome") or "")
            if outcome not in {"success", "failure"}:
                continue
            counts["episode_count"] += 1
            if outcome == "success":
                counts["success_count"] += 1
                if not result.get("compressed_dataset") and not meta.get("compressed_dataset"):
                    counts["uncompressed_success_count"] += 1
            elif outcome == "failure":
                counts["failure_count"] += 1
        return counts

    def _dataset_snapshot_locked(self) -> dict[str, Any]:
        data = dict(self.dataset_status)
        episodes_root = data.get("episodes_root")
        root = Path(episodes_root) if episodes_root else RECORD_ROOT
        data.update(self._dataset_episode_counts(root))
        data["running"] = self._dataset_running()
        return data

    def _eval_snapshot_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        data = dict(self.eval_status)
        data["running"] = self._eval_running()
        data["config"] = dict(self.eval_config)
        data["history"] = list(self.eval_history[-50:])
        if self.eval_started_at_mono is not None:
            end = self.eval_completed_at_mono if self.eval_completed_at_mono is not None else now
            data["elapsed_s"] = round(max(0.0, end - self.eval_started_at_mono), 1)
        else:
            data["elapsed_s"] = None
        if self.eval_deadline_mono is not None and self.eval_completed_at_mono is None:
            data["remaining_s"] = round(max(0.0, self.eval_deadline_mono - now), 1)
        else:
            data["remaining_s"] = None
        if self.eval_attempt_started_at_mono is not None:
            attempt_end = self.eval_completed_at_mono if self.eval_completed_at_mono is not None else now
            data["attempt_elapsed_s"] = round(max(0.0, attempt_end - self.eval_attempt_started_at_mono), 1)
        else:
            data["attempt_elapsed_s"] = None
        if self.eval_attempt_deadline_mono is not None and self.eval_completed_at_mono is None:
            data["attempt_remaining_s"] = round(max(0.0, self.eval_attempt_deadline_mono - now), 1)
        else:
            data["attempt_remaining_s"] = None
        run_duration_s = data.get("run_duration_s")
        if run_duration_s is not None and data["elapsed_s"] is not None:
            data["resume_remaining_s"] = round(max(0.0, float(run_duration_s) - float(data["elapsed_s"])), 1)
        else:
            data["resume_remaining_s"] = None
        teleop = self._teleop_snapshot_locked()
        data["can_resume"] = bool(
            self.eval_config
            and not data["running"]
            and data.get("state") in {"stopped", "failed", "waiting_intervention", "intervention_complete"}
            and data["resume_remaining_s"] is not None
            and data["resume_remaining_s"] > 1.0
            and (not teleop["active"] or teleop["stale"])
        )
        return data

    def _teleop_snapshot_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        lease = dict(self.teleop_lease) if self.teleop_lease is not None else None
        active = lease is not None
        heartbeat_age_s = None
        stale = False
        if lease is not None:
            last_heartbeat = float(lease.get("last_heartbeat_mono") or lease.get("claimed_at_mono") or now)
            heartbeat_age_s = round(max(0.0, now - last_heartbeat), 1)
            stale = heartbeat_age_s > TELEOP_LEASE_TIMEOUT_S
            lease.pop("claimed_at_mono", None)
            lease.pop("last_heartbeat_mono", None)

        state = str(self.eval_status.get("state") or "")
        intervention_ready = bool(
            self.eval_status.get("waiting_for_intervention")
            or state in {"waiting_intervention", "teleop_active", "intervention_complete"}
        )
        control_free = not self._eval_running() and not self._motion_running()
        return {
            "active": active,
            "available": control_free and (not active or stale),
            "stale": stale,
            "timeout_s": TELEOP_LEASE_TIMEOUT_S,
            "heartbeat_age_s": heartbeat_age_s,
            "intervention_ready": intervention_ready,
            "control_free": control_free,
            "lease": lease,
        }

    def _raise_if_teleop_active(self) -> None:
        with self.status_lock:
            snap = self._teleop_snapshot_locked()
            if snap["active"] and not snap["stale"]:
                lease = snap.get("lease") or {}
                raise RuntimeError(f"teleop is claimed by {lease.get('operator') or 'operator'}; release it first")
            if snap["active"] and snap["stale"]:
                self.teleop_lease = None

    def _require_teleop_lease(self, lease_id: str | None) -> None:
        with self.status_lock:
            snap = self._teleop_snapshot_locked()
            if not snap["active"]:
                raise RuntimeError("claim teleop before sending manual commands")
            if snap["stale"]:
                self.teleop_lease = None
                raise RuntimeError("teleop lease expired; claim teleop again")
            assert self.teleop_lease is not None
            current_id = str(self.teleop_lease.get("lease_id") or "")
            if lease_id and lease_id != current_id:
                operator = self.teleop_lease.get("operator") or "operator"
                raise RuntimeError(f"teleop is claimed by {operator}")
            if not lease_id:
                raise RuntimeError("missing teleop lease id")
            self.teleop_lease["last_heartbeat_mono"] = time.monotonic()

    def claim_teleop(self, operator: str = "operator") -> dict[str, Any]:
        if self._eval_running():
            raise RuntimeError("continuous eval is still running; wait for intervention state or stop eval")
        if self._motion_running():
            raise RuntimeError("motion is already running")

        operator = (operator or "operator").strip()[:80] or "operator"
        now = time.monotonic()
        wall_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        should_write_summary = False
        with self.status_lock:
            snap = self._teleop_snapshot_locked()
            if snap["active"] and not snap["stale"]:
                lease = snap.get("lease") or {}
                raise RuntimeError(f"teleop is already claimed by {lease.get('operator') or 'operator'}")

            reason = str(self.eval_status.get("stop_reason") or "manual")
            lease = {
                "lease_id": uuid.uuid4().hex,
                "operator": operator,
                "claimed_at": wall_time,
                "claimed_at_mono": now,
                "last_heartbeat_mono": now,
                "reason": reason,
            }
            self.teleop_lease = lease
            event = {
                "type": "teleop_claim",
                "at": wall_time,
                "operator": operator,
                "lease_id": lease["lease_id"],
                "reason": reason,
                "eval_state": self.eval_status.get("state"),
            }
            if self.eval_config:
                self.eval_history.append(dict(event))
                should_write_summary = True
            if self.eval_status.get("state") in {"waiting_intervention", "intervention_complete"}:
                self.eval_status.update(
                    {
                        "state": "teleop_active",
                        "waiting_for_intervention": True,
                        "last_intervention": dict(event),
                    }
                )
        self.log(f"teleop claimed operator={operator!r} reason={reason}")
        if should_write_summary:
            self._write_eval_summary()
        return self.status().get("teleop", {})

    def heartbeat_teleop(self, lease_id: str) -> dict[str, Any]:
        self._require_teleop_lease(lease_id)
        return self.status().get("teleop", {})

    def release_teleop(self, lease_id: str, outcome: str = "complete") -> dict[str, Any]:
        outcome = (outcome or "complete").strip()[:80] or "complete"
        should_write_summary = False
        with self.status_lock:
            if self.teleop_lease is None:
                raise RuntimeError("no active teleop lease")
            current_id = str(self.teleop_lease.get("lease_id") or "")
            if not lease_id or lease_id != current_id:
                operator = self.teleop_lease.get("operator") or "operator"
                raise RuntimeError(f"teleop is claimed by {operator}")
            now = time.monotonic()
            started = float(self.teleop_lease.get("claimed_at_mono") or now)
            operator = str(self.teleop_lease.get("operator") or "operator")
            event = {
                "type": "teleop_release",
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "operator": operator,
                "lease_id": current_id,
                "outcome": outcome,
                "duration_s": round(max(0.0, now - started), 3),
            }
            self.teleop_lease = None
            if self.eval_config:
                self.eval_history.append(dict(event))
                should_write_summary = True
            if self.eval_status.get("state") == "teleop_active":
                self.eval_status.update(
                    {
                        "state": "intervention_complete",
                        "waiting_for_intervention": False,
                        "last_intervention": dict(event),
                        "stop_reason": "teleop_release",
                    }
                )
        self.log(f"teleop released operator={operator!r} outcome={outcome}")
        if should_write_summary:
            self._write_eval_summary()
        return self.status().get("teleop", {})

    def _set_eval_status(self, **updates: Any) -> None:
        with self.status_lock:
            self.eval_status.update(updates)

    def _write_eval_summary(self) -> None:
        with self.status_lock:
            if self.eval_summary_path is None:
                if self.record_dir is not None:
                    summary_path = self.record_dir / "eval_summary.json"
                else:
                    summary_dir = RECORD_ROOT / f"so101_eval_{time.strftime('%Y%m%d_%H%M%S')}"
                    summary_path = summary_dir / "eval_summary.json"
                self.eval_summary_path = summary_path
                self.eval_status["summary_path"] = str(summary_path)
            summary_path = self.eval_summary_path
            payload = {
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "eval": self._eval_snapshot_locked(),
                "config": dict(self.eval_config),
                "attempts": list(self.eval_history),
                "events": list(self.eval_history),
                "recording": {
                    "dir": None if self.record_dir is None else str(self.record_dir),
                    "cameras": list(self.record_cameras),
                    "counts": dict(self.record_counts),
                    "fps": self.record_fps,
                    "duration_s": self.record_duration_s,
                    "error": self.record_error,
                },
            }
        try:
            assert summary_path is not None
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload, indent=2) + "\n")
        except BaseException as exc:
            with self.status_lock:
                self.eval_status["error"] = f"summary write failed: {exc}"
            self.log(f"eval summary write failed: {exc}")

    def ensure_robot(self) -> SO101Follower:
        with self.robot_lock:
            if self.robot is None:
                cfg = SO101FollowerConfig(
                    port=self.port,
                    id=self.robot_id,
                    max_relative_target=None,
                    disable_torque_on_disconnect=False,
                )
                robot = SO101Follower(cfg)
                robot.connect(calibrate=False)
                self.robot = robot
                self.log(f"connected {self.port} id={self.robot_id}")
            return self.robot

    def ensure_leader(self) -> SO101Leader:
        with self.leader_lock:
            if self.leader is None:
                cfg = SO101LeaderConfig(port=self.leader_port, id=self.leader_id)
                leader = SO101Leader(cfg)
                leader.connect()
                self.leader = leader
                self.log(f"connected leader {self.leader_port} id={self.leader_id}")
            return self.leader

    def disconnect(self) -> None:
        if self._eval_running():
            self.stop_eval()
            if self.eval_thread is not None:
                self.eval_thread.join(timeout=3.0)
        self.stop_policy()
        if self.policy_thread is not None:
            self.policy_thread.join(timeout=3.0)
        with self.robot_lock:
            if self.robot is not None:
                self.robot.disconnect()
                self.robot = None
                self.log("disconnected")
        with self.leader_lock:
            if self.leader is not None:
                self.leader.disconnect()
                self.leader = None
                self.log("leader disconnected")

    def shutdown(self) -> None:
        self.stop_success_tracking(join=True)
        self.stop_recording()
        if self.record_thread is not None:
            self.record_thread.join(timeout=3.0)
        self.stop_hf_teleop()
        self.disconnect()

    def _hf_teleop_running(self) -> bool:
        proc = self.hf_teleop_process
        return proc is not None and proc.poll() is None

    def _lerobot_teleoperate_command(self, duration_s: float, fps: int) -> list[str]:
        teleop_bin = Path(sys.executable).with_name("lerobot-teleoperate")
        executable = str(teleop_bin if teleop_bin.exists() else "lerobot-teleoperate")
        return [
            executable,
            "--robot.type",
            "so101_follower",
            "--robot.port",
            self.port,
            "--robot.id",
            self.robot_id,
            "--teleop.type",
            "so101_leader",
            "--teleop.port",
            self.leader_port,
            "--teleop.id",
            self.leader_id,
            "--fps",
            str(int(fps)),
            "--teleop_time_s",
            str(float(duration_s)),
            "--display_data",
            "false",
        ]

    def start_hf_teleop(self, duration_s: float = 10.0, fps: int = 30) -> dict[str, Any]:
        if duration_s <= 0:
            raise ValueError("teleop duration must be positive")
        if fps <= 0:
            raise ValueError("teleop fps must be positive")
        if self._hf_teleop_running():
            raise RuntimeError("HF teleop is already running")
        if self._recording_running():
            raise RuntimeError("stop station recording before starting HF teleop")
        if self._eval_running():
            raise RuntimeError("stop eval before starting HF teleop")
        if self._motion_running():
            self.stop_policy()
            if self.policy_thread is not None:
                self.policy_thread.join(timeout=3.0)

        self.disconnect()
        command = self._lerobot_teleoperate_command(duration_s=duration_s, fps=fps)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        output: deque[str] = deque(maxlen=80)
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except BaseException as exc:
            with self.status_lock:
                self.hf_teleop_status = self._new_hf_teleop_status()
                self.hf_teleop_status.update({"state": "failed", "error": str(exc), "command": command})
            raise

        with self.status_lock:
            self.hf_teleop_process = proc
            self.hf_teleop_status = self._new_hf_teleop_status()
            self.hf_teleop_status.update(
                {
                    "running": True,
                    "state": "running",
                    "started_at": started_at,
                    "duration_s": float(duration_s),
                    "fps": int(fps),
                    "pid": proc.pid,
                    "command": command,
                }
            )

        def runner() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if not line:
                        continue
                    output.append(line)
                    with self.status_lock:
                        self.hf_teleop_status["output_tail"] = list(output)
                returncode = proc.wait()
                state = "completed" if returncode == 0 else "failed"
                with self.status_lock:
                    self.hf_teleop_status.update(
                        {
                            "running": False,
                            "state": state,
                            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "returncode": returncode,
                            "output_tail": list(output),
                            "error": "" if returncode == 0 else f"lerobot-teleoperate exited {returncode}",
                        }
                    )
                self.log(f"HF teleop {state} returncode={returncode}")
            except BaseException as exc:
                with self.status_lock:
                    self.hf_teleop_status.update(
                        {
                            "running": False,
                            "state": "failed",
                            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "error": str(exc),
                            "output_tail": list(output),
                        }
                    )
                self.log(f"HF teleop error: {exc}")

        self.hf_teleop_thread = threading.Thread(target=runner, daemon=True)
        self.hf_teleop_thread.start()
        self.log(f"HF teleop start duration={duration_s}s fps={fps} pid={proc.pid}")
        return self.hf_teleop_snapshot()

    def stop_hf_teleop(self) -> dict[str, Any]:
        proc = self.hf_teleop_process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        if self.hf_teleop_thread is not None:
            self.hf_teleop_thread.join(timeout=1.0)
        with self.status_lock:
            if self.hf_teleop_status.get("running"):
                self.hf_teleop_status.update(
                    {
                        "running": False,
                        "state": "stopped",
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "returncode": None if proc is None else proc.poll(),
                    }
                )
        return self.hf_teleop_snapshot()

    def hf_teleop_snapshot(self) -> dict[str, Any]:
        with self.status_lock:
            data = dict(self.hf_teleop_status)
        proc = self.hf_teleop_process
        data["running"] = bool(proc is not None and proc.poll() is None)
        if not data["running"] and data.get("state") == "running":
            data["state"] = "completed" if proc is not None and proc.poll() == 0 else "failed"
            data["returncode"] = None if proc is None else proc.poll()
        return data

    def read_state(self, retries: int = 4) -> np.ndarray:
        last: BaseException | None = None
        for _ in range(retries):
            try:
                robot = self.ensure_robot()
                with self.robot_lock:
                    obs = robot.get_observation()
                state = np.array([obs[f"{joint}.pos"] for joint in JOINTS], dtype=np.float32)
                with self.status_lock:
                    self.last_state = [float(x) for x in state]
                return state
            except BaseException as exc:
                last = exc
                time.sleep(0.05)
        assert last is not None
        raise last

    def read_leader_state(self, retries: int = 4) -> np.ndarray:
        last: BaseException | None = None
        for _ in range(retries):
            try:
                leader = self.ensure_leader()
                with self.leader_lock:
                    action = leader.get_action()
                state = np.array([action[f"{joint}.pos"] for joint in JOINTS], dtype=np.float32)
                with self.status_lock:
                    self.last_leader_state = [float(x) for x in state]
                return state
            except BaseException as exc:
                last = exc
                time.sleep(0.05)
        assert last is not None
        raise last

    def read_station_joints(self) -> dict[str, Any]:
        follower = self.read_state()
        leader = self.read_leader_state()
        return {
            "joints": JOINTS,
            "follower": [float(x) for x in follower],
            "leader": [float(x) for x in leader],
        }

    def send_state(self, target: np.ndarray, retries: int = 4) -> None:
        action = {f"{joint}.pos": float(v) for joint, v in zip(JOINTS, target)}
        last: BaseException | None = None
        for _ in range(retries):
            try:
                robot = self.ensure_robot()
                with self.robot_lock:
                    robot.send_action(action)
                with self.status_lock:
                    self.last_action = [float(v) for v in target]
                    self.last_action_at_mono = time.monotonic()
                return
            except BaseException as exc:
                last = exc
                time.sleep(0.05)
        assert last is not None
        raise last

    def health(self) -> dict[str, Any]:
        return self.policy_client.health()

    def status(self, log_limit: int = DEFAULT_STATUS_LOG_LIMIT) -> dict[str, Any]:
        log_limit = max(0, min(MAX_STATUS_LOG_LIMIT, int(log_limit)))
        with self.status_lock:
            record_running = self.record_thread is not None and self.record_thread.is_alive()
            success_running = self.success_thread is not None and self.success_thread.is_alive()
            logs = list(self.logs)
            if log_limit:
                logs = logs[-log_limit:]
            else:
                logs = []
            return {
                "mode": self.mode,
                "stage": self.stage,
                "connected": self.robot is not None,
                "robots": [
                    {
                        "role": "follower",
                        "type": "so101_follower",
                        "id": self.robot_id,
                        "port": self.port,
                        "connected": self.robot is not None,
                        "state": self.last_state,
                        "joints": JOINTS,
                    },
                    {
                        "role": "leader",
                        "type": "so101_leader",
                        "id": self.leader_id,
                        "port": self.leader_port,
                        "connected": self.leader is not None,
                        "state": self.last_leader_state,
                        "joints": JOINTS,
                    },
                ],
                "robot_profiles": [
                    {
                        "id": "blupe_so101",
                        "name": "BluPe SO101",
                        "robot_type": "so101",
                        "follower": {
                            "role": "follower",
                            "type": "so101_follower",
                            "id": self.robot_id,
                            "port": self.port,
                            "connected": self.robot is not None,
                        },
                        "leader": {
                            "role": "leader",
                            "type": "so101_leader",
                            "id": self.leader_id,
                            "port": self.leader_port,
                            "connected": self.leader is not None,
                        },
                        "cameras": [cam.name for cam in self.camera_configs],
                    }
                ],
                "state": self.last_state,
                "last_action": self.last_action,
                "joints": JOINTS,
                "home": [float(x) for x in self.home_pose],
                "home_path": str(HOME_POSE_PATH),
                "error": self.last_error,
                "steps": self.steps,
                "chunks": self.chunks,
                "last_query_s": self.last_query_s,
                "policy_url": self.molmo_url,
                "policy_cameras": list(self.policy_camera_names),
                "log_path": "/tmp/so101_web_intervene.log",
                "running": self.policy_thread is not None and self.policy_thread.is_alive(),
                "elapsed": None if self.started_at is None else round(time.monotonic() - self.started_at, 1),
                "cameras": [self._camera_metadata(cam) for cam in self.camera_configs],
                "recording": {
                    "running": record_running,
                    "dir": None if self.record_dir is None else str(self.record_dir),
                    "cameras": list(self.record_cameras),
                    "camera_configs": [
                        self._camera_metadata(self.cameras_by_name[name])
                        for name in self.record_cameras
                        if name in self.cameras_by_name
                    ],
                    "counts": dict(self.record_counts),
                    "fps": self.record_fps,
                    "duration_s": self.record_duration_s,
                    "capture_mode": self.record_capture_mode,
                    "elapsed": None if self.record_started_at is None else round(time.monotonic() - self.record_started_at, 1),
                    "error": self.record_error,
                },
                "success_tracking": {
                    **dict(self.success_status),
                    "enabled": self.success_enabled,
                    "running": success_running,
                    "fps": self.success_fps,
                    "error": self.success_error,
                },
                "eval": self._eval_snapshot_locked(),
                "dataset": self._dataset_snapshot_locked(),
                "hf_teleop": {
                    **dict(self.hf_teleop_status),
                    "running": self._hf_teleop_running(),
                },
                "teleop": self._teleop_snapshot_locked(),
                "logs": logs,
            }

    def stop_policy(self) -> None:
        self.stop_event.set()
        self.stop_success_tracking(join=False)
        self.set_stage("stopping")
        self.set_mode("stopping")
        self.log("stop requested")

    def reset_success_tracking(
        self,
        recalculate_container: bool = True,
        container_reason: str = "manual_reset",
    ) -> dict[str, Any]:
        with self.success_tracker_lock:
            self.success_tracker.reset(
                recalculate_container=recalculate_container,
                container_reason=container_reason,
            )
            status = self.success_tracker.status()
        reset_overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.status_lock:
            self.success_status = status
            self.success_error = ""
        self._set_success_overlay(reset_overlay)
        source = self.success_status.get("container_mask_source")
        generation = self.success_status.get("container_mask_generation")
        self.log(f"success tracker reset container_gen={generation} source={source}")
        return dict(self.success_status)

    def configure_success_sam3(self, prompt: str, min_score: float) -> dict[str, Any]:
        with self.success_tracker_lock:
            self.success_tracker.configure_sam3(prompt=prompt, min_score=min_score)
            self.success_tracker.reset(
                recalculate_container=True,
                container_reason="sam3_prompt_update",
            )
            status = self.success_tracker.status()
        reset_overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.status_lock:
            self.success_status = status
            self.success_error = ""
        self._set_success_overlay(reset_overlay)
        self.log(
            "success SAM3 prompt "
            f"{status.get('container_mask_prompt')!r} min_score={status.get('container_mask_min_score')}"
        )
        return dict(status)

    def preview_success_sam3(self, prompt: str, min_score: float, camera: Any = "front") -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("SAM3 prompt cannot be empty")
        if min_score < 0.0 or min_score > 1.0:
            raise ValueError("SAM3 confidence must be between 0 and 1")
        cam = self._camera_from_spec(camera)

        rgb = read_mjpeg_frame(cam.url, timeout_s=5.0)
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            raise RuntimeError("failed to encode camera frame")

        payload = {
            "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "prompts": [prompt],
            "max_masks": 1,
            "min_score": float(min_score),
            "alpha": 0.65,
        }
        start = time.monotonic()
        resp = requests.post(
            self.success_tracker.container_sam3_url,
            json=payload,
            timeout=self.success_tracker.container_sam3_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        top = data.get("top_mask") or {}
        top_public = {k: v for k, v in top.items() if k != "mask_png_b64"}
        status = "accepted" if top_public else "no_top_mask"
        if top_public and float(top_public.get("score", 0.0)) < float(min_score):
            status = "low_score"
        return {
            "ok": True,
            "status": status,
            "prompt": prompt,
            "min_score": float(min_score),
            "camera": cam.name,
            "elapsed_s": round(time.monotonic() - start, 3),
            "overlay": data.get("overlay"),
            "top_mask": top_public,
            "results": data.get("results", []),
        }

    def start_success_tracking(self, container_reason: str = "policy_start") -> None:
        if not self.success_enabled:
            return
        if self.success_thread is not None and self.success_thread.is_alive():
            if self.success_stop_event.is_set():
                self.success_thread.join(timeout=1.0)
            if self.success_thread is not None and self.success_thread.is_alive():
                self.success_stop_event.clear()
                self.reset_success_tracking(container_reason=container_reason)
                self.log("success tracker reuse")
                return
        self.reset_success_tracking(container_reason=container_reason)
        if self.success_thread is not None and self.success_thread.is_alive():
            return
        self.success_stop_event.clear()
        self.success_thread = threading.Thread(target=self._success_loop, daemon=True)
        self.success_thread.start()
        self.log("success tracker start")

    def stop_success_tracking(self, join: bool = False) -> None:
        self.success_stop_event.set()
        if join and self.success_thread is not None:
            self.success_thread.join(timeout=2.0)

    def _set_success_overlay(self, overlay_rgb: np.ndarray | None) -> None:
        if overlay_rgb is None:
            return
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if not ok:
            return
        with self.success_condition:
            self.success_overlay_jpeg = encoded.tobytes()
            self.success_condition.notify_all()

    def _set_success_status(self, status: dict[str, Any], overlay_rgb: np.ndarray | None) -> None:
        with self.status_lock:
            self.success_status = status
            if status.get("success_this_frame"):
                self.logs.append(f"{time.strftime('%H:%M:%S')} {status.get('last_event')}")
        self._set_success_overlay(overlay_rgb)

    def _success_loop(self) -> None:
        period = 1.0 / self.success_fps if self.success_fps > 0 else 0.0
        next_process_t = 0.0
        buf = bytearray()
        success_cam = self.camera_configs[0]
        try:
            with urllib.request.urlopen(success_cam.url, timeout=5) as resp:
                while not self.success_stop_event.is_set():
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while not self.success_stop_event.is_set():
                        start = buf.find(b"\xff\xd8")
                        if start < 0:
                            if len(buf) > 1024 * 1024:
                                del buf[:-2]
                            break
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start > 0:
                                del buf[:start]
                            break
                        frame = bytes(buf[start : end + 2])
                        del buf[: end + 2]
                        now = time.monotonic()
                        if now < next_process_t:
                            continue
                        next_process_t = now + period
                        jpg = np.frombuffer(frame, dtype=np.uint8)
                        bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        with self.success_tracker_lock:
                            status, overlay = self.success_tracker.update(rgb)
                        self._set_success_status(status, overlay)
        except BaseException as exc:
            with self.status_lock:
                self.success_error = str(exc)
            self.log(f"success tracker error: {exc}")
        finally:
            with self.status_lock:
                self.success_status = self.success_tracker.status()
            self.log("success tracker stopped")

    def _motion_running(self) -> bool:
        return self.policy_thread is not None and self.policy_thread.is_alive()

    def _recording_running(self) -> bool:
        return self.record_thread is not None and self.record_thread.is_alive()

    def _recording_trajectory_active(self) -> bool:
        with self.status_lock:
            if self.record_capture_mode == "continuous":
                return True
            return self.mode == "policy" and self.stage == "execute"

    def _camera_metadata(self, cam: CameraConfig) -> dict[str, Any]:
        return {
            "id": int(cam.id),
            "name": cam.name,
            "url": cam.url,
            "frames_dir": cam.name,
            "frames_file": f"{cam.name}/frames.jsonl",
            "lerobot_key": cam.lerobot_key,
        }

    def _camera_from_spec(self, spec: Any) -> CameraConfig:
        if isinstance(spec, str):
            value = spec.strip()
            if value in self.cameras_by_name:
                return self.cameras_by_name[value]
            if value.startswith("cam") and value[3:].isdigit():
                cam_id = int(value[3:])
                if cam_id in self.cameras_by_id:
                    return self.cameras_by_id[cam_id]
            if value.isdigit():
                cam_id = int(value)
                if cam_id in self.cameras_by_id:
                    return self.cameras_by_id[cam_id]
        else:
            try:
                cam_id = int(spec)
            except (TypeError, ValueError):
                cam_id = -1
            if cam_id in self.cameras_by_id:
                return self.cameras_by_id[cam_id]
        valid = [cam.name for cam in self.camera_configs]
        raise ValueError(f"invalid camera {spec!r}; valid cameras: {valid}")

    def _normalize_cameras(self, cameras: list[Any] | None, *, default_all: bool = True) -> list[CameraConfig]:
        raw = list(cameras or [])
        selected = self.camera_configs if default_all and not raw else [self._camera_from_spec(item) for item in raw]
        deduped: list[CameraConfig] = []
        seen: set[str] = set()
        for cam in selected:
            if cam.name not in seen:
                deduped.append(cam)
                seen.add(cam.name)
        if not deduped:
            raise ValueError("at least one camera is required")
        return deduped

    def _policy_cameras(self) -> list[CameraConfig]:
        selected = [self.cameras_by_name[name] for name in self.policy_camera_names if name in self.cameras_by_name]
        return selected or self.camera_configs[:2]

    def _wait_motion_done(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._motion_running() and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.policy_thread is not None:
            self.policy_thread.join(timeout=0.1)
        return not self._motion_running()

    def start_recording(
        self,
        duration_s: float,
        fps: float,
        cameras: list[Any],
        task: str = DEFAULT_INSTRUCTION,
        name_prefix: str = "so101_recording",
        extra_meta: dict[str, Any] | None = None,
        capture_mode: str = "policy_execute",
    ) -> dict[str, Any]:
        if self._recording_running():
            raise RuntimeError("recording is already running")
        if duration_s <= 0:
            raise ValueError("recording duration must be positive")
        if fps <= 0:
            raise ValueError("recording fps must be positive")
        capture_mode = capture_mode.strip().lower()
        if capture_mode in {"policy_execute_stage_only", "policy"}:
            capture_mode = "policy_execute"
        if capture_mode not in {"policy_execute", "continuous"}:
            raise ValueError("recording capture_mode must be policy_execute or continuous")
        camera_configs = self._normalize_cameras(cameras, default_all=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_prefix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name_prefix).strip("_")
        if not safe_prefix:
            safe_prefix = "so101_recording"
        out_root = RAW_RECORD_ROOT if capture_mode == "continuous" else RECORD_ROOT
        if capture_mode == "continuous" and not safe_prefix.startswith("session"):
            safe_prefix = "session"
        out_dir = out_root / f"{safe_prefix}_{timestamp}"
        if out_dir.exists():
            out_dir = out_root / f"{safe_prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        record_start_mono = time.monotonic()
        task = task.strip() or DEFAULT_INSTRUCTION
        if capture_mode == "continuous":
            trajectory_time_source = "continuous_recording"
            notes = "Frames and samples are written continuously for manual/teleop demonstrations."
            sample_file = "samples.jsonl"
            meta_file = "session_meta.json"
            format_name = "blupe_so101_recording_session"
        else:
            trajectory_time_source = "policy_execute_stage_only"
            notes = "Frames and samples are only written while policy stage is execute; inference/capture waits are excluded."
            sample_file = "lerobot_samples.jsonl"
            meta_file = "episode_meta.json"
            format_name = "blupe_so101_episode"
        recording_meta = {
            "format": format_name,
            "format_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task": task,
            "robot_type": "so101_follower",
            "joints": JOINTS,
            "state_units": "degrees",
            "action_units": "degrees",
            "fps": float(fps),
            "duration_s": float(duration_s),
            "capture_mode": capture_mode,
            "trajectory_time_source": trajectory_time_source,
            "notes": notes,
            "sample_file": sample_file,
            "samples_file": sample_file,
            "cameras": [self._camera_metadata(cam) for cam in camera_configs],
        }
        if extra_meta:
            recording_meta.update(dict(extra_meta))
        (out_dir / meta_file).write_text(json.dumps(recording_meta, indent=2) + "\n")

        self.record_stop_event.clear()
        camera_names = [cam.name for cam in camera_configs]
        with self.status_lock:
            self.record_started_at = record_start_mono
            self.record_duration_s = float(duration_s)
            self.record_fps = float(fps)
            self.record_dir = out_dir
            self.record_cameras = camera_names
            self.record_counts = {name: 0 for name in camera_names}
            self.record_counts["samples"] = 0
            self.record_error = ""
            self.record_capture_mode = capture_mode

        self.record_thread = threading.Thread(
            target=self._record_loop,
            args=(out_dir, duration_s, fps, camera_configs, record_start_mono, sample_file),
            daemon=True,
        )
        self.record_thread.start()
        self.log(
            f"recording start duration={duration_s}s fps={fps} cameras={camera_names} "
            f"capture_mode={capture_mode} task={task!r} out={out_dir}"
        )
        return {"dir": str(out_dir), "cameras": camera_names, "capture_mode": capture_mode, "meta_file": meta_file}

    def stop_recording(self) -> None:
        if self._recording_running():
            self.record_stop_event.set()
            self.log("recording stop requested")

    def _finalize_recorded_episode(
        self,
        out_dir: Path | None,
        *,
        outcome: str,
        reason: str,
        attempt: int,
        started_at: str,
        ended_at: str,
        duration_s: float,
        event: dict[str, Any] | None = None,
    ) -> str | None:
        if out_dir is None:
            return None
        if self._recording_running():
            self.stop_recording()
        if self.record_thread is not None:
            self.record_thread.join(timeout=10.0)

        with self.status_lock:
            counts = dict(self.record_counts)
            error = self.record_error

        result = {
            "format": "blupe_so101_episode_result",
            "format_version": 1,
            "outcome": outcome,
            "reason": reason,
            "attempt": int(attempt),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": round(max(0.0, duration_s), 3),
            "counts": counts,
            "recording_error": error,
            "event": event or {},
        }
        try:
            meta_path = out_dir / "episode_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            meta.update(
                {
                    "outcome": outcome,
                    "reason": reason,
                    "attempt": int(attempt),
                    "ended_at": ended_at,
                    "result_file": "episode_result.json",
                    "record_counts": counts,
                    "recording_error": error,
                }
            )
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            (out_dir / "episode_result.json").write_text(json.dumps(result, indent=2) + "\n")
        except BaseException as exc:
            self.log(f"episode result write failed out={out_dir}: {exc}")

        label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in outcome).strip("_") or "unknown"
        if out_dir.name.startswith("so101_episode_"):
            target = out_dir.with_name(out_dir.name.replace("so101_episode_", f"so101_{label}_", 1))
        else:
            target = out_dir.with_name(f"so101_{label}_{out_dir.name}")
        if target != out_dir:
            base = target
            idx = 2
            while target.exists():
                target = base.with_name(f"{base.name}_{idx}")
                idx += 1
            try:
                out_dir.rename(target)
                out_dir = target
            except BaseException as exc:
                self.log(f"episode rename failed target={target}: {exc}")

        with self.status_lock:
            if self.record_dir is not None and self.record_dir.name != out_dir.name:
                self.record_dir = out_dir
        self.log(f"episode finalized outcome={outcome} reason={reason} out={out_dir}")
        return str(out_dir)

    def _record_loop(
        self,
        out_dir: Path,
        duration_s: float,
        fps: float,
        cameras: list[CameraConfig],
        record_start_mono: float,
        sample_file: str,
    ) -> None:
        threads = [
            threading.Thread(
                target=self._record_camera_loop,
                args=(cam, out_dir / cam.name, duration_s, fps, record_start_mono),
                daemon=True,
            )
            for cam in cameras
        ]
        threads.append(
            threading.Thread(
                target=self._record_robot_sample_loop,
                args=(out_dir, duration_s, fps, record_start_mono, sample_file),
                daemon=True,
            )
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        with self.status_lock:
            counts = dict(self.record_counts)
        if self.record_stop_event.is_set():
            self.log(f"recording stopped out={out_dir} counts={counts}")
        else:
            self.log(f"recording complete out={out_dir} counts={counts}")

    def _record_camera_loop(
        self,
        cam: CameraConfig,
        out_dir: Path,
        duration_s: float,
        fps: float,
        record_start_mono: float,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        period = 1.0 / fps
        end_t = record_start_mono + duration_s
        next_save_t = 0.0
        saved = 0
        buf = bytearray()
        frames_log = out_dir / "frames.jsonl"
        try:
            with urllib.request.urlopen(cam.url, timeout=5) as resp:
                while not self.record_stop_event.is_set() and time.monotonic() < end_t:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while not self.record_stop_event.is_set():
                        start = buf.find(b"\xff\xd8")
                        if start < 0:
                            if len(buf) > 1024 * 1024:
                                del buf[:-2]
                            break
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start > 0:
                                del buf[:start]
                            break
                        frame = bytes(buf[start : end + 2])
                        del buf[: end + 2]
                        now = time.monotonic()
                        if not self._recording_trajectory_active():
                            next_save_t = now
                            continue
                        if now >= next_save_t:
                            frame_name = f"frame_{saved:05d}.jpg"
                            (out_dir / frame_name).write_bytes(frame)
                            append_jsonl(
                                frames_log,
                                {
                                    "camera": cam.name,
                                    "frame_idx": saved,
                                    "frame": frame_name,
                                    "timestamp_s": round(saved * period, 6),
                                    "wall_elapsed_s": round(now - record_start_mono, 6),
                                    "monotonic_s": now,
                                },
                            )
                            saved += 1
                            next_save_t = now + period
                            with self.status_lock:
                                self.record_counts[cam.name] = saved
                        if time.monotonic() >= end_t:
                            break
        except BaseException as exc:
            with self.status_lock:
                self.record_error = f"{cam.name}: {exc}"
            self.log(f"recording {cam.name} error: {exc}")
        finally:
            with self.status_lock:
                self.record_counts[cam.name] = saved

    def _record_robot_sample_loop(
        self,
        out_dir: Path,
        duration_s: float,
        fps: float,
        record_start_mono: float,
        sample_file: str,
    ) -> None:
        samples_path = out_dir / sample_file
        period = 1.0 / fps
        end_t = record_start_mono + duration_s
        next_sample_t = record_start_mono
        sample_idx = 0
        while not self.record_stop_event.is_set() and time.monotonic() < end_t:
            now = time.monotonic()
            if not self._recording_trajectory_active():
                next_sample_t = now
                time.sleep(0.01)
                continue
            if now < next_sample_t:
                time.sleep(min(0.02, next_sample_t - now))
                continue

            error = ""
            state: list[float] | None = None
            try:
                measured = self.read_state()
                state = [float(x) for x in measured]
            except BaseException as exc:
                error = str(exc)
                with self.status_lock:
                    self.record_error = f"state: {exc}"

            with self.status_lock:
                action = list(self.last_action) if self.last_action is not None else state
                action_age_s = (
                    None
                    if self.last_action_at_mono is None
                    else round(max(0.0, now - self.last_action_at_mono), 6)
                )
                mode = self.mode
                stage = self.stage

            append_jsonl(
                samples_path,
                {
                    "sample_idx": sample_idx,
                    "timestamp_s": round(sample_idx * period, 6),
                    "wall_elapsed_s": round(now - record_start_mono, 6),
                    "monotonic_s": now,
                    "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "observation_state": state,
                    "action": action,
                    "action_age_s": action_age_s,
                    "mode": mode,
                    "stage": stage,
                    "error": error,
                },
            )
            sample_idx += 1
            with self.status_lock:
                self.record_counts["samples"] = sample_idx
            next_sample_t += period

    def start_eval(
        self,
        instruction: str,
        run_duration_s: float,
        attempt_duration_s: float,
        max_consecutive_failures: int,
        record_fps: float,
        exec_steps: int,
        max_step_deg: float,
        hz: float,
        allow_teleop: bool = True,
        record_episodes: bool = True,
    ) -> None:
        if self._eval_running():
            raise RuntimeError("continuous eval is already running")
        if self._motion_running():
            raise RuntimeError("motion is already running")
        self._raise_if_teleop_active()
        if run_duration_s <= 0:
            raise ValueError("eval run duration must be positive")
        if attempt_duration_s <= 0:
            raise ValueError("eval attempt duration must be positive")
        if max_consecutive_failures < 1:
            raise ValueError("max consecutive failures must be at least 1")
        if record_fps <= 0:
            raise ValueError("eval record fps must be positive")

        instruction = instruction.strip() or DEFAULT_INSTRUCTION
        now = time.monotonic()
        self.eval_stop_event.clear()
        config = {
            "instruction": instruction,
            "run_duration_s": float(run_duration_s),
            "attempt_duration_s": float(attempt_duration_s),
            "max_consecutive_failures": int(max_consecutive_failures),
            "record_fps": float(record_fps),
            "exec_steps": int(exec_steps),
            "max_step_deg": float(max_step_deg),
            "hz": float(hz),
            "allow_teleop": bool(allow_teleop),
            "record_episodes": bool(record_episodes),
        }
        with self.status_lock:
            self.eval_history = []
            self.eval_config = dict(config)
            self.eval_started_at_mono = now
            self.eval_completed_at_mono = None
            self.eval_deadline_mono = now + float(run_duration_s)
            self.eval_attempt_started_at_mono = None
            self.eval_attempt_deadline_mono = None
            self.eval_summary_path = None
            self.eval_status = self._new_eval_status()
            self.eval_status.update(
                {
                    "running": True,
                    "state": "starting",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "run_duration_s": float(run_duration_s),
                    "attempt_duration_s": float(attempt_duration_s),
                    "no_success_timeout_s": float(attempt_duration_s),
                    "max_consecutive_failures": int(max_consecutive_failures),
                    "allow_teleop": bool(allow_teleop),
                    "record_episodes": bool(record_episodes),
                }
            )
        self.reset_success_tracking(container_reason="eval_start")
        self.eval_thread = threading.Thread(
            target=self._eval_loop,
            args=(
                instruction,
                float(run_duration_s),
                float(attempt_duration_s),
                int(max_consecutive_failures),
                float(record_fps),
                int(exec_steps),
                float(max_step_deg),
                float(hz),
                bool(allow_teleop),
                bool(record_episodes),
                False,
            ),
            daemon=True,
        )
        self.eval_thread.start()
        self.log(
            f"eval start run={run_duration_s}s no_success_timeout={attempt_duration_s}s "
            f"max_failures={max_consecutive_failures} record_fps={record_fps} "
            f"allow_teleop={allow_teleop} record_episodes={record_episodes}"
        )

    def resume_eval(self) -> None:
        if self._eval_running():
            raise RuntimeError("continuous eval is already running")
        if self._motion_running():
            raise RuntimeError("motion is already running")
        self._raise_if_teleop_active()
        with self.status_lock:
            if not self.eval_config:
                raise RuntimeError("no previous eval to resume")
            if self.eval_started_at_mono is None:
                raise RuntimeError("previous eval has no start time")
            run_duration_s = float(self.eval_config.get("run_duration_s", self.eval_status.get("run_duration_s") or 0))
            if run_duration_s <= 0:
                raise RuntimeError("previous eval has no run duration")
            end = self.eval_completed_at_mono if self.eval_completed_at_mono is not None else time.monotonic()
            elapsed_s = max(0.0, end - self.eval_started_at_mono)
            remaining_s = max(0.0, run_duration_s - elapsed_s)
            if remaining_s <= 1.0:
                raise RuntimeError("previous eval has no remaining time")
            now = time.monotonic()
            self.eval_started_at_mono = now - elapsed_s
            self.eval_completed_at_mono = None
            self.eval_deadline_mono = now + remaining_s
            self.eval_attempt_started_at_mono = None
            self.eval_attempt_deadline_mono = None
            self.eval_summary_path = None
            self.eval_status.update(
                {
                    "running": True,
                    "state": "resuming",
                    "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "completed_at": None,
                    "waiting_for_intervention": False,
                    "current_attempt": None,
                    "recording_dir": None,
                    "summary_path": None,
                    "stop_reason": "",
                    "error": "",
                    "consecutive_failures": 0,
                    "max_consecutive_failures": int(
                        self.eval_config.get("max_consecutive_failures", DEFAULT_EVAL_MAX_CONSECUTIVE_FAILURES)
                    ),
                }
            )
            config = dict(self.eval_config)

        self.reset_success_tracking(container_reason="eval_resume")
        self.eval_stop_event.clear()
        self.eval_thread = threading.Thread(
            target=self._eval_loop,
            args=(
                str(config.get("instruction", DEFAULT_INSTRUCTION)),
                float(remaining_s),
                float(config.get("attempt_duration_s", DEFAULT_EVAL_ATTEMPT_DURATION_S)),
                int(config.get("max_consecutive_failures", DEFAULT_EVAL_MAX_CONSECUTIVE_FAILURES)),
                float(config.get("record_fps", DEFAULT_EVAL_RECORD_FPS)),
                int(config.get("exec_steps", DEFAULT_EXEC_STEPS)),
                float(config.get("max_step_deg", DEFAULT_MAX_STEP_DEG)),
                float(config.get("hz", DEFAULT_HZ)),
                bool(config.get("allow_teleop", True)),
                bool(config.get("record_episodes", True)),
                True,
            ),
            daemon=True,
        )
        self.eval_thread.start()
        self.log(f"eval resume remaining={remaining_s:.1f}s")

    def stop_eval(self) -> None:
        self.eval_stop_event.set()
        with self.status_lock:
            if self._eval_running():
                self.eval_status.update({"state": "stopping", "stop_reason": "user_stop"})
            elif self.eval_status.get("state") not in {"idle", "stopped", "complete", "waiting_intervention"}:
                self.eval_status.update({"state": "stopped", "stop_reason": "user_stop"})
        self.stop_policy()
        self.stop_success_tracking(join=False)
        self.stop_recording()
        self.log("eval stop requested; policy, tracker, and recording stop signaled")

    def clear_eval(self) -> dict[str, Any]:
        if self._eval_running() or self._motion_running():
            raise RuntimeError("stop eval before clearing")
        self._raise_if_teleop_active()
        with self.status_lock:
            self.eval_history = []
            self.eval_started_at_mono = None
            self.eval_completed_at_mono = None
            self.eval_deadline_mono = None
            self.eval_attempt_started_at_mono = None
            self.eval_attempt_deadline_mono = None
            self.eval_summary_path = None
            self.eval_config = {}
            self.eval_status = self._new_eval_status()
        self.log("eval cleared")
        return self.status().get("eval", {})

    def extract_busyboard_segments(
        self,
        source_name: str,
        segments: Any,
        cameras: list[str] | None = None,
        output_dataset_name: str = "",
    ) -> dict[str, Any]:
        if self._recording_running():
            raise RuntimeError("stop recording before extracting busyboard subepisodes")
        if self._eval_running() or self._motion_running():
            raise RuntimeError("stop eval/policy before extracting busyboard subepisodes")
        if not isinstance(segments, list) or not segments:
            raise ValueError("segments must be a non-empty list")

        source_dir = _safe_episode_dir((source_name or "latest").strip() or "latest")
        output_root = RECORD_ROOT / _safe_dataset_name(output_dataset_name, fallback="edited-dataset")
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="busyboard_segments_", delete=False) as f:
                json.dump({"segments": segments}, f)
                tmp_path = Path(f.name)
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "extract_so101_subepisodes.py"),
                "--source-dir",
                str(source_dir),
                "--output-root",
                str(output_root),
                "--segments-json",
                str(tmp_path),
            ]
            for camera in cameras or []:
                camera = str(camera).strip()
                if camera:
                    cmd.extend(["--camera", camera])
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout.strip() or f"extract command failed exit={proc.returncode}")
            summary = json.loads(proc.stdout)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
        self.log(
            "busyboard extracted "
            f"source={source_dir.name} episodes={summary.get('episode_count')} frames={summary.get('total_frames')}"
        )
        return summary

    def start_dataset_compression(
        self,
        *,
        repo_id: str = "",
        dataset_name: str = "",
        root: str = "",
        episodes_root_name: str = "",
        upload: bool = False,
        private: bool = False,
        include_failures: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if self._dataset_running():
            raise RuntimeError("dataset compression is already running")
        if self._eval_running() or self._motion_running() or self._recording_running():
            raise RuntimeError("stop eval/recording before compressing episodes")

        repo_id = repo_id.strip()
        dataset_name = dataset_name.strip()
        root = root.strip()
        episodes_root = RECORD_ROOT / _safe_dataset_name(episodes_root_name) if episodes_root_name.strip() else RECORD_ROOT
        counts = self._dataset_episode_counts(episodes_root)
        if counts["uncompressed_success_count"] < 1 and not include_failures:
            raise RuntimeError("no uncompressed successful episodes to compress")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compress_so101_episodes.py"),
            "--episodes-root",
            str(episodes_root),
        ]
        if repo_id:
            cmd.extend(["--repo-id", repo_id])
        if dataset_name:
            cmd.extend(["--dataset-name", dataset_name])
        if root:
            cmd.extend(["--root", root])
        if upload:
            cmd.append("--upload")
        if private:
            cmd.append("--private")
        if include_failures:
            cmd.append("--include-failures")
        if overwrite:
            cmd.append("--overwrite")

        with self.status_lock:
            self.dataset_status = self._new_dataset_status()
            self.dataset_status.update(
                {
                    "running": True,
                    "state": "starting",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "repo_id": repo_id,
                    "dataset_root": root,
                    "episodes_root": str(episodes_root),
                    "upload": bool(upload),
                    "private": bool(private),
                    "overwrite": bool(overwrite),
                    **counts,
                }
            )

        self.dataset_thread = threading.Thread(target=self._dataset_compress_loop, args=(cmd,), daemon=True)
        self.dataset_thread.start()
        self.log(f"dataset compression start upload={upload} repo_id={repo_id or 'auto'}")
        return self.status().get("dataset", {})

    def _dataset_compress_loop(self, cmd: list[str]) -> None:
        output: deque[str] = deque(maxlen=80)
        returncode: int | None = None
        error = ""
        try:
            with self.status_lock:
                self.dataset_status["state"] = "running"
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                output.append(line)
                with self.status_lock:
                    self.dataset_status["output_tail"] = list(output)
                    if "dataset_root" in line:
                        self.dataset_status["state"] = "planning"
                    elif line.startswith("saved episode"):
                        self.dataset_status["state"] = "encoding"
                    elif line.startswith("uploaded dataset"):
                        self.dataset_status["uploaded"] = True
                self.log(f"dataset {line[:180]}")
            returncode = proc.wait()
            if returncode != 0:
                error = f"compression command failed exit={returncode}"
        except BaseException as exc:
            error = str(exc)
        with self.status_lock:
            episodes_root = self.dataset_status.get("episodes_root")
            counts = self._dataset_episode_counts(Path(episodes_root) if episodes_root else RECORD_ROOT)
            self.dataset_status.update(
                {
                    "running": False,
                    "state": "failed" if error else "complete",
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "returncode": returncode,
                    "error": error,
                    "output_tail": list(output),
                    **counts,
                }
            )
        if error:
            self.log(f"dataset compression failed: {error}")
        else:
            self.log("dataset compression complete")

    def _current_success_status(self) -> dict[str, Any]:
        with self.status_lock:
            return dict(self.success_status)

    @staticmethod
    def _tracker_event_context(success_status: dict[str, Any] | None) -> dict[str, Any]:
        if not success_status:
            return {}
        return {
            "container_mask_generation": success_status.get("container_mask_generation"),
            "container_mask_source": success_status.get("container_mask_source"),
            "container_mask_area": success_status.get("container_mask_area"),
            "container_mask_box_xyxy": success_status.get("container_mask_box_xyxy"),
            "container_mask_calculated_at_frame": success_status.get("container_mask_calculated_at_frame"),
            "container_mask_prompt": success_status.get("container_mask_prompt"),
            "container_mask_score": success_status.get("container_mask_score"),
            "container_mask_min_score": success_status.get("container_mask_min_score"),
            "container_mask_sam3_status": success_status.get("container_mask_sam3_status"),
            "container_mask_sam3_error": success_status.get("container_mask_sam3_error"),
            "container_mask_sam3_raw_score": success_status.get("container_mask_sam3_raw_score"),
            "container_mask_sam3_raw_area": success_status.get("container_mask_sam3_raw_area"),
            "container_mask_sam3_box_xyxy": success_status.get("container_mask_sam3_box_xyxy"),
        }

    def _eval_home(self) -> bool:
        self._set_eval_status(state="homing", current_attempt=None, home_duration_s=DEFAULT_EVAL_HOME_S)
        self.log(f"eval home start duration={DEFAULT_EVAL_HOME_S:.1f}s")
        try:
            self.start_home(max_step_deg=HOME_STEP_DEG, hz=DEFAULT_HZ, from_eval=True)
        except BaseException as exc:
            with self.status_lock:
                self.eval_status["error"] = str(exc)
            self.log(f"eval home start failed: {exc}")
            return False

        end_t = time.monotonic() + DEFAULT_EVAL_HOME_S
        while time.monotonic() < end_t and self._motion_running() and not self.eval_stop_event.is_set():
            time.sleep(0.05)
        if self._motion_running():
            self.stop_policy()
            self._wait_motion_done(3.0)
        self.log("eval home pulse complete")
        return True

    def _eval_loop(
        self,
        instruction: str,
        run_duration_s: float,
        attempt_duration_s: float,
        max_consecutive_failures: int,
        record_fps: float,
        exec_steps: int,
        max_step_deg: float,
        hz: float,
        allow_teleop: bool = True,
        record_episodes: bool = True,
        resume: bool = False,
    ) -> None:
        end_t = time.monotonic() + run_duration_s
        stop_reason = "complete"
        waiting_for_intervention = False
        active_record_dir: Path | None = None
        active_record_run_num = 0
        active_record_started_at = ""
        active_record_started_mono: float | None = None

        def request_intervention(reason: str) -> bool:
            nonlocal stop_reason, waiting_for_intervention
            stop_reason = reason
            if allow_teleop:
                waiting_for_intervention = True
                return True
            return False

        if resume:
            with self.status_lock:
                successes = int(self.eval_status.get("successes") or 0)
                failures = int(self.eval_status.get("failures") or 0)
                consecutive_failures = int(self.eval_status.get("consecutive_failures") or 0)
                run_num = int(self.eval_status.get("attempt") or 0)
        else:
            successes = 0
            failures = 0
            consecutive_failures = 0
            run_num = 0
        try:
            if self._recording_running():
                self.log("eval replacing active recording")
                self.stop_recording()
                if self.record_thread is not None:
                    self.record_thread.join(timeout=5.0)
                if self._recording_running():
                    raise RuntimeError("recording is still stopping; retry eval start")

            summary_dir = RECORD_ROOT / f"so101_eval_{time.strftime('%Y%m%d_%H%M%S')}"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / "eval_summary.json"
            with self.status_lock:
                self.eval_summary_path = summary_path
                self.eval_status.update(
                    {
                        "state": "running",
                        "recording_dir": None,
                        "summary_path": str(summary_path),
                        "allow_teleop": bool(allow_teleop),
                        "record_episodes": bool(record_episodes),
                        "episode_root": str(summary_dir),
                    }
                )
            self._write_eval_summary()

            while not self.eval_stop_event.is_set() and time.monotonic() < end_t:
                remaining_s = end_t - time.monotonic()
                if remaining_s < 1.0:
                    break
                if record_episodes and remaining_s < attempt_duration_s:
                    stop_reason = "run_duration_complete"
                    break

                run_num += 1
                run_started_mono = time.monotonic()
                watchdog_started_mono = run_started_mono
                watchdog_deadline = min(watchdog_started_mono + attempt_duration_s, end_t)
                run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                run_record: dict[str, Any] = {
                    "attempt": run_num,
                    "run": run_num,
                    "started_at": run_started_at,
                    "no_success_timeout_s": round(attempt_duration_s, 3),
                    "outcome": "running",
                    "reason": "",
                    "successes": 0,
                }
                attempt_record_dir: Path | None = None
                if record_episodes:
                    record_duration_s = max(1.0, watchdog_deadline - time.monotonic() + MOLMO_TIMEOUT_S + 15.0)
                    recording = self.start_recording(
                        record_duration_s,
                        record_fps,
                        [cam.name for cam in self.camera_configs],
                        task=instruction,
                        name_prefix=f"so101_episode_run{run_num:04d}",
                        extra_meta={
                            "eval_attempt": run_num,
                            "eval_run": run_num,
                            "eval_started_at": run_started_at,
                            "eval_summary_path": str(summary_path),
                            "expected_outcomes": ["success", "failure"],
                        },
                    )
                    attempt_record_dir = Path(recording["dir"])
                    active_record_dir = attempt_record_dir
                    active_record_run_num = run_num
                    active_record_started_at = run_started_at
                    active_record_started_mono = run_started_mono
                    run_record["recording_dir"] = recording["dir"]
                with self.status_lock:
                    self.eval_attempt_started_at_mono = watchdog_started_mono
                    self.eval_attempt_deadline_mono = watchdog_deadline
                    self.eval_status.update(
                        {
                            "state": "policy_running",
                            "attempt": run_num,
                            "current_attempt": dict(run_record),
                            "recording_dir": None if attempt_record_dir is None else str(attempt_record_dir),
                            "successes": successes,
                            "failures": failures,
                            "consecutive_failures": consecutive_failures,
                            "error": "",
                        }
                    )
                self._write_eval_summary()
                self.log(f"eval policy run {run_num} start no_success_timeout={attempt_duration_s:.1f}s")

                success_status: dict[str, Any] | None = None
                policy_duration_s = max(0.1, end_t - time.monotonic())
                self.start_policy(
                    instruction=instruction,
                    duration_s=policy_duration_s,
                    exec_steps=exec_steps,
                    max_step_deg=max_step_deg,
                    hz=hz,
                    from_eval=True,
                    success_container_reason=f"eval_run_{run_num}",
                )

                run_outcome = "running"
                run_reason = ""
                run_event: dict[str, Any] | None = None
                segment_successes = 0
                last_tracker_success_count = 0
                while not self.eval_stop_event.is_set():
                    now = time.monotonic()
                    success_status = self._current_success_status()
                    tracker_success_count = int(success_status.get("success_count") or 0)
                    if tracker_success_count > last_tracker_success_count:
                        delta = tracker_success_count - last_tracker_success_count
                        last_tracker_success_count = tracker_success_count
                        segment_successes += delta
                        successes += delta
                        consecutive_failures = 0
                        event_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                        watchdog_started_mono = now
                        watchdog_deadline = min(now + attempt_duration_s, end_t)
                        success_event = {
                            "type": "success",
                            "attempt": run_num,
                            "run": run_num,
                            "sequence": successes,
                            "at": event_time,
                            "elapsed_s": round(now - run_started_mono, 3),
                            "outcome": "success",
                            "reason": "mask_success",
                            "success": success_status.get("last_success"),
                            "tracker_success_count": tracker_success_count,
                            "tracker_state": success_status.get("state"),
                            "tracker_overlap": success_status.get("current_overlap"),
                            "recording_dir": None if attempt_record_dir is None else str(attempt_record_dir),
                            **self._tracker_event_context(success_status),
                        }
                        current = {
                            **run_record,
                            "successes": segment_successes,
                            "last_success_at": event_time,
                        }
                        run_outcome = "success"
                        run_reason = "mask_success"
                        run_event = dict(success_event)
                        with self.status_lock:
                            self.eval_history.append(dict(success_event))
                            self.eval_attempt_started_at_mono = None
                            self.eval_attempt_deadline_mono = None
                            self.eval_status.update(
                                {
                                    "state": "success",
                                    "current_attempt": None,
                                    "successes": successes,
                                    "failures": failures,
                                    "consecutive_failures": consecutive_failures,
                                    "last_attempt": dict(success_event),
                                    "last_success": success_status.get("last_success"),
                                    "last_success_at": event_time,
                                    "stop_reason": "",
                                }
                            )
                        self.log(f"eval success {successes} run={run_num}; ending episode")
                        self._write_eval_summary()
                        self.stop_policy()
                        break

                    if now >= end_t:
                        run_outcome = "complete"
                        run_reason = "run_duration_complete"
                        self.stop_policy()
                        break

                    if now >= watchdog_deadline:
                        failures += 1
                        consecutive_failures += 1
                        run_outcome = "failure"
                        run_reason = "no_success_timeout"
                        failure_event = {
                            "type": "failure",
                            "attempt": run_num,
                            "run": run_num,
                            "started_at": time.strftime(
                                "%Y-%m-%dT%H:%M:%S%z",
                                time.localtime(time.time() - max(0.0, now - watchdog_started_mono)),
                            ),
                            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "duration_s": round(now - watchdog_started_mono, 3),
                            "outcome": "failure",
                            "reason": run_reason,
                            "successes_in_run": segment_successes,
                            "success_count": tracker_success_count,
                            "tracker_state": success_status.get("state"),
                            "tracker_overlap": success_status.get("current_overlap"),
                            "recording_dir": None if attempt_record_dir is None else str(attempt_record_dir),
                            **self._tracker_event_context(success_status),
                        }
                        run_event = dict(failure_event)
                        with self.status_lock:
                            self.eval_history.append(dict(failure_event))
                            self.eval_status.update(
                                {
                                    "state": "no_success_timeout",
                                    "successes": successes,
                                    "failures": failures,
                                    "consecutive_failures": consecutive_failures,
                                    "last_attempt": dict(failure_event),
                                    "current_attempt": None,
                                }
                            )
                        self.log(
                            f"eval run {run_num} failure no success for {attempt_duration_s:.1f}s "
                            f"streak={consecutive_failures}"
                        )
                        self._write_eval_summary()
                        self.stop_policy()
                        break

                    if not self._motion_running():
                        with self.status_lock:
                            err = self.last_error
                        if time.monotonic() >= end_t:
                            run_outcome = "complete"
                            run_reason = "run_duration_complete"
                            break
                        if err:
                            run_outcome = "failure"
                            run_reason = "policy_error"
                            failure_event = {
                                "type": "failure",
                                "attempt": run_num,
                                "run": run_num,
                                "started_at": run_started_at,
                                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "duration_s": round(time.monotonic() - run_started_mono, 3),
                                "outcome": "failure",
                                "reason": run_reason,
                                "successes_in_run": segment_successes,
                                "success_count": tracker_success_count,
                                "tracker_state": success_status.get("state"),
                                "tracker_overlap": success_status.get("current_overlap"),
                                "policy_error": err,
                                "recording_dir": None if attempt_record_dir is None else str(attempt_record_dir),
                                **self._tracker_event_context(success_status),
                            }
                            run_event = dict(failure_event)
                            with self.status_lock:
                                self.eval_history.append(dict(failure_event))
                                self.eval_attempt_started_at_mono = None
                                self.eval_attempt_deadline_mono = None
                                self.eval_status.update(
                                    {
                                        "state": "waiting_intervention" if allow_teleop else "policy_error",
                                        "successes": successes,
                                        "failures": failures,
                                        "consecutive_failures": consecutive_failures,
                                        "waiting_for_intervention": bool(allow_teleop),
                                        "last_attempt": dict(failure_event),
                                        "current_attempt": None,
                                        "stop_reason": run_reason,
                                        "error": err,
                                    }
                                )
                            self.log(f"eval run {run_num} intervention reason={run_reason}: {err}")
                            self._write_eval_summary()
                            request_intervention(run_reason)
                            break
                        failures += 1
                        consecutive_failures += 1
                        run_outcome = "failure"
                        run_reason = "policy_stopped_without_timeout"
                        failure_event = {
                            "type": "failure",
                            "attempt": run_num,
                            "run": run_num,
                            "started_at": run_started_at,
                            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "duration_s": round(time.monotonic() - run_started_mono, 3),
                            "outcome": "failure",
                            "reason": run_reason,
                            "successes_in_run": segment_successes,
                            "success_count": tracker_success_count,
                            "tracker_state": success_status.get("state"),
                            "tracker_overlap": success_status.get("current_overlap"),
                            "policy_error": err,
                            "recording_dir": None if attempt_record_dir is None else str(attempt_record_dir),
                            **self._tracker_event_context(success_status),
                        }
                        run_event = dict(failure_event)
                        with self.status_lock:
                            self.eval_history.append(dict(failure_event))
                            self.eval_attempt_started_at_mono = None
                            self.eval_attempt_deadline_mono = None
                            self.eval_status.update(
                                {
                                    "state": "policy_stopped",
                                    "successes": successes,
                                    "failures": failures,
                                    "consecutive_failures": consecutive_failures,
                                    "last_attempt": dict(failure_event),
                                    "current_attempt": None,
                                }
                            )
                        self.log(f"eval run {run_num} failure reason={run_reason}")
                        self._write_eval_summary()
                        break
                    time.sleep(0.2)

                if self.eval_stop_event.is_set():
                    run_outcome = "stopped"
                    run_reason = "user_stop"
                    self.stop_policy()

                motion_stopped = self._wait_motion_done(MOLMO_TIMEOUT_S + 15.0)
                if not motion_stopped:
                    already_counted_failure = run_outcome == "failure"
                    if not already_counted_failure:
                        failures += 1
                        consecutive_failures += 1
                    run_outcome = "failure"
                    run_reason = "motion_stop_timeout"
                    request_intervention(run_reason)
                    failure_event = {
                        "type": "failure",
                        "attempt": run_num,
                        "run": run_num,
                        "started_at": run_started_at,
                        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "duration_s": round(time.monotonic() - run_started_mono, 3),
                        "outcome": "failure",
                        "reason": run_reason,
                        "successes_in_run": segment_successes,
                        "recording_dir": None if attempt_record_dir is None else str(attempt_record_dir),
                        **self._tracker_event_context(success_status),
                    }
                    run_event = dict(failure_event)
                    with self.status_lock:
                        if not already_counted_failure:
                            self.eval_history.append(dict(failure_event))
                        self.eval_status.update(
                            {
                                "state": "motion_stop_timeout",
                                "successes": successes,
                                "failures": failures,
                                "consecutive_failures": consecutive_failures,
                                "last_attempt": dict(failure_event),
                                "current_attempt": None,
                            }
                        )
                    self._write_eval_summary()

                if attempt_record_dir is not None:
                    episode_ended_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    final_recording_dir = self._finalize_recorded_episode(
                        attempt_record_dir,
                        outcome=run_outcome if run_outcome != "running" else "stopped",
                        reason=run_reason or stop_reason,
                        attempt=run_num,
                        started_at=run_started_at,
                        ended_at=episode_ended_at,
                        duration_s=time.monotonic() - run_started_mono,
                        event=run_event,
                    )
                    if final_recording_dir:
                        if run_event is not None:
                            run_event["recording_dir"] = final_recording_dir
                        with self.status_lock:
                            for event in reversed(self.eval_history):
                                if int(event.get("run") or -1) == run_num and not event.get("recording_final_dir"):
                                    event["recording_dir"] = final_recording_dir
                                    event["recording_final_dir"] = final_recording_dir
                                    break
                            last_attempt = self.eval_status.get("last_attempt")
                            if isinstance(last_attempt, dict) and int(last_attempt.get("run") or -1) == run_num:
                                last_attempt["recording_dir"] = final_recording_dir
                                last_attempt["recording_final_dir"] = final_recording_dir
                                self.eval_status["last_attempt"] = last_attempt
                            self.eval_status["recording_dir"] = final_recording_dir
                    active_record_dir = None
                    active_record_run_num = 0
                    active_record_started_at = ""
                    active_record_started_mono = None
                    self._write_eval_summary()

                with self.status_lock:
                    self.eval_attempt_started_at_mono = None
                    self.eval_attempt_deadline_mono = None

                if run_outcome == "stopped":
                    stop_reason = "user_stop"
                    break
                if waiting_for_intervention:
                    break
                if run_outcome == "complete":
                    break

                if run_outcome == "failure" and consecutive_failures >= max_consecutive_failures:
                    request_intervention("consecutive_failures")
                    break

                if run_outcome in {"success", "failure"}:
                    home_ok = self._eval_home()
                    self._write_eval_summary()
                    if not home_ok:
                        request_intervention("home_failed")
                        break

            if self.eval_stop_event.is_set() and stop_reason == "complete":
                stop_reason = "user_stop"
        except BaseException as exc:
            request_intervention("error")
            with self.status_lock:
                self.eval_status["error"] = str(exc)
            self.set_error(exc)
        finally:
            if self._motion_running():
                self.stop_policy()
                self._wait_motion_done(MOLMO_TIMEOUT_S + 15.0)
            if active_record_dir is not None:
                final_recording_dir = self._finalize_recorded_episode(
                    active_record_dir,
                    outcome="stopped" if stop_reason == "user_stop" else "failure",
                    reason=stop_reason or "eval_shutdown",
                    attempt=active_record_run_num,
                    started_at=active_record_started_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    duration_s=0.0
                    if active_record_started_mono is None
                    else time.monotonic() - active_record_started_mono,
                    event={"type": "failure", "reason": stop_reason or "eval_shutdown"},
                )
                if final_recording_dir:
                    with self.status_lock:
                        self.eval_status["recording_dir"] = final_recording_dir
                active_record_dir = None
            elif self._recording_running():
                self.stop_recording()
                if self.record_thread is not None:
                    self.record_thread.join(timeout=5.0)

            completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            final_state = (
                "waiting_intervention"
                if waiting_for_intervention
                else "stopped"
                if stop_reason == "user_stop"
                else "complete"
                if stop_reason in {"complete", "run_duration_complete"}
                else "failed"
            )
            with self.status_lock:
                self.eval_completed_at_mono = time.monotonic()
                self.eval_attempt_started_at_mono = None
                self.eval_attempt_deadline_mono = None
                self.eval_status.update(
                    {
                        "running": False,
                        "state": final_state,
                        "completed_at": completed_at,
                        "successes": successes,
                        "failures": failures,
                        "consecutive_failures": consecutive_failures,
                        "waiting_for_intervention": waiting_for_intervention,
                        "current_attempt": None,
                        "stop_reason": stop_reason,
                    }
                )
            self._write_eval_summary()
            self.eval_stop_event.set()
            self.log(f"eval {final_state} reason={stop_reason} successes={successes} failures={failures}")

    def start_policy(
        self,
        instruction: str,
        duration_s: float,
        exec_steps: int,
        max_step_deg: float,
        hz: float,
        from_eval: bool = False,
        success_container_reason: str = "policy_start",
    ) -> None:
        if self._eval_running() and not from_eval:
            raise RuntimeError("continuous eval is running; stop it first")
        if self._motion_running():
            raise RuntimeError("policy is already running")
        if not from_eval:
            self._raise_if_teleop_active()
        instruction = instruction.strip() or DEFAULT_INSTRUCTION
        self.stop_event.clear()
        self.start_success_tracking(container_reason=success_container_reason)
        with self.status_lock:
            self.steps = 0
            self.chunks = 0
            self.last_error = ""
            self.last_query_s = None
            self.started_at = time.monotonic()
        self.policy_thread = threading.Thread(
            target=self._policy_loop,
            args=(instruction, duration_s, exec_steps, max_step_deg, hz),
            daemon=True,
        )
        self.policy_thread.start()

    def _policy_loop(
        self,
        instruction: str,
        duration_s: float,
        exec_steps: int,
        max_step_deg: float,
        hz: float,
    ) -> None:
        self.set_mode("policy")
        self.set_stage("starting")
        effective_max_step_deg, max_step_clamped = policy_step_limit(max_step_deg)
        clamp_note = f" clamped_to={effective_max_step_deg}" if max_step_clamped else ""
        self.log(
            f"policy start duration={duration_s}s exec_steps={exec_steps} "
            f"max_step={max_step_deg}{clamp_note} hz={hz} "
            f"policy_to_robot_signs={_fmt(POLICY_TO_ROBOT_JOINT_SIGNS)} "
            f"policy_to_robot_offsets={_fmt(POLICY_TO_ROBOT_JOINT_OFFSETS_DEG)} "
            f"instruction={instruction!r}"
        )
        start = time.monotonic()
        period = 1.0 / hz if hz > 0 else 0.0
        try:
            while not self.stop_event.is_set() and time.monotonic() - start < duration_s:
                next_chunk = self.chunks + 1
                self.set_stage("read_state")
                measured = self.read_state()
                policy_state = robot_state_to_policy_state(measured)
                self.log(
                    f"chunk {next_chunk} capture start measured {_fmt(measured)} "
                    f"policy_state {_fmt(policy_state)}"
                )
                self.set_stage("capture")
                images: dict[str, np.ndarray] = {}
                capture_times = []
                for cam in self._policy_cameras():
                    t_cam = time.monotonic()
                    images[cam.name] = read_mjpeg_frame(cam.url)
                    capture_times.append(time.monotonic() - t_cam)
                    self.log(f"chunk {next_chunk} {cam.name} frame={capture_times[-1]:.2f}s")
                t0 = time.monotonic()
                self.set_stage("query")
                self.log(f"chunk {next_chunk} query start timeout={MOLMO_TIMEOUT_S:.0f}s")
                chunk = self.policy_client.act(
                    images=images,
                    state=policy_state,
                    instruction=instruction,
                    joints=JOINTS,
                )
                query_s = time.monotonic() - t0
                with self.status_lock:
                    self.last_query_s = round(query_s, 3)
                if self.stop_event.is_set():
                    self.log(f"chunk {next_chunk} query returned after stop; dropping chunk query={query_s:.1f}s")
                    break
                chunk = np.asarray(chunk, dtype=np.float32).reshape(-1, len(JOINTS))
                n = min(max(exec_steps, 0), len(chunk))
                with self.status_lock:
                    self.chunks += 1
                    chunk_num = self.chunks
                self.log(
                    f"chunk {chunk_num} query={query_s:.1f}s capture={sum(capture_times):.1f}s "
                    f"shape={chunk.shape} executing={n}"
                )
                cur_cmd = measured.copy()
                executed = 0
                last_measured = measured
                for k in range(n):
                    if self.stop_event.is_set() or time.monotonic() - start >= duration_s:
                        break
                    self.set_stage("execute")
                    model_target = policy_action_to_robot_target(chunk[k])
                    if effective_max_step_deg > 0:
                        target = cur_cmd + np.clip(
                            model_target - cur_cmd,
                            -effective_max_step_deg,
                            effective_max_step_deg,
                        )
                    else:
                        target = model_target
                    self.send_state(target)
                    after = self.read_state()
                    with self.status_lock:
                        self.steps += 1
                    executed += 1
                    last_measured = after
                    cur_cmd = target
                    if period > 0:
                        time.sleep(period)
                if executed:
                    with self.status_lock:
                        total_steps = self.steps
                    self.log(
                        f"chunk {chunk_num} executed={executed} total_steps={total_steps} "
                        f"last_measured {_fmt(last_measured)}"
                    )
        except BaseException as exc:
            self.set_error(exc)
        finally:
            if self.stop_event.is_set():
                self.log("policy stopped")
            else:
                self.log("policy complete")
            self.stop_success_tracking(join=True)
            self.set_stage("idle")
            self.set_mode("idle")

    def start_home(self, max_step_deg: float = HOME_STEP_DEG, hz: float = DEFAULT_HZ, from_eval: bool = False) -> None:
        if self._eval_running() and not from_eval:
            raise RuntimeError("continuous eval is running; stop it first")
        if self._motion_running():
            self.stop_policy()
            self.policy_thread.join(timeout=1.0)
            if self._motion_running():
                raise RuntimeError("motion is stopping; retry go home in a moment")
        self.stop_event.clear()
        with self.status_lock:
            self.steps = 0
            self.chunks = 0
            self.last_error = ""
            self.last_query_s = None
            self.started_at = time.monotonic()
        self.policy_thread = threading.Thread(
            target=self._home_loop,
            args=(max_step_deg, hz),
            daemon=True,
        )
        self.policy_thread.start()

    def _home_loop(self, max_step_deg: float, hz: float) -> None:
        self.set_mode("home")
        self.set_stage("home")
        home_pose = self.home_pose.copy()
        self.log(f"go home start target {_fmt(home_pose)} max_step={max_step_deg} hz={hz}")
        period = 1.0 / hz if hz > 0 else 0.0
        try:
            while not self.stop_event.is_set():
                measured = self.read_state()
                err = home_pose - measured
                if float(np.max(np.abs(err))) <= HOME_EPS_DEG:
                    self.send_state(home_pose)
                    after = self.read_state()
                    with self.status_lock:
                        self.last_state = [float(x) for x in after]
                    self.log(f"home reached {_fmt(after)}")
                    break
                if max_step_deg > 0:
                    target = measured + np.clip(err, -max_step_deg, max_step_deg)
                else:
                    target = home_pose.copy()
                self.send_state(target)
                after = self.read_state()
                with self.status_lock:
                    self.steps += 1
                    step_num = self.steps
                self.log(f"home step {step_num} target {_fmt(target)} measured {_fmt(after)}")
                if period > 0:
                    time.sleep(period)
        except BaseException as exc:
            self.set_error(exc)
        finally:
            if self.stop_event.is_set():
                self.log("home stopped")
            self.set_stage("idle")
            self.set_mode("idle")

    def save_current_as_home(self) -> list[float]:
        if self._eval_running():
            raise RuntimeError("continuous eval is running; stop it first")
        if self._motion_running():
            self.stop_policy()
            self.log("saving home while motion stops")
        state = self.read_state()
        self.home_pose = state.copy()
        save_home_pose(self.home_pose)
        self.log(f"saved home {_fmt(self.home_pose)} -> {HOME_POSE_PATH}")
        return [float(x) for x in self.home_pose]

    def nudge(self, joint: str, delta: float, lease_id: str | None = None) -> list[float]:
        self._require_teleop_lease(lease_id)
        if self._eval_running():
            raise RuntimeError("continuous eval is running; stop it first")
        if self._motion_running():
            self.stop_policy()
            self.log("manual nudge proceeding while policy stops")
        if joint not in JOINTS:
            raise ValueError(f"unknown joint {joint!r}")
        state = self.read_state()
        target = state.copy()
        target[JOINTS.index(joint)] += float(delta)
        self.send_state(target)
        time.sleep(0.1)
        after = self.read_state()
        self.log(f"nudge {joint} {delta:+.2f}: {_fmt(after)}")
        return [float(x) for x in after]

    def set_gripper(self, value: float, lease_id: str | None = None) -> list[float]:
        self._require_teleop_lease(lease_id)
        if self._eval_running():
            raise RuntimeError("continuous eval is running; stop it first")
        if self._motion_running():
            self.stop_policy()
            self.log("manual gripper proceeding while policy stops")
        state = self.read_state()
        target = state.copy()
        target[JOINTS.index("gripper")] = float(value)
        self.send_state(target)
        time.sleep(0.1)
        after = self.read_state()
        self.log(f"gripper {value:.2f}: {_fmt(after)}")
        return [float(x) for x in after]

HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO101 Control</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
body { margin: 0; background: #111; color: #eee; }
header { display:flex; align-items:center; justify-content:space-between; gap: 12px; padding: 12px 16px; border-bottom: 1px solid #333; }
h1 { margin: 0; font-size: 18px; }
.page-tabs { display:flex; align-items:center; gap: 6px; flex-wrap: wrap; }
.page-tab.active { background:#1f6feb; border-color:#2f81f7; }
.header-actions { display:flex; align-items:center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.button-link { display:inline-flex; align-items:center; color:#fff; text-decoration:none; background:#333; border:1px solid #555; border-radius:5px; padding:8px 10px; }
main { display: grid; grid-template-columns: minmax(320px, 1.2fr) minmax(320px, .8fr); gap: 14px; padding: 14px; }
.cams { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
.cams img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #000; border:1px solid #333; border-radius: 6px; }
.panel { border: 1px solid #333; border-radius: 6px; padding: 12px; background: #1b1b1b; }
.row { display:flex; gap: 8px; align-items:center; flex-wrap: wrap; margin: 8px 0; }
label { color:#bbb; font-size: 13px; }
input, textarea, select, button { font: inherit; }
input, textarea, select { background:#111; color:#eee; border:1px solid #444; border-radius:5px; padding: 7px; }
textarea { width: 100%; box-sizing: border-box; min-height: 62px; resize: vertical; }
button { background:#333; color:#fff; border:1px solid #555; border-radius:5px; padding:8px 10px; cursor:pointer; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button.danger { background:#9e2f2f; border-color:#c43b3b; }
button.good { background:#2d6b3f; border-color:#3c8a52; }
button:disabled { opacity: .45; cursor: not-allowed; }
.state { display:grid; grid-template-columns: 1.2fr .8fr 2.2fr; gap: 6px 10px; font-variant-numeric: tabular-nums; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.status { padding: 4px 8px; border-radius: 999px; background:#333; }
.success { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 12px; font-size: 13px; }
.success b { color:#fff; }
.success .ok { color:#70e08a; }
.success .warn { color:#ffd166; }
.eval { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 12px; font-size: 13px; margin: 8px 0; }
.eval b { color:#fff; }
.eval .ok { color:#70e08a; }
.eval .warn { color:#ffd166; }
.eval .bad { color:#ff8a8a; }
.teleop { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 12px; font-size: 13px; margin: 8px 0; }
.teleop b { color:#fff; }
.teleop .ok { color:#70e08a; }
.teleop .warn { color:#ffd166; }
.teleop .bad { color:#ff8a8a; }
.teleop-panel { margin-top:12px; }
.live-summary { display:flex; align-items:stretch; justify-content:space-between; gap: 10px; margin-top: 12px; }
.live-prompt { display:grid; grid-template-columns: 1fr; gap: 8px; margin-bottom: 8px; }
.live-prompt-item { min-width:0; background:#0b0b0b; border:1px solid #333; border-radius:5px; padding:7px 9px; font-size: 13px; }
.live-prompt-item span { display:block; color:#aaa; font-size: 11px; margin-bottom: 3px; }
.live-prompt-item b { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#fff; font-weight:600; }
.live-events { flex:1; display:flex; align-items:flex-start; align-content:flex-start; flex-wrap:wrap; gap:6px; min-height: 40px; max-height: 104px; overflow:hidden auto; background:#0b0b0b; border:1px solid #333; border-radius:5px; padding:7px; }
.live-stats { display:grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); gap: 8px; margin-top: 8px; font-size: 13px; }
.live-stat { background:#0b0b0b; border:1px solid #333; border-radius:5px; padding:8px; min-height: 34px; }
.live-stat span { display:block; color:#aaa; font-size: 11px; margin-bottom: 3px; }
.live-stat b { font-size: 16px; }
.live-stat em { display:block; color:#888; font-style:normal; font-size: 11px; margin-top: 3px; }
.live-stat.ok b { color:#70e08a; }
.live-stat.bad b { color:#ff8a8a; }
.live-stat.warn b { color:#ffd166; }
.errorbar { position:relative; height:8px; margin-top:7px; border-radius:999px; background:#262626; border:1px solid #383838; overflow:hidden; }
.errorbar-range { position:absolute; top:0; bottom:0; border-radius:999px; background:rgba(112, 224, 138, .38); }
.errorbar-point { position:absolute; top:-3px; bottom:-3px; width:2px; background:#fff; box-shadow:0 0 0 1px rgba(0,0,0,.55); }
.errorbar.empty { opacity:.35; }
.event-pill { display:inline-flex; align-items:center; justify-content:center; flex:0 0 auto; width: 32px; height: 28px; padding: 0; border-radius: 999px; font-size: 17px; line-height:1; font-weight: 700; white-space: nowrap; }
.event-pill.success { background: rgba(45, 107, 63, .35); border: 1px solid #3c8a52; }
.event-pill.failure { background: rgba(158, 47, 47, .35); border: 1px solid #c43b3b; color:#ffb4b4; }
.live-events.compact { gap:5px; }
.live-events.compact .event-pill { width: 27px; height: 24px; font-size: 15px; }
.live-events.dense { gap:4px; }
.live-events.dense .event-pill { width: 22px; height: 20px; font-size: 13px; }
.live-events.tiny { gap:3px; }
.live-events.tiny .event-pill { width: 18px; height: 17px; font-size: 11px; }
.monitor-actions { display:flex; align-items:center; justify-content:flex-end; gap: 8px; flex-wrap:wrap; }
.monitor-actions label { display:inline-flex; align-items:center; gap: 4px; }
.dataset-prompt { margin-top:12px; }
.dataset-prompt h2 { margin:0 0 8px; font-size:15px; }
.dataset-prompt p { margin:6px 0; color:#ccc; font-size:13px; }
.dataset-output { max-height:120px; overflow:auto; background:#0b0b0b; border:1px solid #333; border-radius:5px; padding:8px; white-space:pre-wrap; font-size:12px; }
.sam3-editor input[type=text] { flex: 1 1 260px; min-width: 0; }
.sam3-editor img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background:#000; border:1px solid #333; border-radius:5px; margin-top: 8px; }
.sam3-editor .sam3-status { min-height: 18px; color:#bbb; font-size: 12px; }
body.live-view main { grid-template-columns: 1fr; }
body.live-view.page-setup .control-panel { display:none; }
body.live-view .status-panel { display:none; }
body.live-view .cams { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
body.live-view .cams img { height: min(62vh, 680px); max-height: none; }
body.live-view .visual-pane { min-height: calc(100vh - 74px); }
body.page-setup .monitor-only { display:none !important; }
body.page-monitor .setup-only { display:none !important; }
body.page-monitor main { grid-template-columns: 1fr; }
body.page-monitor .control-panel { display:none; }
body.page-monitor .status-panel { display:none; }
body.page-monitor .cams { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
body.page-monitor .cams img { height: min(62vh, 680px); max-height: none; }
body.page-monitor .visual-pane { min-height: calc(100vh - 74px); }
body.page-monitor #liveToggle { display:none; }
.hidden { display:none !important; }
.log { height: 220px; overflow:auto; background:#0b0b0b; border:1px solid #333; border-radius:5px; padding:8px; white-space:pre-wrap; font-size:12px; }
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; }
  .cams, body.live-view .cams, body.page-monitor .cams { grid-template-columns: 1fr; }
  body.live-view .cams img, body.page-monitor .cams img { height:auto; }
  .live-prompt { grid-template-columns: 1fr; }
  .live-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
</head>
<body class="page-setup">
<header>
  <h1>SO101 Control</h1>
  <nav class="page-tabs" aria-label="pages">
    <button id="setupTab" class="page-tab active" onclick="setPage('setup')">Setup</button>
    <button id="monitorTab" class="page-tab" onclick="setPage('monitor')">Continuous Monitor</button>
  </nav>
  <div class="header-actions">
    <a class="button-link" href="/episodes" target="_blank">Episodes</a>
    <button id="liveToggle" onclick="toggleLiveView()">Live View</button>
    <span id="mode" class="status">...</span>
  </div>
</header>
<main>
	  <section class="visual-pane">
	    <div class="live-prompt mono" id="livePrompt"></div>
	    <div class="cams">
	      <img id="cam0" alt="camera 0">
      <img id="cam1" alt="camera 1">
      <img id="cam2" alt="camera 2">
      <img id="successOverlay" alt="success tracking overlay">
    </div>
    <div class="live-summary monitor-only">
      <div class="live-events" id="liveEvents"></div>
      <div class="monitor-actions">
        <label><input id="evalWithTeleop" type="checkbox" checked> w teleop</label>
        <label><input id="evalRecordEpisodes" type="checkbox" checked> record episodes</label>
        <button id="startEvalButton" class="primary" onclick="startEval()">Start</button>
        <button class="primary" onclick="resumeEval()">Resume</button>
        <button id="liveStopEvalButton" class="danger" onclick="stopEval()">Stop</button>
        <button onclick="clearEval()">Clear</button>
        <span id="evalStatus" class="mono"></span>
      </div>
    </div>
    <div class="live-stats mono monitor-only" id="liveStats"></div>
    <div class="panel dataset-prompt monitor-only hidden" id="datasetPrompt"></div>
    <div class="panel teleop-panel monitor-only">
      <label>Teleop Intervention</label>
      <div class="teleop mono" id="teleop"></div>
      <div class="row">
        <label>Operator <input id="teleopOperator" type="text" value="operator" style="width:130px"></label>
        <button id="claimTeleopButton" class="primary" onclick="claimTeleop()">Claim Teleop</button>
        <button id="releaseTeleopButton" onclick="releaseTeleop()" disabled>Release Teleop</button>
        <button id="releaseResumeButton" class="good" onclick="releaseAndResumeEval()" disabled>Release + Resume Eval</button>
      </div>
      <div class="row">
        <select id="joint"></select>
        <button class="teleop-command" onclick="nudge(-20)" disabled>-20</button>
        <button class="teleop-command" onclick="nudge(-10)" disabled>-10</button>
        <button class="teleop-command" onclick="nudge(-2)" disabled>-2</button>
        <button class="teleop-command" onclick="nudge(-0.5)" disabled>-0.5</button>
        <button class="teleop-command" onclick="nudge(0.5)" disabled>+0.5</button>
        <button class="teleop-command" onclick="nudge(2)" disabled>+2</button>
        <button class="teleop-command" onclick="nudge(10)" disabled>+10</button>
        <button class="teleop-command" onclick="nudge(20)" disabled>+20</button>
      </div>
      <div class="row">
        <button class="teleop-command" onclick="gripper(2)" disabled>Gripper Closed</button>
        <button class="teleop-command" onclick="gripper(45)" disabled>Gripper Open</button>
      </div>
    </div>
    <div class="panel status-panel setup-only" style="margin-top:12px">
      <div class="success mono" id="success"></div>
    </div>
    <div class="panel status-panel setup-only" style="margin-top:12px">
      <div class="state mono" id="state"></div>
    </div>
  </section>
  <section class="panel control-panel">
    <label class="setup-only">Instruction</label>
    <textarea id="instruction" class="setup-only">Move to blue ball, grab it, and place it in the cardboard cylinder</textarea>
    <div class="row setup-only">
      <label>Duration <input id="duration" type="number" value="300" min="1" max="900" style="width:72px"></label>
      <label>Steps/chunk <input id="execSteps" type="number" value="30" min="1" max="30" style="width:72px"></label>
      <label>Max step deg <input id="maxStep" type="number" value="1" min="0.1" max="10" step="0.1" style="width:72px"></label>
      <label>Hz <input id="hz" type="number" value="30" min="1" max="60" step="0.5" style="width:64px"></label>
    </div>
    <div class="row setup-only">
      <button class="primary" onclick="startPolicy()">Start MolmoAct</button>
      <button class="danger" onclick="stopPolicy()">Stop</button>
      <button class="good" onclick="goHome()">Go Home</button>
      <button class="good" onclick="saveHome()">Save As Home</button>
      <button onclick="connectRobot()">Connect/Read</button>
      <button onclick="disconnectRobot()">Disconnect</button>
      <button onclick="resetSuccess()">Reset Success</button>
    </div>
    <hr class="setup-only" style="border-color:#333">
    <div class="sam3-editor setup-only">
      <label>SAM3 Container Prompt</label>
      <div class="row">
        <input id="sam3Prompt" type="text" value="cardboard cylinder along with the insides">
      </div>
      <div class="row">
        <label>Confidence <input id="sam3MinScore" type="number" value="0.15" min="0" max="1" step="0.01" style="width:72px"></label>
        <label>Camera <select id="sam3Camera"><option value="front">Front</option><option value="side">Side</option><option value="wrist">Wrist</option></select></label>
        <button id="sam3PreviewButton" onclick="previewSam3()">Preview SAM3</button>
        <button class="good" onclick="applySam3Prompt()">Use For Success</button>
      </div>
      <div id="sam3PreviewStatus" class="sam3-status mono"></div>
      <img id="sam3Preview" alt="SAM3 prompt preview">
    </div>
    <div class="hidden" aria-hidden="true">
      <input id="evalHours" type="number" value="1" min="0.05" max="8" step="0.05">
      <input id="evalAttempt" type="number" value="60" min="5" max="600" step="1">
      <input id="evalMaxFails" type="number" value="4" min="1" max="10" step="1">
      <input id="evalRecordFps" type="number" value="30" min="0.5" max="30" step="0.5">
    </div>
    <div class="eval mono hidden" id="eval"></div>
    <hr class="hidden" style="border-color:#333">
    <div class="row hidden" aria-hidden="true">
      <label>Record duration <input id="recordDuration" type="number" value="120" min="1" max="900" style="width:72px"></label>
      <label>Record FPS <input id="recordFps" type="number" value="30" min="0.5" max="30" step="0.5" style="width:64px"></label>
      <label><input id="recordFront" type="checkbox" checked> Front</label>
      <label><input id="recordSide" type="checkbox" checked> Side</label>
      <label><input id="recordWrist" type="checkbox" checked> Wrist</label>
    </div>
    <div class="row hidden" aria-hidden="true">
      <button class="primary" onclick="startRecording()">Start Recording</button>
      <button class="danger" onclick="stopRecording()">Stop Recording</button>
      <span id="recordStatus" class="mono"></span>
    </div>
    <div class="log mono" id="log"></div>
  </section>
</main>
<script>
const host = location.hostname;
document.getElementById('successOverlay').src = `/api/success.mjpg`;
let liveViewPinned = false;
let liveViewSuppressed = false;
let sam3PromptDirty = false;
let sam3MinScoreDirty = false;
let lastStatus = null;
let teleopLeaseId = sessionStorage.getItem('so101TeleopLeaseId') || '';
const teleopOperatorKey = 'so101TeleopOperator';
let datasetPromptDismissedFor = sessionStorage.getItem('so101DatasetPromptDismissedFor') || '';
const cameraTimers = [];

function setPage(page) {
  const selected = page === 'monitor' ? 'monitor' : 'setup';
  document.body.classList.toggle('page-setup', selected === 'setup');
  document.body.classList.toggle('page-monitor', selected === 'monitor');
  document.getElementById('setupTab')?.classList.toggle('active', selected === 'setup');
  document.getElementById('monitorTab')?.classList.toggle('active', selected === 'monitor');
  localStorage.setItem('so101ControlPage', selected);
  if (location.hash !== `#${selected}`) history.replaceState(null, '', `#${selected}`);
  updateLiveView(lastStatus || {});
}
window.addEventListener('hashchange', () => {
  setPage(location.hash.replace('#', '') || localStorage.getItem('so101ControlPage') || 'setup');
});

function startLowLatencyImage(id, path, intervalMs) {
  const el = document.getElementById(id);
  if (!el) return;
  let pending = false;
  let lastOkSrc = '';
  const tick = () => {
    if (pending) return;
    pending = true;
    const img = new Image();
    img.onload = () => {
      lastOkSrc = img.src;
      el.src = lastOkSrc;
      pending = false;
    };
    img.onerror = () => {
      pending = false;
      if (lastOkSrc) el.src = lastOkSrc;
    };
    img.src = `${path}?t=${Date.now()}`;
  };
  tick();
  cameraTimers.push(setInterval(tick, intervalMs));
}

startLowLatencyImage('cam0', '/cam0.jpg', 125);
startLowLatencyImage('cam1', '/cam1.jpg', 125);
startLowLatencyImage('cam2', '/cam2.jpg', 125);

async function api(path, body) {
  const opts = body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const res = await fetch(path, opts);
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}
function val(id) { return document.getElementById(id).value; }
const teleopOperatorInput = document.getElementById('teleopOperator');
if (teleopOperatorInput) {
  teleopOperatorInput.value = localStorage.getItem(teleopOperatorKey) || 'operator';
  teleopOperatorInput.addEventListener('input', () => {
    localStorage.setItem(teleopOperatorKey, teleopOperatorInput.value || 'operator');
  });
}
document.getElementById('sam3Prompt')?.addEventListener('input', () => { sam3PromptDirty = true; });
document.getElementById('sam3MinScore')?.addEventListener('input', () => { sam3MinScoreDirty = true; });
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
}
function toggleLiveView() {
  const active = document.body.classList.contains('live-view');
  if (active) {
    liveViewPinned = false;
    liveViewSuppressed = true;
    document.body.classList.remove('live-view');
  } else {
    liveViewPinned = true;
    liveViewSuppressed = false;
    document.body.classList.add('live-view');
  }
  updateLiveView(lastStatus || {});
}
function resetSuccessStream() {
  const img = document.getElementById('successOverlay');
  if (img) img.src = `/api/success.mjpg?t=${Date.now()}`;
}
async function startPolicy() {
  await api('/api/start', {
    instruction: val('instruction'),
    duration_s: Number(val('duration')),
    exec_steps: Number(val('execSteps')),
    max_step_deg: Number(val('maxStep')),
    hz: Number(val('hz')),
  });
  resetSuccessStream();
  await refresh();
}
async function stopPolicy() { await api('/api/stop', {}); await refresh(); }
async function goHome() { await api('/api/home', {}); await refresh(); }
async function saveHome() { await api('/api/save_home', {}); await refresh(); }
async function connectRobot() { await api('/api/connect', {}); await refresh(); }
async function disconnectRobot() { await api('/api/disconnect', {}); await refresh(); }
async function resetSuccess() { await api('/api/success/reset', {}); resetSuccessStream(); await refresh(); }
function sam3EditorConfig() {
  return {
    prompt: val('sam3Prompt'),
    min_score: Number(val('sam3MinScore')),
    camera: val('sam3Camera'),
  };
}
function setSam3PreviewStatus(text) {
  const el = document.getElementById('sam3PreviewStatus');
  if (el) el.textContent = text || '';
}
async function previewSam3() {
  if (lastStatus && (lastStatus.mode === 'policy' || lastStatus.mode === 'stopping' || lastStatus.eval?.running)) {
    setSam3PreviewStatus('preview disabled while policy/eval is running');
    return;
  }
  const img = document.getElementById('sam3Preview');
  setSam3PreviewStatus('requesting');
  try {
    const data = await api('/api/sam3/preview', sam3EditorConfig());
    if (img && data.overlay) img.src = data.overlay;
    const top = data.top_mask || {};
    const score = top.score === undefined ? '' : ` score=${Number(top.score).toFixed(2)}`;
    const area = top.area_px === undefined ? '' : ` area=${top.area_px}`;
    const box = Array.isArray(top.box_xyxy) ? ` box=${top.box_xyxy.map(v => Number(v).toFixed(0)).join(',')}` : '';
    setSam3PreviewStatus(`${data.status || ''}${score}${area}${box} ${data.elapsed_s || ''}s`);
  } catch (e) {
    setSam3PreviewStatus(e.message);
  }
}
async function applySam3Prompt() {
  const cfg = sam3EditorConfig();
  await api('/api/success/sam3', {prompt: cfg.prompt, min_score: cfg.min_score});
  sam3PromptDirty = false;
  sam3MinScoreDirty = false;
  resetSuccessStream();
  setSam3PreviewStatus(`using "${cfg.prompt}" min=${cfg.min_score}`);
  await refresh();
}
async function startEval() {
  await api('/api/eval/start', {
    instruction: val('instruction'),
    run_duration_s: Number(val('evalHours')) * 3600,
    attempt_duration_s: Number(val('evalAttempt')),
    max_consecutive_failures: Number(val('evalMaxFails')),
    record_fps: Number(val('evalRecordFps')),
    exec_steps: Number(val('execSteps')),
    max_step_deg: Number(val('maxStep')),
    hz: Number(val('hz')),
    allow_teleop: !!document.getElementById('evalWithTeleop')?.checked,
    record_episodes: !!document.getElementById('evalRecordEpisodes')?.checked,
  });
  resetSuccessStream();
  await refresh();
}
async function resumeEval() { await api('/api/eval/resume', {}); resetSuccessStream(); await refresh(); }
function setEvalButtonBusy(busy) {
  const idleLabels = {stopEvalButton: 'Stop Eval', liveStopEvalButton: 'Stop'};
  for (const id of Object.keys(idleLabels)) {
    const button = document.getElementById(id);
    if (button) {
      button.disabled = busy;
      button.textContent = busy ? 'Stopping...' : idleLabels[id];
    }
  }
}
async function stopEval() {
  const statusEl = document.getElementById('evalStatus');
  if (statusEl) statusEl.textContent = 'stopping eval, policy, tracker, and recording...';
  setEvalButtonBusy(true);
  try {
    await api('/api/eval/stop', {});
  } finally {
    setEvalButtonBusy(false);
    resetSuccessStream();
    await refresh();
  }
}
async function clearEval() {
  await api('/api/eval/clear', {});
  resetSuccessStream();
  await refresh();
}
function rememberTeleopLease(data) {
  const lease = data?.teleop?.lease || data?.lease || {};
  teleopLeaseId = lease.lease_id || '';
  if (teleopLeaseId) {
    sessionStorage.setItem('so101TeleopLeaseId', teleopLeaseId);
  } else {
    sessionStorage.removeItem('so101TeleopLeaseId');
  }
}
async function claimTeleop() {
  const data = await api('/api/teleop/claim', {operator: val('teleopOperator') || 'operator'});
  rememberTeleopLease(data);
  await refresh();
}
async function heartbeatTeleop() {
  if (!teleopLeaseId) return;
  try {
    await api('/api/teleop/heartbeat', {lease_id: teleopLeaseId});
  } catch (e) {
    teleopLeaseId = '';
    sessionStorage.removeItem('so101TeleopLeaseId');
    await refresh();
  }
}
async function releaseTeleop(outcome = 'complete') {
  if (!teleopLeaseId) return;
  try {
    await api('/api/teleop/release', {lease_id: teleopLeaseId, outcome});
  } finally {
    teleopLeaseId = '';
    sessionStorage.removeItem('so101TeleopLeaseId');
    await refresh();
  }
}
async function releaseAndResumeEval() {
  await releaseTeleop('resume_eval');
  await resumeEval();
}
async function nudge(delta) {
  await api('/api/nudge', {joint: val('joint'), delta, lease_id: teleopLeaseId});
  await refresh();
}
async function gripper(value) {
  await api('/api/gripper', {value, lease_id: teleopLeaseId});
  await refresh();
}
function selectedCameras() {
  const cams = [];
  if (document.getElementById('recordFront').checked) cams.push('front');
  if (document.getElementById('recordSide').checked) cams.push('side');
  if (document.getElementById('recordWrist').checked) cams.push('wrist');
  return cams;
}
async function startRecording() {
  await api('/api/record/start', {
    duration_s: Number(val('recordDuration')),
    fps: Number(val('recordFps')),
    cameras: selectedCameras(),
    task: val('instruction'),
  });
  await refresh();
}
async function stopRecording() { await api('/api/record/stop', {}); await refresh(); }
function renderState(data) {
  const el = document.getElementById('state');
  const joints = data.joints || [];
  const state = data.state || [];
  const home = data.home || [];
  if (!joints.length) return;
  const sel = document.getElementById('joint');
  if (!sel.children.length) joints.forEach(j => sel.add(new Option(j, j)));
  const rows = [`<div>joint</div><div>current</div><div>home</div>`];
  rows.push(...joints.map((j, i) => (
    `<div>${j}</div><div>${state[i] === undefined ? '' : state[i].toFixed(2)}</div><div>${home[i] === undefined ? '' : home[i].toFixed(2)}</div>`
  )));
  el.innerHTML = rows.join('');
}
function renderSuccess(data) {
  const el = document.getElementById('success');
  const s = data.success_tracking || {};
  const promptInput = document.getElementById('sam3Prompt');
  if (promptInput && !sam3PromptDirty && document.activeElement !== promptInput && s.container_mask_prompt) {
    promptInput.value = s.container_mask_prompt;
  }
  const minScoreInput = document.getElementById('sam3MinScore');
  if (minScoreInput && !sam3MinScoreDirty && document.activeElement !== minScoreInput && s.container_mask_min_score !== undefined && s.container_mask_min_score !== null) {
    minScoreInput.value = s.container_mask_min_score;
  }
  const overlap = s.current_overlap === null || s.current_overlap === undefined ? 'missing' : s.current_overlap.toFixed(3);
  const last = s.last_success
    ? `#${s.last_success.success} over=${s.last_success.over_start_frame} leave=${s.last_success.leave_frame}`
    : '';
  const cls = s.success_count ? 'ok' : (s.over_active ? 'warn' : '');
  const containerMask = `gen ${s.container_mask_generation || 0} ${s.container_mask_source || ''} area=${s.container_mask_area || 0}`;
  const containerScore = s.container_mask_score === null || s.container_mask_score === undefined
    ? `min ${s.container_mask_min_score || ''}`
    : `${s.container_mask_score.toFixed(2)} / min ${s.container_mask_min_score || ''}`;
  const sam3RawScore = s.container_mask_sam3_raw_score === null || s.container_mask_sam3_raw_score === undefined
    ? ''
    : ` score=${s.container_mask_sam3_raw_score.toFixed(2)}`;
  const sam3RawArea = s.container_mask_sam3_raw_area === null || s.container_mask_sam3_raw_area === undefined
    ? ''
    : ` area=${s.container_mask_sam3_raw_area}`;
  const sam3Box = Array.isArray(s.container_mask_sam3_box_xyxy)
    ? ` box=${s.container_mask_sam3_box_xyxy.map(v => Number(v).toFixed(0)).join(',')}`
    : '';
  const sam3Diag = `${s.container_mask_sam3_status || ''}${sam3RawScore}${sam3RawArea}${sam3Box} ${s.container_mask_sam3_error || ''}`;
  el.innerHTML = [
    `<div>successes</div><div class="${cls}"><b>${s.success_count || 0}</b></div>`,
    `<div>tracker</div><div>${s.running ? 'running' : 'idle'} / ${s.state || ''}</div>`,
    `<div>mask overlap</div><div>${overlap}</div>`,
    `<div>ball area</div><div>${s.ball_area || 0}</div>`,
    `<div>container mask</div><div>${containerMask}</div>`,
    `<div>SAM3 prompt</div><div>${s.container_mask_prompt || ''} ${containerScore}</div>`,
    `<div>SAM3 status</div><div>${sam3Diag}</div>`,
    `<div>over run</div><div>${s.over_len || 0} frame(s)</div>`,
    `<div>last</div><div>${last}</div>`,
    `<div>error</div><div>${s.error || ''}</div>`,
  ].join('');
}
function renderEval(data) {
  const el = document.getElementById('eval');
  const e = data.eval || {};
  const running = e.running ? 'running' : 'idle';
  const last = e.last_attempt
    ? `run ${e.last_attempt.run || e.last_attempt.attempt} ${e.last_attempt.outcome} ${e.last_attempt.reason}`
    : '';
  const cls = e.waiting_for_intervention ? 'bad' : e.running ? 'warn' : (e.successes ? 'ok' : '');
  const remaining = e.remaining_s === null || e.remaining_s === undefined ? '' : `${e.remaining_s}s left`;
  const noSuccessRemaining = e.attempt_remaining_s === null || e.attempt_remaining_s === undefined ? '' : `${e.attempt_remaining_s}s`;
  document.getElementById('evalStatus').textContent = `${running} ${e.state || ''} ${remaining}`;
  const teleopOption = document.getElementById('evalWithTeleop');
  if (teleopOption && e.config && e.config.allow_teleop !== undefined && !e.running) teleopOption.checked = !!e.config.allow_teleop;
  const recordOption = document.getElementById('evalRecordEpisodes');
  if (recordOption && e.config && e.config.record_episodes !== undefined && !e.running) recordOption.checked = !!e.config.record_episodes;
  el.innerHTML = [
    `<div>state</div><div class="${cls}"><b>${e.state || 'idle'}</b></div>`,
    `<div>policy run</div><div>${e.attempt || 0}</div>`,
    `<div>episode timer</div><div>${noSuccessRemaining}</div>`,
    `<div>success / fail</div><div>${e.successes || 0} / ${e.failures || 0}</div>`,
    `<div>fail streak</div><div>${e.consecutive_failures || 0} / ${e.max_consecutive_failures || 0}</div>`,
    `<div>last</div><div>${last}</div>`,
    `<div>w teleop</div><div>${e.config?.allow_teleop === false ? 'off' : 'on'}</div>`,
    `<div>record episodes</div><div>${e.config?.record_episodes === false ? 'off' : 'on'}</div>`,
    `<div>recording</div><div>${e.recording_dir || ''}</div>`,
    `<div>summary</div><div>${e.summary_path || ''}</div>`,
    `<div>reason</div><div>${e.stop_reason || e.error || ''}</div>`,
  ].join('');
}
function renderTeleop(data) {
  const el = document.getElementById('teleop');
  const t = data.teleop || {};
  const e = data.eval || {};
  const lease = t.lease || {};
  const heldHere = !!teleopLeaseId && !!lease.lease_id && lease.lease_id === teleopLeaseId && !t.stale;
  if (teleopLeaseId && (!t.active || (lease.lease_id && lease.lease_id !== teleopLeaseId) || t.stale)) {
    teleopLeaseId = '';
    sessionStorage.removeItem('so101TeleopLeaseId');
  }
  const active = !!t.active && !t.stale;
  const status = active
    ? (heldHere ? 'claimed here' : `claimed by ${lease.operator || 'operator'}`)
    : t.stale ? 'stale' : t.available ? 'available' : 'not available';
  const cls = heldHere ? 'ok' : active ? 'warn' : e.waiting_for_intervention ? 'bad' : '';
  const reason = lease.reason || e.stop_reason || '';
  const age = t.heartbeat_age_s === null || t.heartbeat_age_s === undefined ? '' : `${t.heartbeat_age_s}s`;
  const canReleaseResume = heldHere && !!e.config && e.resume_remaining_s !== null && e.resume_remaining_s !== undefined && e.resume_remaining_s > 1;
  const claimButton = document.getElementById('claimTeleopButton');
  const releaseButton = document.getElementById('releaseTeleopButton');
  const releaseResumeButton = document.getElementById('releaseResumeButton');
  if (claimButton) claimButton.disabled = heldHere || (!t.available && !t.stale);
  if (releaseButton) releaseButton.disabled = !heldHere;
  if (releaseResumeButton) releaseResumeButton.disabled = !canReleaseResume;
  document.querySelectorAll('.teleop-command').forEach(button => { button.disabled = !heldHere; });
  if (el) {
    el.innerHTML = [
      `<div>state</div><div class="${cls}"><b>${status}</b></div>`,
      `<div>intervention</div><div>${t.intervention_ready ? 'ready' : 'manual'}</div>`,
      `<div>operator</div><div>${lease.operator || '-'}</div>`,
      `<div>heartbeat</div><div>${age}</div>`,
      `<div>reason</div><div>${escapeHtml(reason || '-')}</div>`,
      `<div>resume</div><div>${canReleaseResume ? 'ready after release' : e.can_resume ? 'ready' : '-'}</div>`,
    ].join('');
  }
}
function renderLiveEvents(data) {
  const el = document.getElementById('liveEvents');
  const history = ((data.eval || {}).history || []).filter(ev => ev && (ev.type === 'success' || ev.type === 'failure')).slice(-80);
  el.classList.toggle('compact', history.length > 12);
  el.classList.toggle('dense', history.length > 28);
  el.classList.toggle('tiny', history.length > 48);
  el.innerHTML = history.map(ev => {
    const ok = ev.type === 'success' || ev.outcome === 'success';
    const label = ok ? '✓' : '×';
    const cls = ok ? 'success' : 'failure';
    const stamp = ev.at || ev.ended_at || '';
    const title = `${stamp} run ${ev.run || ev.attempt || ''} ${ev.reason || ev.outcome || ''}`;
    return `<span class="event-pill ${cls}" title="${escapeHtml(title)}">${label}</span>`;
  }).join('');
}
function renderLivePrompt(data) {
  const el = document.getElementById('livePrompt');
  if (!el) return;
  const e = data.eval || {};
  const instructionInput = document.getElementById('instruction');
  const taskPrompt = e.config?.instruction || instructionInput?.value || '';
  el.innerHTML = [
    `<div class="live-prompt-item"><span>task prompt</span><b title="${escapeHtml(taskPrompt)}">${escapeHtml(taskPrompt || '-')}</b></div>`,
  ].join('');
}
function clampPercent(value) {
  return Math.max(0, Math.min(100, value));
}
function errorBar(ci) {
  if (!ci) return `<div class="errorbar empty"></div><em>95% CI -</em>`;
  const left = clampPercent(ci.loPct);
  const right = clampPercent(ci.hiPct);
  const width = Math.max(0, right - left);
  const point = clampPercent(ci.pointPct);
  return [
    `<div class="errorbar" title="95% CI ${ci.text}">`,
    `<div class="errorbar-range" style="left:${left}%;width:${width}%"></div>`,
    `<div class="errorbar-point" style="left:${point}%"></div>`,
    `</div><em>95% CI ${ci.text}</em>`,
  ].join('');
}
function wilsonPercentCi(successes, total) {
  if (!total) return null;
  const z = 1.96;
  const p = successes / total;
  const denom = 1 + (z * z) / total;
  const center = (p + (z * z) / (2 * total)) / denom;
  const half = (z * Math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total)) / denom;
  const lo = Math.max(0, Math.round((center - half) * 100));
  const hi = Math.min(100, Math.round((center + half) * 100));
  return {text: `${lo}-${hi}%`, loPct: lo, hiPct: hi, pointPct: p * 100};
}
function throughputCi(successes, elapsedMin) {
  if (!elapsedMin) return null;
  if (!successes) {
    const hi = 3 / elapsedMin;
    return {text: `0.00-${hi.toFixed(2)}/min`, loPct: 0, hiPct: 100, pointPct: 0};
  }
  const z = 1.96;
  const point = successes / elapsedMin;
  const lo = Math.max(0, (successes - z * Math.sqrt(successes)) / elapsedMin);
  const hi = (successes + z * Math.sqrt(successes)) / elapsedMin;
  const scale = Math.max(hi, point, 1e-9);
  return {
    text: `${lo.toFixed(2)}-${hi.toFixed(2)}/min`,
    loPct: (lo / scale) * 100,
    hiPct: (hi / scale) * 100,
    pointPct: (point / scale) * 100,
  };
}
function renderLiveStats(data) {
  const el = document.getElementById('liveStats');
  const e = data.eval || {};
  const successes = Number(e.successes || 0);
  const failures = Number(e.failures || 0);
  const total = successes + failures;
  const successRate = total ? `${Math.round((successes / total) * 100)}%` : '-';
  const successRateCi = wilsonPercentCi(successes, total);
  const successRateCiText = successRateCi ? successRateCi.text : '-';
  const elapsedMin = e.elapsed_s ? Number(e.elapsed_s) / 60 : 0;
  const throughput = elapsedMin > 0 ? `${(successes / elapsedMin).toFixed(2)}/min` : '-';
  const throughputCiResult = throughputCi(successes, elapsedMin);
  const throughputCiText = throughputCiResult ? throughputCiResult.text : '-';
  const noSuccessRemaining = e.attempt_remaining_s === null || e.attempt_remaining_s === undefined ? '-' : `${e.attempt_remaining_s}s`;
  const elapsed = e.elapsed_s === null || e.elapsed_s === undefined ? '-' : `${e.elapsed_s}s`;
  const remaining = e.remaining_s === null || e.remaining_s === undefined ? '-' : `${e.remaining_s}s`;
  const last = e.last_attempt ? `${e.last_attempt.outcome || ''} ${e.last_attempt.reason || ''}`.trim() : '-';
  el.innerHTML = [
    `<div class="live-stat ok"><span>success rate</span><b>${successRate}</b></div>`,
    `<div class="live-stat ok"><span>confidence interval</span><b>${successRateCiText}</b>${errorBar(successRateCi)}</div>`,
    `<div class="live-stat ok"><span>throughput</span><b>${throughput}</b><em>95% CI ${throughputCiText}</em></div>`,
    `<div class="live-stat"><span>success / fail</span><b>${successes} / ${failures}</b></div>`,
    `<div class="live-stat ${e.consecutive_failures ? 'bad' : ''}"><span>fail streak</span><b>${e.consecutive_failures || 0}/${e.max_consecutive_failures || 0}</b></div>`,
    `<div class="live-stat warn"><span>episode timer</span><b>${noSuccessRemaining}</b></div>`,
    `<div class="live-stat"><span>policy run</span><b>${e.attempt || 0}</b></div>`,
    `<div class="live-stat"><span>elapsed</span><b>${elapsed}</b></div>`,
    `<div class="live-stat"><span>remaining</span><b>${remaining}</b></div>`,
    `<div class="live-stat"><span>last event</span><b>${escapeHtml(last)}</b></div>`,
  ].join('');
}
function datasetFingerprint(dataset) {
  return `${dataset.success_count || 0}:${dataset.failure_count || 0}:${dataset.uncompressed_success_count || 0}:${dataset.started_at || ''}:${dataset.completed_at || ''}`;
}
async function compressDataset(upload = false) {
  const body = {upload};
  if (upload) {
    const prior = localStorage.getItem('so101HfDatasetRepoId') || '';
    const repoId = prompt('HF dataset repo id', prior || 'username/so101-ball-cup-success');
    if (!repoId) return;
    body.repo_id = repoId.trim();
    localStorage.setItem('so101HfDatasetRepoId', body.repo_id);
  }
  await api('/api/dataset/compress', body);
  datasetPromptDismissedFor = '';
  sessionStorage.removeItem('so101DatasetPromptDismissedFor');
  await refresh();
}
function dismissDatasetPrompt(dataset) {
  datasetPromptDismissedFor = datasetFingerprint(dataset || {});
  sessionStorage.setItem('so101DatasetPromptDismissedFor', datasetPromptDismissedFor);
  renderDatasetPrompt(lastStatus || {});
}
function renderDatasetPrompt(data) {
  const el = document.getElementById('datasetPrompt');
  if (!el) return;
  const dataset = data.dataset || {};
  const e = data.eval || {};
  const pending = Number(dataset.uncompressed_success_count || 0);
  const running = !!dataset.running;
  const show = running || pending > 0 || ['complete', 'failed'].includes(dataset.state || '');
  const fingerprint = datasetFingerprint(dataset);
  if (!show || (!running && pending > 0 && datasetPromptDismissedFor === fingerprint) || e.running) {
    el.classList.add('hidden');
    el.innerHTML = '';
    return;
  }
  el.classList.remove('hidden');
  const output = (dataset.output_tail || []).slice(-12).map(escapeHtml).join('\n');
  const status = running
    ? `${dataset.state || 'running'} ${dataset.repo_id || ''}`
    : pending > 0
      ? `${pending} success episode(s) ready`
      : `${dataset.state || 'idle'} ${dataset.error || dataset.repo_id || ''}`;
  const disabled = running ? 'disabled' : '';
  el.innerHTML = [
    `<h2>Dataset</h2>`,
    `<p class="mono">${escapeHtml(status)}</p>`,
    `<div class="row">`,
    `<button class="primary" ${disabled} onclick="compressDataset(false)">Compress Successes</button>`,
    `<button class="good" ${disabled} onclick="compressDataset(true)">Compress + Upload</button>`,
    `<button onclick="window.open('/episodes', '_blank')">Open Episodes</button>`,
    `<button ${running ? 'disabled' : ''} onclick="dismissDatasetPrompt(lastStatus?.dataset || {})">Dismiss</button>`,
    `</div>`,
    output ? `<div class="dataset-output mono">${output}</div>` : '',
  ].join('');
}
function updateLiveView(data) {
  const e = data.eval || {};
  const autoActive = !!e.running || data.mode === 'policy';
  if (!autoActive && !liveViewPinned) liveViewSuppressed = false;
  const active = !liveViewSuppressed && (liveViewPinned || autoActive);
  document.body.classList.toggle('live-view', active);
  const button = document.getElementById('liveToggle');
  if (button) {
    button.textContent = active ? 'Exit Live View' : 'Live View';
  }
}
let refreshInFlight = false;
let lastLogText = '';
async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const data = await api('/api/status?log_limit=60');
    lastStatus = data;
    const query = data.last_query_s === null || data.last_query_s === undefined ? '' : ` query=${data.last_query_s}s`;
    document.getElementById('mode').textContent = `${data.mode}/${data.stage} ${data.connected ? 'connected' : 'disconnected'} steps=${data.steps} chunks=${data.chunks}${query}`;
    const rec = data.recording || {};
    const recCounts = rec.counts ? Object.entries(rec.counts).map(([k, v]) => `${k}=${v}`).join(' ') : '';
    const recDir = rec.dir ? ` ${rec.dir}` : '';
    const recErr = rec.error ? ` error=${rec.error}` : '';
    document.getElementById('recordStatus').textContent = rec.running
      ? `recording ${rec.elapsed || 0}s ${recCounts}${recDir}${recErr}`
      : rec.dir ? `idle ${recCounts}${recDir}${recErr}` : '';
    renderSuccess(data);
    renderEval(data);
    renderTeleop(data);
    renderLivePrompt(data);
    renderLiveEvents(data);
    renderLiveStats(data);
    renderDatasetPrompt(data);
    updateLiveView(data);
    const sam3PreviewButton = document.getElementById('sam3PreviewButton');
    if (sam3PreviewButton) {
      sam3PreviewButton.disabled = data.mode === 'policy' || data.mode === 'stopping' || !!data.eval?.running;
    }
    renderState(data);
    const nextLogText = (data.logs || []).join('\n');
    const log = document.getElementById('log');
    if (nextLogText !== lastLogText) {
      log.textContent = nextLogText;
      log.scrollTop = log.scrollHeight;
      lastLogText = nextLogText;
    }
  } catch (e) {
    document.getElementById('mode').textContent = e.message;
  } finally {
    refreshInFlight = false;
  }
}
setPage(location.hash.replace('#', '') || localStorage.getItem('so101ControlPage') || 'setup');
setInterval(refresh, 1500);
setInterval(heartbeatTeleop, 10000);
refresh();
</script>
</body>
</html>
"""


TEST_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO101 Eval Event Test</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
body { margin:0; background:#111; color:#eee; }
header { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 16px; border-bottom:1px solid #333; }
h1 { margin:0; font-size:18px; }
main { padding:14px; display:grid; gap:12px; }
button { background:#333; color:#fff; border:1px solid #555; border-radius:5px; padding:8px 10px; cursor:pointer; }
button.good { background:#2d6b3f; border-color:#3c8a52; }
button.danger { background:#9e2f2f; border-color:#c43b3b; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.panel { border:1px solid #333; border-radius:6px; background:#1b1b1b; padding:12px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.live-summary { display:flex; align-items:stretch; gap:10px; }
.live-events { flex:1; display:flex; align-items:flex-start; align-content:flex-start; flex-wrap:wrap; gap:6px; min-height:40px; max-height:220px; overflow:hidden auto; background:#0b0b0b; border:1px solid #333; border-radius:5px; padding:7px; }
.event-pill { display:inline-flex; align-items:center; justify-content:center; flex:0 0 auto; width:32px; height:28px; padding:0; border-radius:999px; font-size:17px; line-height:1; font-weight:700; white-space:nowrap; }
.event-pill.success { background: rgba(45, 107, 63, .35); border: 1px solid #3c8a52; color:#dcffe4; }
.event-pill.failure { background: rgba(158, 47, 47, .35); border: 1px solid #c43b3b; color:#ffb4b4; }
.live-events.compact { gap:5px; }
.live-events.compact .event-pill { width:27px; height:24px; font-size:15px; }
.live-events.dense { gap:4px; }
.live-events.dense .event-pill { width:22px; height:20px; font-size:13px; }
.live-events.tiny { gap:3px; }
.live-events.tiny .event-pill { width:18px; height:17px; font-size:11px; }
.stats { display:grid; grid-template-columns: repeat(4, minmax(100px, 1fr)); gap:8px; }
.stat { background:#0b0b0b; border:1px solid #333; border-radius:5px; padding:8px; }
.stat span { display:block; color:#aaa; font-size:11px; margin-bottom:3px; }
.stat b { font-size:18px; }
@media (max-width: 800px) {
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
</head>
<body>
<header>
  <h1>Eval Event Test</h1>
  <a href="/" style="color:#9ecbff">Back to live UI</a>
</header>
<main>
  <section class="panel">
    <div class="row">
      <button class="good" onclick="addEvents('success', 1)">Add Check</button>
      <button class="danger" onclick="addEvents('failure', 1)">Add X</button>
      <button class="good" onclick="addEvents('success', 10)">Add 10 Checks</button>
      <button class="danger" onclick="addEvents('failure', 10)">Add 10 Xs</button>
      <button onclick="pattern()">Pattern</button>
      <button onclick="fillMany()">Fill 80</button>
      <button onclick="clearEvents()">Clear</button>
    </div>
  </section>
  <section class="panel">
    <div class="live-summary">
      <div class="live-events mono" id="liveEvents"></div>
    </div>
  </section>
  <section class="stats mono" id="stats"></section>
</main>
<script>
let events = [];
let nextId = 1;
function addEvents(type, count) {
  for (let i = 0; i < count; i++) {
    events.push({type, id: nextId++});
  }
  render();
}
function pattern() {
  ['success','success','failure','success','failure','failure','success','success','success','failure'].forEach(type => addEvents(type, 1));
}
function fillMany() {
  clearEvents(false);
  for (let i = 0; i < 80; i++) {
    events.push({type: i % 5 === 0 || i % 7 === 0 ? 'failure' : 'success', id: nextId++});
  }
  render();
}
function clearEvents(renderNow = true) {
  events = [];
  nextId = 1;
  if (renderNow) render();
}
function render() {
  const el = document.getElementById('liveEvents');
  el.classList.toggle('compact', events.length > 12);
  el.classList.toggle('dense', events.length > 28);
  el.classList.toggle('tiny', events.length > 48);
  el.innerHTML = events.map(ev => {
    const ok = ev.type === 'success';
    return `<span class="event-pill ${ok ? 'success' : 'failure'}" title="${ok ? 'success' : 'failure'} #${ev.id}">${ok ? '✓' : '×'}</span>`;
  }).join('');
  const successes = events.filter(ev => ev.type === 'success').length;
  const failures = events.length - successes;
  const rate = events.length ? `${Math.round((successes / events.length) * 100)}%` : '-';
  document.getElementById('stats').innerHTML = [
    `<div class="stat"><span>total</span><b>${events.length}</b></div>`,
    `<div class="stat"><span>success</span><b>${successes}</b></div>`,
    `<div class="stat"><span>fail</span><b>${failures}</b></div>`,
    `<div class="stat"><span>success rate</span><b>${rate}</b></div>`,
  ].join('');
}
fillMany();
</script>
</body>
</html>
"""


EPISODE_VIEWER_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO101 Dataset Editor</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
body { margin:0; background:#101214; color:#f1f3f4; }
header { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 16px; border-bottom:1px solid #2d3338; background:#15181b; }
h1 { margin:0; font-size:18px; }
a { color:#9ecbff; }
main { padding:14px; display:grid; gap:12px; grid-template-columns:minmax(0, 1fr) 260px; align-items:start; }
.editor-main { display:grid; gap:12px; min-width:0; }
.episode-panel { position:sticky; top:12px; max-height:calc(100vh - 88px); overflow:auto; }
.panel { border:1px solid #30363d; border-radius:6px; background:#171b1f; padding:12px; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
label { color:#bbb; font-size:13px; }
select, input, button, textarea { font:inherit; background:#1b2025; color:#f1f3f4; border:1px solid #3a424a; border-radius:5px; padding:7px 9px; }
button { background:#333; color:#fff; border-color:#555; cursor:pointer; padding:8px 10px; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button:disabled { opacity:.45; cursor:not-allowed; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.cams { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:8px; }
.cam { display:grid; gap:5px; min-width:0; }
.cam img { width:100%; max-height:210px; aspect-ratio:4 / 3; object-fit:contain; background:#050607; border:1px solid #30363d; border-radius:5px; }
.cam span { display:none; }
#frameSlider { flex:1 1 360px; min-width:180px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { border-bottom:1px solid #303030; padding:6px; text-align:left; vertical-align:top; }
th { color:#aaa; font-weight:600; }
td input, td select { width:100%; box-sizing:border-box; }
td textarea { width:100%; min-height:38px; box-sizing:border-box; }
tr.selected { background:#13233a; outline:1px solid #2f81f7; }
tr.active:not(.selected) { background:#1d2630; }
tr.other-source { background:#13171b; }
tr.other-source td { color:#c5ced8; }
.compact { width:72px; }
.status { color:#aaa; min-height:20px; }
.status.ok { color:#9ee493; }
.status.warn { color:#ffd166; }
.status.error { color:#ffb4b4; }
.progress-shell { width:min(520px, 100%); height:12px; border:1px solid #30363d; border-radius:5px; background:#0f1215; overflow:hidden; }
.progress-fill { height:100%; width:0%; background:#2f81f7; transition:width .2s ease; }
.progress-fill.ok { background:#2ea043; }
.progress-fill.error { background:#da3633; }
.segment-meta { color:#9aa4af; font-size:12px; margin-top:4px; }
.draft-grid { display:grid; grid-template-columns:minmax(150px, 1.2fr) 92px 92px minmax(180px, 2fr) 110px 120px minmax(140px, 1.5fr) minmax(160px, 1.4fr) auto auto; gap:8px; align-items:end; }
.draft-grid label { display:grid; gap:4px; }
.draft-grid input, .draft-grid select { width:100%; box-sizing:border-box; }
.draft-source { color:#f1f3f4; border:1px solid #30363d; border-radius:5px; padding:8px 9px; min-height:20px; background:#0f1215; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.timeline { position:relative; flex:1 1 360px; min-width:180px; height:18px; border:1px solid #30363d; border-radius:5px; background:#0f1215; overflow:hidden; }
.timeline-empty { color:#9aa4af; font-size:12px; line-height:18px; padding-left:7px; }
.timeline-segment { position:absolute; top:3px; height:10px; min-width:3px; border-radius:3px; background:#2ea043; opacity:.82; cursor:pointer; }
.timeline-segment.selected { background:#9ee493; opacity:1; }
.timeline-segment.active { outline:1px solid #ffd166; }
.timeline-pending { position:absolute; top:3px; height:10px; min-width:3px; border-radius:3px; background:#ffd166; opacity:.9; }
.timeline-cursor { position:absolute; top:0; bottom:0; width:2px; background:#ffdf7e; pointer-events:none; }
.camera-picker { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.camera-picker label { background:#0f1215; border:1px solid #30363d; border-radius:5px; padding:5px 7px; }
.episode-list { display:grid; grid-template-columns:1fr; gap:8px; }
.episode-button { text-align:left; min-height:52px; }
.episode-button.active { border-color:#9ee493; background:#17351f; }
.episode-button span { display:block; color:#9aa4af; font-size:11px; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.episode-mini-timeline { position:relative; height:8px; margin-top:8px; border-radius:3px; background:#0f1215; overflow:hidden; border:1px solid #30363d; }
.episode-mini-segment { position:absolute; top:0; bottom:0; min-width:3px; background:#2ea043; }
.episode-output-list { display:grid; gap:3px; margin-top:7px; color:#d7e1ea; font-size:11px; }
.episode-output-item { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
@media (max-width: 900px) {
  main { grid-template-columns:1fr; }
  .episode-panel { position:static; max-height:none; }
  .draft-grid { grid-template-columns:1fr 1fr; }
  .cams { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<header>
  <h1>SO101 Dataset Editor</h1>
  <a href="/">Back to control UI</a>
</header>
<main>
  <div class="editor-main">
  <section class="panel">
    <div class="row">
      <label>Dataset repo <input id="datasetRepoInput" type="text" placeholder="andlyu/move_blue_ball_20260625_012617" style="min-width:310px"></label>
      <button onclick="loadDatasetRepo()">Load Dataset</button>
      <button class="primary" onclick="downloadDatasetForEditing()">Download for Editing</button>
      <label>Local Dataset <select id="sourceDatasetSelect"></select></label>
      <label style="display:none">Episode <select id="episodeSelect"></select></label>
      <button onclick="loadEpisodes()">Refresh</button>
      <button id="playButton" class="primary" onclick="togglePlay()">Play</button>
      <button onclick="markStart()">Mark Start</button>
      <button onclick="markEnd()">Mark End</button>
      <label>Frame <input id="frameNumber" type="number" min="0" value="0" style="width:84px"></label>
      <input id="frameSlider" type="range" min="0" max="0" value="0">
      <div class="timeline" id="segmentTimeline"></div>
      <div class="status mono" id="datasetStatus"></div>
    </div>
  </section>
  <section class="panel">
    <div class="draft-grid">
      <label>Draft source <div class="draft-source mono" id="draftSource">-</div></label>
      <label>Start <input id="draftStart" type="number" step="0.001" oninput="updatePendingStart(this.value)"></label>
      <label>End <input id="draftEnd" type="number" step="0.001" oninput="updatePendingEnd(this.value)"></label>
      <label>Prompt <input id="draftPrompt" type="text" placeholder="Prompt for this output episode"></label>
      <label>Outcome <select id="draftOutcome"><option value="success">success</option><option value="failure">failure</option></select></label>
      <label>Type <select id="draftType"><option value="teleop">teleop</option><option value="intervention">intervention</option></select></label>
      <label>Notes <input id="draftNotes" type="text" placeholder="Optional"></label>
      <label>Camera Views <div class="camera-picker mono" id="draftCameraPicker"></div></label>
      <button class="primary" onclick="addSegment(event)">Add Output Episode</button>
      <button onclick="clearDraft()">Clear</button>
    </div>
    <div class="status mono" id="manifestStatus" style="margin-top:8px"></div>
  </section>
  <section class="cams" id="cams"></section>
  <section class="panel">
    <div style="overflow:auto; margin-top:10px">
      <table>
        <thead>
          <tr><th>Dataset / Episode</th><th>Start</th><th>End</th><th>Prompt</th><th>Outcome</th><th>Type</th><th>Notes</th><th>Camera_Views</th><th></th></tr>
        </thead>
        <tbody id="segmentsBody"></tbody>
      </table>
    </div>
  </section>
  <section class="panel">
    <div class="row" style="justify-content:space-between">
      <div class="row">
        <label>New Dataset Name <input id="outputDatasetName" type="text" placeholder="move-blue-ball-edited" style="min-width:220px"></label>
        <label>HF Repo <input id="outputRepoId" type="text" placeholder="andlyu/move-blue-ball-edited" style="min-width:240px"></label>
        <label><input id="encodeAfterExport" type="checkbox"> Encode + upload</label>
        <button onclick="loadManifest()">Load Manifest</button>
        <button onclick="saveManifest()">Save Manifest</button>
        <button class="primary" onclick="exportSegments()">Export Episodes</button>
      </div>
      <div class="row">
        <div class="progress-shell"><div class="progress-fill" id="exportProgressFill"></div></div>
        <div class="status mono" id="exportStatus"></div>
      </div>
    </div>
  </section>
  <section class="panel" style="display:none">
    <div class="mono" id="sample"></div>
    <div class="mono" id="stats"></div>
  </section>
  </div>
  <aside class="panel episode-panel">
    <div class="episode-list" id="episodeList"></div>
  </aside>
</main>
<script>
let episodes = [];
let episode = null;
let frameIdx = 0;
let playTimer = null;
let segments = [];
let selectedSegmentIdx = -1;
let dirty = false;
let segmentDrafts = {};
let lastAddSegmentAtMs = 0;
let pendingStartS = null;
let pendingEndS = null;
let outputRepoUserEdited = false;

async function api(path) {
  const res = await fetch(path);
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
}
function fmtVec(values) {
  if (!Array.isArray(values)) return '-';
  return values.map(v => Number(v).toFixed(2)).join(', ');
}
function timestampFor(idx) {
  const sample = (episode?.samples || [])[idx] || {};
  if (sample.timestamp_s !== undefined) return Number(sample.timestamp_s);
  const fps = Math.max(1, Number(episode?.meta?.fps || 30));
  return idx / fps;
}
function queryEpisodeName() {
  return new URLSearchParams(location.search).get('episode') || '';
}
function sourceKey(ep) {
  return ep?.source_repo_id || ep?.dataset_repo_id || (ep?.name || '').replace(/_ep\d+$/, '') || 'local recordings';
}
function defaultPrompt() {
  return document.getElementById('defaultPrompt')?.value.trim() || '';
}
function currentTimeS() {
  return Number(timestampFor(frameIdx).toFixed(3));
}
function segmentLabel(idx) {
  return idx >= 0 ? `episode ${idx + 1}` : 'no output episode selected';
}
function durationLabel(seg) {
  if (!seg) return '-';
  return `${Math.max(0, Number(seg.end_s || 0) - Number(seg.start_s || 0)).toFixed(3)}s`;
}
function activeSegmentIndex() {
  const t = currentTimeS();
  return segments.findIndex(seg => Number(seg.start_s) <= t && t <= Number(seg.end_s));
}
function setStatus(message, cls = '') {
  const el = document.getElementById('manifestStatus');
  if (!el) return;
  el.className = `status mono ${cls}`;
  el.textContent = message;
}
function cloneSegments(value) {
  return (Array.isArray(value) ? value : []).map(seg => ({...seg, cameras: Array.isArray(seg.cameras) ? [...seg.cameras] : seg.cameras}));
}
function currentEpisodeName() {
  return episode?.name || document.getElementById('episodeSelect')?.value || '';
}
function persistDraftFor(name, nextSegments = segments, nextDirty = dirty) {
  if (!name) return;
  segmentDrafts[name] = {segments: cloneSegments(nextSegments), dirty: Boolean(nextDirty)};
  const ep = episodes.find(item => item.name === name);
  if (ep) ep.segments = cloneSegments(nextSegments);
}
function persistCurrentDraft() {
  persistDraftFor(currentEpisodeName());
}
function datasetRepoInput() {
  return document.getElementById('datasetRepoInput')?.value.trim() || '';
}
function setDatasetStatus(message, cls = '') {
  const el = document.getElementById('datasetStatus');
  if (!el) return;
  el.className = `status mono ${cls}`;
  el.textContent = message;
}
async function loadEpisodes() {
  const data = await api('/api/episodes');
  episodes = data.episodes || [];
  await renderSourcePicker();
}
async function renderSourcePicker() {
  const sources = [...new Map(episodes.map(ep => [sourceKey(ep), sourceKey(ep)])).values()];
  const sourceSelect = document.getElementById('sourceDatasetSelect');
  const requestedRepo = datasetRepoInput();
  const previous = sourceSelect.value || new URLSearchParams(location.search).get('source') || requestedRepo || sources[0] || '';
  sourceSelect.innerHTML = sources.map(source => `<option value="${escapeHtml(source)}">${escapeHtml(source)}</option>`).join('');
  if (previous && sources.includes(previous)) sourceSelect.value = previous;
  await renderEpisodePicker();
}
async function loadDatasetRepo() {
  const repo = datasetRepoInput();
  if (!repo) {
    setDatasetStatus('enter a dataset repo id', 'warn');
    return;
  }
  await loadEpisodes();
  const sourceSelect = document.getElementById('sourceDatasetSelect');
  const sources = [...sourceSelect.options].map(option => option.value);
  if (!sources.includes(repo)) {
    setDatasetStatus(`${repo} is not downloaded for editing yet`, 'warn');
    return;
  }
  sourceSelect.value = repo;
  await renderEpisodePicker();
  setDatasetStatus(`loaded local dataset ${repo}`, 'ok');
}
async function downloadDatasetForEditing() {
  const repo = datasetRepoInput();
  if (!repo) {
    setDatasetStatus('enter a dataset repo id', 'warn');
    return;
  }
  setDatasetStatus(`downloading ${repo} for editing...`, 'warn');
  try {
    const data = await postJson('/api/datasets/download', {repo_id: repo});
    setDatasetStatus(`${data.status || 'ready'}: ${repo}`, 'ok');
    await loadEpisodes();
    const sourceSelect = document.getElementById('sourceDatasetSelect');
    if ([...sourceSelect.options].some(option => option.value === repo)) {
      sourceSelect.value = repo;
      await renderEpisodePicker();
    }
  } catch (err) {
    setDatasetStatus(err.message, 'error');
  }
}
async function renderEpisodePicker() {
  const activeSource = document.getElementById('sourceDatasetSelect')?.value || '';
  const visible = visibleEpisodes();
  visible.forEach((ep, idx) => ep.label = `Episode ${idx + 1}`);
  ensureOutputDatasetName();
  const select = document.getElementById('episodeSelect');
  const selected = select.value || queryEpisodeName() || visible[0]?.name || '';
  select.innerHTML = visible.map(ep => `<option value="${escapeHtml(ep.name)}">${escapeHtml(ep.label)}</option>`).join('');
  if (selected && visible.some(ep => ep.name === selected)) select.value = selected;
  if (select.value) await loadEpisode(select.value);
  renderEpisodeList();
}
function episodeDuration(ep) {
  return Number(ep?.duration_s || ep?.meta?.duration_s || ep?.result?.duration_s || 0);
}
function visibleEpisodes() {
  const activeSource = document.getElementById('sourceDatasetSelect')?.value || '';
  return episodes.filter(ep => sourceKey(ep) === activeSource);
}
function cameraNamesForEpisode(ep) {
  const fromMeta = Array.isArray(ep?.cameras) ? ep.cameras : [];
  if (fromMeta.length) return fromMeta.map(cam => cam.name).filter(Boolean);
  if (ep?.frames && typeof ep.frames === 'object') return Object.keys(ep.frames);
  return [];
}
function segmentCameraNames(seg, ep = episode) {
  const available = cameraNamesForEpisode(ep);
  const selected = Array.isArray(seg?.cameras) ? seg.cameras.filter(name => available.includes(name)) : [];
  return selected.length ? selected : available;
}
function episodeSegments(ep) {
  const draft = segmentDrafts[ep.name];
  if (ep.name === episode?.name) return segments;
  return draft?.segments || ep.manifest?.segments || ep.segments || [];
}
function outputRows() {
  return visibleEpisodes().flatMap((ep, epIdx) => {
    const ranges = episodeSegments(ep);
    return ranges.map((seg, segmentIdx) => ({ep, epIdx, seg, segmentIdx}));
  });
}
function shortDatasetLabel(ep) {
  const source = sourceKey(ep);
  return source.includes('/') ? source.split('/').pop() : source;
}
function slugify(value, fallback = '') {
  const slug = String(value || '').trim().replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/-+/g, '-').replace(/^[-_.]+|[-_.]+$/g, '');
  return slug || fallback;
}
function defaultOutputDatasetName() {
  const activeSource = document.getElementById('sourceDatasetSelect')?.value || '';
  return slugify(`${activeSource.split('/').pop() || 'dataset'}-edited`, 'edited-dataset');
}
function defaultOutputRepoId() {
  const activeSource = document.getElementById('sourceDatasetSelect')?.value || '';
  const namespace = activeSource.includes('/') ? activeSource.split('/')[0] : 'andlyu';
  return `${namespace}/${slugify(document.getElementById('outputDatasetName')?.value || defaultOutputDatasetName(), 'edited-dataset')}`;
}
function ensureOutputDatasetName() {
  const el = document.getElementById('outputDatasetName');
  if (el && !el.value.trim()) el.value = defaultOutputDatasetName();
  const repo = document.getElementById('outputRepoId');
  if (repo && !repo.value.trim()) repo.value = defaultOutputRepoId();
}
function setExportProgress(percent, message, cls = '') {
  const fill = document.getElementById('exportProgressFill');
  if (fill) {
    fill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    fill.className = `progress-fill ${cls}`;
  }
  const status = document.getElementById('exportStatus');
  if (status) {
    status.className = `status mono ${cls}`;
    status.textContent = message;
  }
}
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
function setDraftValue(id, value) {
  const el = document.getElementById(id);
  if (!el || document.activeElement === el) return;
  el.value = value;
}
function draftStartValue() {
  if (pendingStartS !== null) return Number(pendingStartS);
  return currentTimeS();
}
function draftEndValue() {
  if (pendingEndS !== null) return Number(pendingEndS);
  return Number((draftStartValue() + 5).toFixed(3));
}
function ensureDraftPrompt() {
  const prompt = document.getElementById('draftPrompt');
  if (prompt && !prompt.value.trim()) {
    prompt.value = episode?.meta?.task || episode?.task || '';
  }
}
function renderDraftExtraction() {
  const source = document.getElementById('draftSource');
  if (source) {
    const epIdx = visibleEpisodes().findIndex(ep => ep.name === episode?.name);
    const epLabel = epIdx >= 0 ? `Episode ${epIdx + 1}` : 'Episode -';
    source.textContent = episode ? `DS: ${shortDatasetLabel(episode)} EP: ${epLabel}` : '-';
    source.title = episode ? sourceKey(episode) : '';
  }
  ensureDraftPrompt();
  renderDraftCameraPicker();
  setDraftValue('draftStart', draftStartValue().toFixed(3));
  setDraftValue('draftEnd', draftEndValue().toFixed(3));
}
function renderDraftCameraPicker() {
  const el = document.getElementById('draftCameraPicker');
  if (!el) return;
  const previous = [...document.querySelectorAll('input[name="draftCamera"]:checked')].map(input => input.value);
  const previousSet = new Set(previous);
  el.innerHTML = cameraList().map(cam => `
    <label title="Include ${escapeHtml(cam.name)} in this output episode">
      <input type="checkbox" name="draftCamera" value="${escapeHtml(cam.name)}" ${!previous.length || previousSet.has(cam.name) ? 'checked' : ''}>
      ${escapeHtml(cam.name)}
    </label>
  `).join('');
}
function selectedDraftCameraNames() {
  const checked = [...document.querySelectorAll('input[name="draftCamera"]:checked')].map(input => input.value);
  return checked.length ? checked : cameraList().map(cam => cam.name);
}
function updatePendingStart(value) {
  const next = Number(value);
  pendingStartS = Number.isFinite(next) ? next : null;
  renderTimeline();
}
function updatePendingEnd(value) {
  const next = Number(value);
  pendingEndS = Number.isFinite(next) ? next : null;
  renderTimeline();
}
function draftSegment() {
  const start = draftStartValue();
  const end = draftEndValue();
  const prompt = document.getElementById('draftPrompt')?.value.trim() || episode?.meta?.task || episode?.task || '';
  const outcome = document.getElementById('draftOutcome')?.value || 'success';
  const type = document.getElementById('draftType')?.value || 'teleop';
  const notes = document.getElementById('draftNotes')?.value || '';
  return {
    start_s: Number(Math.min(start, end).toFixed(3)),
    end_s: Number(Math.max(start, end).toFixed(3)),
    task: prompt,
    outcome,
    type,
    notes,
    cameras: selectedDraftCameraNames(),
  };
}
function clearDraft() {
  pendingStartS = null;
  pendingEndS = null;
  const notes = document.getElementById('draftNotes');
  if (notes) notes.value = '';
  renderDraftExtraction();
  renderTimeline();
  setStatus('cleared draft extraction', '');
}
function miniTimeline(ep) {
  const ranges = episodeSegments(ep);
  const maxEnd = Math.max(0, ...ranges.map(seg => Number(seg.end_s || 0)));
  const total = Math.max(0.001, episodeDuration(ep) || maxEnd || 1);
  const bars = ranges.map(seg => {
    const start = Math.max(0, Math.min(100, Number(seg.start_s || 0) / total * 100));
    const end = Math.max(start, Math.min(100, Number(seg.end_s || 0) / total * 100));
    const width = Math.max(0.5, end - start);
    const title = `${Number(seg.start_s || 0).toFixed(3)}-${Number(seg.end_s || 0).toFixed(3)}s`;
    return `<div class="episode-mini-segment" title="${escapeHtml(title)}" style="left:${start}%;width:${width}%"></div>`;
  }).join('');
  return `<div class="episode-mini-timeline">${bars}</div>`;
}
function episodeOutputSummary(ep) {
  const ranges = episodeSegments(ep);
  if (!ranges.length) return '';
  const items = ranges.slice(0, 3).map((seg, idx) => {
    const label = `Output ${idx + 1}: ${Number(seg.start_s || 0).toFixed(3)}-${Number(seg.end_s || 0).toFixed(3)}s`;
    const prompt = seg.task ? ` ${seg.task}` : '';
    return `<div class="episode-output-item" title="${escapeHtml(label + prompt)}">${escapeHtml(label + prompt)}</div>`;
  }).join('');
  const more = ranges.length > 3 ? `<div class="episode-output-item">+${ranges.length - 3} more</div>` : '';
  return `<div class="episode-output-list">${items}${more}</div>`;
}
function renderEpisodeList() {
  const el = document.getElementById('episodeList');
  if (!el) return;
  const visible = visibleEpisodes();
  const current = document.getElementById('episodeSelect')?.value || '';
  el.innerHTML = visible.map((ep, idx) => {
    const active = ep.name === current ? 'active' : '';
    const label = ep.label || `Episode ${idx + 1}`;
    return `<button class="episode-button ${active}" value="${escapeHtml(ep.name)}" title="${escapeHtml(ep.name)}" onclick="selectEpisode(this.value)">
      ${escapeHtml(label)}
      ${ep.task ? `<span>${escapeHtml(ep.task)}</span>` : ''}
      ${miniTimeline(ep)}
      ${episodeOutputSummary(ep)}
    </button>`;
  }).join('');
}
function selectEpisode(name) {
  const previousName = episode?.name || '';
  persistDraftFor(previousName);
  document.getElementById('episodeSelect').value = name;
  loadEpisode(name);
}
async function loadEpisode(name) {
  const previousName = episode?.name || '';
  if (previousName && previousName !== name) persistDraftFor(previousName);
  episode = await api(`/api/episode?name=${encodeURIComponent(name)}`);
  history.replaceState(null, '', `/episodes?episode=${encodeURIComponent(name)}`);
  const select = document.getElementById('episodeSelect');
  if (select && select.value !== name) select.value = name;
  frameIdx = 0;
  pendingStartS = null;
  pendingEndS = null;
  selectedSegmentIdx = -1;
  const draft = segmentDrafts[name];
  segments = draft ? cloneSegments(draft.segments) : cloneSegments(episode.manifest?.segments);
  dirty = Boolean(draft?.dirty);
  const maxIdx = Math.max(0, Number(episode.length || 0) - 1);
  const slider = document.getElementById('frameSlider');
  const num = document.getElementById('frameNumber');
  slider.max = maxIdx;
  num.max = maxIdx;
  renderCameraPicker();
  renderDraftExtraction();
  renderCameras();
  renderSegments();
  renderFrame(0);
  renderEpisodeList();
  setStatus(`loaded ${episode.name}`, segments.length ? '' : 'warn');
}
function cameraList() {
  const fromMeta = Array.isArray(episode?.cameras) ? episode.cameras : [];
  if (fromMeta.length) return fromMeta.map(cam => ({name: cam.name, frames_dir: cam.frames_dir || cam.name}));
  return Object.keys(episode?.frames || {}).map(name => ({name, frames_dir: name}));
}
function renderCameraPicker() {
  renderDraftCameraPicker();
}
function renderCameras() {
  const cams = document.getElementById('cams');
  cams.innerHTML = cameraList().map(cam => `
    <div class="cam">
      <img id="camera-${escapeHtml(cam.name)}" alt="${escapeHtml(cam.name)} frame">
      <span class="mono" id="camera-${escapeHtml(cam.name)}-info"></span>
    </div>
  `).join('');
}
function frameFor(camera, idx) {
  const frames = episode?.frames?.[camera] || [];
  if (!frames.length) return null;
  return frames[Math.max(0, Math.min(idx, frames.length - 1))];
}
function frameRef(frame, fallbackIdx) {
  if (!frame) return '';
  if (frame.frame !== undefined && frame.frame !== null && String(frame.frame) !== '') return String(frame.frame);
  if (frame.index !== undefined && frame.index !== null && String(frame.index) !== '') return String(frame.index);
  if (frame.path) return String(fallbackIdx);
  return String(fallbackIdx);
}
function renderFrame(idx) {
  if (!episode) return;
  frameIdx = Math.max(0, Math.min(idx, Math.max(0, Number(episode.length || 0) - 1)));
  document.getElementById('frameSlider').value = frameIdx;
  document.getElementById('frameNumber').value = frameIdx;
  renderDraftExtraction();
  for (const cam of cameraList()) {
    const camera = cam.name;
    const frame = frameFor(camera, frameIdx);
    const img = document.getElementById(`camera-${camera}`);
    const info = document.getElementById(`camera-${camera}-info`);
    if (!img || !info) continue;
    if (frame) {
      const ref = frameRef(frame, frameIdx);
      img.src = `/episode_frame?name=${encodeURIComponent(episode.name)}&camera=${encodeURIComponent(camera)}&frame=${encodeURIComponent(ref)}`;
      info.textContent = `${camera} ${ref} t=${frame.timestamp_s ?? frame.timestamp ?? '-'}s`;
    } else {
      img.removeAttribute('src');
      info.textContent = `${camera} no frame`;
    }
  }
  const sample = (episode.samples || [])[frameIdx] || {};
  document.getElementById('sample').innerHTML = [
    '<div>sample</div>', `<div>${frameIdx} / ${Math.max(0, Number(episode.length || 0) - 1)}</div>`,
    '<div>timestamp</div>', `<div>${sample.timestamp_s ?? '-'}</div>`,
    '<div>state</div>', `<div>${escapeHtml(fmtVec(sample.observation_state))}</div>`,
    '<div>action</div>', `<div>${escapeHtml(fmtVec(sample.action))}</div>`,
    '<div>action age</div>', `<div>${sample.action_age_s ?? '-'}</div>`,
    '<div>error</div>', `<div>${escapeHtml(sample.error || '-')}</div>`,
  ].join('');
  renderStats();
  renderSegments();
  renderTimeline();
}
function renderStats() {
  const meta = episode?.meta || {};
  const counts = episode?.counts || {};
  const summary = episode?.eval_summary || {};
  const result = episode?.result || {};
  const cameraCounts = cameraList().map(cam => `${cam.name}:${counts[cam.name] || 0}`).join(' ');
  document.getElementById('stats').innerHTML = [
    `<div class="stat"><span>task</span><b title="${escapeHtml(meta.task || '')}">${escapeHtml(meta.task || '-')}</b></div>`,
    `<div class="stat"><span>kind</span><b>${escapeHtml(episode?.kind || '-')}</b></div>`,
    `<div class="stat"><span>result</span><b>${escapeHtml(result.outcome || summary.state || '-')}</b></div>`,
    `<div class="stat"><span>samples</span><b>${counts.samples || 0}</b></div>`,
    `<div class="stat"><span>cameras</span><b title="${escapeHtml(cameraCounts)}">${escapeHtml(cameraCounts || '-')}</b></div>`,
    `<div class="stat"><span>fps</span><b>${Number(meta.fps || 0).toFixed(1)}</b></div>`,
  ].join('');
}
function renderSegments() {
  const body = document.getElementById('segmentsBody');
  const activeIdx = activeSegmentIndex();
  const rows = outputRows();
  body.innerHTML = rows.map(({ep, epIdx, seg, segmentIdx}) => {
    const isCurrent = ep.name === episode?.name;
    const isSelected = isCurrent && segmentIdx === selectedSegmentIdx;
    const isActive = isCurrent && segmentIdx === activeIdx;
    const sourceLabel = ep.label || `Episode ${epIdx + 1}`;
    const rowClick = isCurrent ? `selectSegment(${segmentIdx})` : `selectOutputEpisode('${escapeJs(ep.name)}', ${segmentIdx})`;
    const readOnly = isCurrent ? '' : 'disabled';
    const sourceSummary = `DS: ${shortDatasetLabel(ep)} EP: ${sourceLabel}`;
    return `
    <tr class="${isSelected ? 'selected' : ''} ${isActive ? 'active' : ''} ${isCurrent ? '' : 'other-source'}" onclick="${rowClick}">
      <td><button title="${escapeHtml(sourceKey(ep))}" onclick="event.stopPropagation(); selectOutputEpisode('${escapeJs(ep.name)}', ${segmentIdx})">${escapeHtml(sourceSummary)}</button></td>
      <td><input class="compact" type="number" step="0.001" value="${Number(seg.start_s || 0)}" ${readOnly} onclick="event.stopPropagation()" oninput="updateSegment(${segmentIdx}, 'start_s', this.value)"></td>
      <td><input class="compact" type="number" step="0.001" value="${Number(seg.end_s || 0)}" ${readOnly} onclick="event.stopPropagation()" oninput="updateSegment(${segmentIdx}, 'end_s', this.value)"></td>
      <td><input value="${escapeHtml(seg.task || '')}" placeholder="Prompt for this output episode" ${readOnly} onclick="event.stopPropagation()" oninput="updateSegment(${segmentIdx}, 'task', this.value)"></td>
      <td><select ${readOnly} onclick="event.stopPropagation()" oninput="updateSegment(${segmentIdx}, 'outcome', this.value)">
        <option value="success" ${seg.outcome === 'success' ? 'selected' : ''}>success</option>
        <option value="failure" ${seg.outcome === 'failure' ? 'selected' : ''}>failure</option>
      </select></td>
      <td><select ${readOnly} onclick="event.stopPropagation()" oninput="updateSegment(${segmentIdx}, 'type', this.value)">
        <option value="teleop" ${seg.type !== 'intervention' ? 'selected' : ''}>teleop</option>
        <option value="intervention" ${seg.type === 'intervention' ? 'selected' : ''}>intervention</option>
      </select></td>
      <td><textarea ${readOnly} onclick="event.stopPropagation()" oninput="updateSegment(${segmentIdx}, 'notes', this.value)">${escapeHtml(seg.notes || '')}</textarea><div class="segment-meta mono">${durationLabel(seg)} ${isActive ? 'active at current frame' : ''}</div></td>
      <td>${renderSegmentCameraControls(ep, segmentIdx, seg, isCurrent)}</td>
      <td><button onclick="event.stopPropagation(); removeOutputEpisode('${escapeJs(ep.name)}', ${segmentIdx})">Remove</button></td>
    </tr>
  `;
  }).join('');
  renderTimeline();
  renderEpisodeList();
}
function renderSegmentCameraControls(ep, idx, seg, isCurrent) {
  const selected = new Set(segmentCameraNames(seg, ep));
  const disabled = isCurrent ? '' : 'disabled';
  return `<div class="camera-picker mono">${cameraNamesForEpisode(ep).map(name => `
    <label title="Include ${escapeHtml(name)} in this output episode">
      <input type="checkbox" ${disabled} ${selected.has(name) ? 'checked' : ''} onclick="event.stopPropagation()" onchange="updateSegmentCamera(${idx}, '${escapeJs(name)}', this.checked)">
      ${escapeHtml(name)}
    </label>
  `).join('')}</div>`;
}
function updateSegmentCamera(idx, camera, checked) {
  if (!segments[idx]) return;
  const available = cameraNamesForEpisode(episode);
  const selected = new Set(segmentCameraNames(segments[idx], episode));
  if (checked) selected.add(camera);
  else selected.delete(camera);
  const next = available.filter(name => selected.has(name));
  segments[idx].cameras = next.length ? next : available;
  dirty = true;
  persistCurrentDraft();
  renderEpisodeList();
  setStatus(`${segmentLabel(idx)} camera views edited; save manifest when ready`, 'warn');
}
function escapeJs(value) {
  return String(value).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}
async function selectOutputEpisode(name, idx) {
  if (episode?.name !== name) {
    await loadEpisode(name);
  }
  selectSegment(idx);
}
function selectSegment(idx) {
  selectedSegmentIdx = idx;
  const seg = segments[idx];
  if (seg) renderFrame(frameIndexForTime(Number(seg.start_s || 0)));
  renderSegments();
  setStatus(`selected ${segmentLabel(idx)}`, dirty ? 'warn' : '');
}
function frameIndexForTime(t) {
  const samples = episode?.samples || [];
  if (!samples.length) return 0;
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < samples.length; i++) {
    const dist = Math.abs(Number(samples[i].timestamp_s ?? samples[i].timestamp ?? 0) - t);
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  return best;
}
function renderTimeline() {
  const el = document.getElementById('segmentTimeline');
  if (!el) return;
  const duration = Math.max(0.001, Number(episode?.meta?.duration_s || episode?.result?.duration_s || timestampFor(Math.max(0, Number(episode?.length || 1) - 1)) || 1));
  const activeIdx = activeSegmentIndex();
  const finalizedSpans = segments.map((seg, idx) => {
    const start = Math.max(0, Math.min(100, Number(seg.start_s || 0) / duration * 100));
    const end = Math.max(start, Math.min(100, Number(seg.end_s || 0) / duration * 100));
    const width = Math.max(0.5, end - start);
    const classes = ['timeline-segment', idx === selectedSegmentIdx ? 'selected' : '', idx === activeIdx ? 'active' : ''].filter(Boolean).join(' ');
    const title = `kept in export: ${segmentLabel(idx)} ${Number(seg.start_s || 0).toFixed(3)}-${Number(seg.end_s || 0).toFixed(3)}s`;
    return `<div class="${classes}" title="${escapeHtml(title)}" style="left:${start}%;width:${width}%" onclick="selectSegment(${idx})"></div>`;
  }).join('');
  let pendingSpan = '';
  if (pendingStartS !== null || pendingEndS !== null) {
    const draft = draftSegment();
    const start = Math.max(0, Math.min(100, Number(draft.start_s || 0) / duration * 100));
    const end = Math.max(start, Math.min(100, Number(draft.end_s || 0) / duration * 100));
    const width = Math.max(0.5, end - start);
    const title = `draft extraction: ${Number(draft.start_s || 0).toFixed(3)}-${Number(draft.end_s || 0).toFixed(3)}s`;
    pendingSpan = `<div class="timeline-pending" title="${escapeHtml(title)}" style="left:${start}%;width:${width}%"></div>`;
  }
  const cursor = Math.max(0, Math.min(100, currentTimeS() / duration * 100));
  el.innerHTML = (finalizedSpans || pendingSpan ? finalizedSpans + pendingSpan : '<div class="timeline-empty">no output episodes marked</div>') + `<div class="timeline-cursor" style="left:${cursor}%"></div>`;
}
function updateSegment(idx, key, value) {
  if (!segments[idx]) return;
  segments[idx][key] = ['start_s', 'end_s'].includes(key) ? Number(value) : value;
  dirty = true;
  persistCurrentDraft();
  renderTimeline();
  renderEpisodeList();
  setStatus(`${segmentLabel(idx)} edited; save manifest when ready`, 'warn');
}
function addSegment(event = null) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
    const now = Date.now();
    if (now - lastAddSegmentAtMs < 500) return selectedSegmentIdx;
    lastAddSegmentAtMs = now;
  }
  const next = draftSegment();
  segments.push(next);
  selectedSegmentIdx = segments.length - 1;
  pendingStartS = null;
  pendingEndS = null;
  const notes = document.getElementById('draftNotes');
  if (notes) notes.value = '';
  dirty = true;
  persistCurrentDraft();
  renderDraftExtraction();
  renderSegments();
  setStatus(`added output ${segmentLabel(selectedSegmentIdx)}`, 'warn');
  return selectedSegmentIdx;
}
function removeSegment(idx) {
  segments.splice(idx, 1);
  selectedSegmentIdx = Math.min(selectedSegmentIdx, segments.length - 1);
  if (!segments.length) selectedSegmentIdx = -1;
  dirty = true;
  persistCurrentDraft();
  renderSegments();
  setStatus(`removed output episode ${idx + 1}; save manifest when ready`, 'warn');
}
function removeOutputEpisode(name, idx) {
  if (episode?.name === name) {
    removeSegment(idx);
    return;
  }
  const ep = episodes.find(item => item.name === name);
  if (!ep) return;
  const next = cloneSegments(episodeSegments(ep));
  next.splice(idx, 1);
  persistDraftFor(name, next, true);
  renderSegments();
  setStatus(`removed output episode ${idx + 1}; save manifest when ready`, 'warn');
}
function editableSegmentIndex() {
  return -1;
}
function markStart() {
  const t = currentTimeS();
  pendingStartS = t;
  if (pendingEndS !== null && pendingEndS <= t) {
    pendingEndS = Number((t + 1).toFixed(3));
  }
  selectedSegmentIdx = -1;
  renderDraftExtraction();
  renderTimeline();
  renderSegments();
  setStatus(`draft start marked at ${t.toFixed(3)}s; press Add Output Episode to keep it`, 'warn');
}
function markEnd() {
  const t = currentTimeS();
  pendingEndS = t;
  if (pendingStartS !== null && pendingEndS <= pendingStartS) {
    pendingStartS = Number(Math.max(0, t - 1).toFixed(3));
  }
  selectedSegmentIdx = -1;
  renderDraftExtraction();
  renderTimeline();
  renderSegments();
  setStatus(`draft end marked at ${t.toFixed(3)}s; press Add Output Episode to keep it`, 'warn');
}
async function postJson(path, payload) {
  const res = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}
async function loadManifest() {
  const data = await api(`/api/segments?source=${encodeURIComponent(episode?.name || 'latest')}`);
  segments = cloneSegments(data.segments);
  selectedSegmentIdx = segments.length ? 0 : -1;
  dirty = false;
  persistCurrentDraft();
  setStatus(`loaded ${segments.length} output episode(s)`, 'ok');
  renderSegments();
}
async function saveManifest() {
  persistCurrentDraft();
  let saved = 0;
  for (const ep of visibleEpisodes()) {
    const sourceSegments = cloneSegments(episodeSegments(ep));
    if (!sourceSegments.length && !segmentDrafts[ep.name]) continue;
    const data = await postJson('/api/segments/save', {source: ep.name, segments: sourceSegments});
    saved += data.segments.length;
    persistDraftFor(ep.name, data.segments, false);
    if (episode?.name === ep.name) {
      segments = cloneSegments(data.segments);
      selectedSegmentIdx = Math.min(selectedSegmentIdx, segments.length - 1);
      if (!segments.length) selectedSegmentIdx = -1;
      dirty = false;
    }
  }
  setStatus(`saved ${saved} output episode(s) across ${visibleEpisodes().length} source episode(s)`, 'ok');
  renderSegments();
}
function repoIdForExport() {
  ensureOutputDatasetName();
  const repo = document.getElementById('outputRepoId')?.value.trim() || defaultOutputRepoId();
  return repo;
}
async function encodeExportedDataset(outputDatasetName, rowCount) {
  const repoId = repoIdForExport();
  if (!repoId.includes('/')) {
    throw new Error('HF repo must look like namespace/dataset');
  }
  setExportProgress(45, `starting encode/upload to ${repoId}...`, 'warn');
  await postJson('/api/dataset/compress', {
    dataset_name: outputDatasetName,
    episodes_root_name: outputDatasetName,
    repo_id: repoId,
    upload: true,
    overwrite: true,
  });
  while (true) {
    await sleep(1000);
    const status = await api('/api/status?log_limit=80');
    const dataset = status.dataset || {};
    const tail = Array.isArray(dataset.output_tail) ? dataset.output_tail : [];
    const encoded = tail.filter(line => String(line).startsWith('saved episode')).length;
    let percent = dataset.state === 'encoding'
      ? 45 + Math.min(40, (encoded / Math.max(1, rowCount)) * 40)
      : dataset.uploaded
        ? 95
        : 50;
    if (dataset.state === 'complete') {
      setExportProgress(100, `encoded and uploaded ${repoId}`, 'ok');
      return dataset;
    }
    if (dataset.state === 'failed') {
      throw new Error(dataset.error || tail.slice(-1)[0] || 'encode/upload failed');
    }
    const label = dataset.state === 'encoding'
      ? `encoding ${encoded}/${rowCount} episode(s)...`
      : dataset.uploaded
        ? `finalizing upload to ${repoId}...`
        : `${dataset.state || 'running'}...`;
    setExportProgress(percent, label, 'warn');
  }
}
async function exportSegments() {
  persistCurrentDraft();
  ensureOutputDatasetName();
  const outputDatasetName = slugify(document.getElementById('outputDatasetName')?.value, '');
  if (!outputDatasetName) {
    setStatus('enter a new dataset name before export', 'warn');
    return;
  }
  let episodeCount = 0;
  let frameCount = 0;
  let outputRoot = '';
  const rows = outputRows();
  if (!rows.length) {
    setExportProgress(0, 'add at least one output episode before export', 'warn');
    return;
  }
  try {
    setExportProgress(1, `exporting 0/${rows.length} output episode(s)...`, 'warn');
    for (let i = 0; i < rows.length; i++) {
      const {ep, seg} = rows[i];
      const sourceSegment = {...seg, cameras: segmentCameraNames(seg, ep)};
      const data = await postJson('/api/busyboard/extract', {source: ep.name, segments: [sourceSegment], cameras: sourceSegment.cameras, output_dataset_name: outputDatasetName});
      episodeCount += Number(data.episode_count || 0);
      frameCount += Number(data.total_frames || 0);
      outputRoot = data.output_root || outputRoot;
      setExportProgress(5 + ((i + 1) / rows.length) * 40, `exported ${i + 1}/${rows.length} output episode(s)...`, 'warn');
    }
    const encode = document.getElementById('encodeAfterExport')?.checked;
    if (encode) {
      await encodeExportedDataset(outputDatasetName, rows.length);
    } else {
      setExportProgress(100, `created ${episodeCount} episode(s), ${frameCount} frames in ${outputRoot || outputDatasetName}`, 'ok');
    }
  } catch (err) {
    setExportProgress(100, err.message || String(err), 'error');
    throw err;
  }
  setStatus(`exported ${episodeCount} output episode(s) to ${outputDatasetName}`, 'ok');
  await loadEpisodes();
}
function stepFrame(delta) {
  renderFrame(frameIdx + delta);
}
function togglePlay() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    document.getElementById('playButton').textContent = 'Play';
    return;
  }
  document.getElementById('playButton').textContent = 'Pause';
  const fps = Math.max(1, Math.min(60, Number(episode?.meta?.fps || 30)));
  playTimer = setInterval(() => {
    if (!episode || frameIdx >= Number(episode.length || 0) - 1) {
      togglePlay();
      return;
    }
    renderFrame(frameIdx + 1);
  }, Math.max(16, Math.round(1000 / fps)));
}
document.getElementById('sourceDatasetSelect').addEventListener('change', () => renderEpisodePicker());
document.getElementById('episodeSelect').addEventListener('change', e => loadEpisode(e.target.value));
document.getElementById('frameSlider').addEventListener('input', e => renderFrame(Number(e.target.value)));
document.getElementById('frameNumber').addEventListener('change', e => renderFrame(Number(e.target.value)));
document.getElementById('outputRepoId')?.addEventListener('input', () => { outputRepoUserEdited = true; });
document.getElementById('outputDatasetName')?.addEventListener('input', () => {
  if (!outputRepoUserEdited) {
    const repo = document.getElementById('outputRepoId');
    if (repo) repo.value = defaultOutputRepoId();
  }
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') stepFrame(-1);
  if (e.key === 'ArrowRight') stepFrame(1);
  if (e.key === ' ') {
    e.preventDefault();
    togglePlay();
  }
});
loadEpisodes().catch(err => {
  document.getElementById('stats').innerHTML = `<div class="stat"><span>error</span><b>${escapeHtml(err.message)}</b></div>`;
});
</script>
</body>
</html>
"""


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
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


def _recording_meta_path(path: Path) -> Path | None:
    if (path / "session_meta.json").exists():
        return path / "session_meta.json"
    if (path / "episode_meta.json").exists():
        return path / "episode_meta.json"
    return None


def _recording_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in (RAW_RECORD_ROOT, RECORD_ROOT):
        if not root.exists():
            continue
        dirs.extend(path for path in root.iterdir() if path.is_dir() and _recording_meta_path(path) is not None)
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def _episode_dirs() -> list[Path]:
    return [path for path in _recording_dirs() if (path / "episode_meta.json").exists()]


def _recordings_for_repo(repo_id: str) -> list[Path]:
    repo_id = repo_id.strip()
    if not repo_id:
        return []
    matches = []
    for path in _recording_dirs():
        meta_path = _recording_meta_path(path)
        meta = _read_json_file(meta_path) if meta_path is not None else {}
        if (meta.get("source_repo_id") or meta.get("dataset_repo_id")) == repo_id:
            matches.append(path)
    return matches


def _download_dataset_for_editing(repo_id: str, overwrite: bool = False) -> dict[str, Any]:
    repo_id = repo_id.strip().strip("/")
    if not repo_id or "/" not in repo_id or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise ValueError("repo_id must look like namespace/dataset")

    existing = _recordings_for_repo(repo_id)
    if existing and not overwrite:
        return {
            "status": "already_imported",
            "repo_id": repo_id,
            "recordings": [_episode_summary(path) for path in existing],
        }

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required to download datasets") from exc

    import import_lerobot_recordings

    dataset_root = DEFAULT_LEROBOT_DATASET_ROOT / repo_id
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_root),
        local_dir_use_symlinks=False,
    )
    summary = import_lerobot_recordings.import_dataset(
        argparse.Namespace(
            dataset_root=str(dataset_root),
            repo_id=repo_id,
            recordings_root=str(RAW_RECORD_ROOT),
            name_prefix="",
            overwrite=overwrite,
        )
    )
    return {
        "status": "downloaded",
        "repo_id": repo_id,
        **summary,
        "recordings": [_episode_summary(path) for path in _recordings_for_repo(repo_id)],
    }


def _safe_episode_dir(name: str) -> Path:
    dirs = _recording_dirs()
    if name == "latest":
        if not dirs:
            raise FileNotFoundError("no recordings")
        return dirs[0]
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("invalid episode name")
    for root in (RAW_RECORD_ROOT, RECORD_ROOT):
        root_resolved = root.resolve()
        path = (root / name).resolve()
        if path != root_resolved and root_resolved not in path.parents:
            continue
        if path.is_dir() and _recording_meta_path(path) is not None:
            return path
    raise FileNotFoundError(f"recording not found: {name}")


def _safe_dataset_name(name: str, fallback: str = "edited-dataset") -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(name or "").strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-_.") or fallback


def _sample_file_for(path: Path, meta: dict[str, Any]) -> str:
    if "sample_file" in meta:
        return str(meta.get("sample_file") or "lerobot_samples.jsonl")
    if (path / "samples.jsonl").exists():
        return "samples.jsonl"
    return "lerobot_samples.jsonl"


def _camera_specs_for(path: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    cameras = meta.get("cameras")
    if isinstance(cameras, list):
        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            name = str(cam.get("name") or f"cam{cam.get('id')}").strip()
            if not name or name == "camNone":
                continue
            frames_dir = str(cam.get("frames_dir") or name)
            frames_file = str(cam.get("frames_file") or f"{frames_dir}/frames.jsonl")
            specs.append({**cam, "name": name, "frames_dir": frames_dir, "frames_file": frames_file})
    if specs:
        return specs
    inferred = []
    for child in sorted(path.iterdir()) if path.exists() else []:
        if child.is_dir() and (child / "frames.jsonl").exists():
            inferred.append(
                {
                    "name": child.name,
                    "frames_dir": child.name,
                    "frames_file": f"{child.name}/frames.jsonl",
                    "lerobot_key": f"observation.images.{child.name}",
                }
            )
    return inferred or [
        {"name": "cam0", "frames_dir": "cam0", "frames_file": "cam0/frames.jsonl"},
        {"name": "cam1", "frames_dir": "cam1", "frames_file": "cam1/frames.jsonl"},
    ]


def _manifest_path(path: Path) -> Path:
    return path / "segment_manifest.json"


def _read_segment_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json_file(_manifest_path(path))
    if not manifest:
        return {"recording_dir": str(path), "segments": []}
    if "recording_dir" not in manifest:
        manifest["recording_dir"] = str(path)
    if not isinstance(manifest.get("segments"), list):
        manifest["segments"] = []
    return manifest


def _write_segment_manifest(path: Path, segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(segments, list):
        raise ValueError("segments must be a list")
    cleaned: list[dict[str, Any]] = []
    for idx, raw in enumerate(segments, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"segment {idx} must be an object")
        start_s = float(raw.get("start_s"))
        end_s = float(raw.get("end_s"))
        if end_s <= start_s:
            raise ValueError(f"segment {idx} end_s must be greater than start_s")
        task = str(raw.get("task") or "").strip()
        if not task:
            raise ValueError(f"segment {idx} task is required")
        outcome = str(raw.get("outcome") or "success").strip().lower()
        if outcome not in {"success", "failure"}:
            raise ValueError(f"segment {idx} outcome must be success or failure")
        segment_type = str(raw.get("type") or "teleop").strip().lower()
        if segment_type not in {"teleop", "intervention"}:
            raise ValueError(f"segment {idx} type must be teleop or intervention")
        cleaned.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "task": task,
                "outcome": outcome,
                "type": segment_type,
                "notes": str(raw.get("notes") or "").strip(),
            }
        )
    manifest = {
        "recording_dir": str(path),
        "segments": cleaned,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _manifest_path(path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _episode_summary(path: Path) -> dict[str, Any]:
    meta_path = _recording_meta_path(path)
    meta = _read_json_file(meta_path) if meta_path is not None else {}
    result = _read_json_file(path / "episode_result.json")
    eval_summary = _read_json_file(path / "eval_summary.json").get("eval", {})
    samples = _read_jsonl_file(path / _sample_file_for(path, meta))
    camera_counts = {}
    cameras = _camera_specs_for(path, meta)
    for cam in cameras:
        camera_counts[str(cam["name"])] = len(_read_jsonl_file(path / str(cam["frames_file"])))
    manifest = _read_segment_manifest(path)
    return {
        "name": path.name,
        "path": str(path),
        "kind": "recording" if meta_path is not None and meta_path.name == "session_meta.json" else "episode",
        "created_at": meta.get("created_at"),
        "source_repo_id": meta.get("source_repo_id") or meta.get("dataset_repo_id"),
        "task": meta.get("task"),
        "duration_s": meta.get("duration_s") or result.get("duration_s"),
        "result": result,
        "counts": {"samples": len(samples), **camera_counts},
        "cameras": cameras,
        "segment_count": len(manifest.get("segments", [])),
        "eval": {
            "state": result.get("outcome") or eval_summary.get("state"),
            "successes": eval_summary.get("successes", 0),
            "failures": eval_summary.get("failures", 0),
        },
    }


def _episode_detail(path: Path) -> dict[str, Any]:
    meta_path = _recording_meta_path(path)
    meta = _read_json_file(meta_path) if meta_path is not None else {}
    result = _read_json_file(path / "episode_result.json")
    eval_summary = _read_json_file(path / "eval_summary.json").get("eval", {})
    samples = _read_jsonl_file(path / _sample_file_for(path, meta))
    cameras = _camera_specs_for(path, meta)
    frames = {str(cam["name"]): _read_jsonl_file(path / str(cam["frames_file"])) for cam in cameras}
    length = min([len(samples), *(len(v) for v in frames.values() if v)] or [0])
    counts = {"samples": len(samples), **{camera: len(rows) for camera, rows in frames.items()}}
    return {
        "name": path.name,
        "path": str(path),
        "kind": "recording" if meta_path is not None and meta_path.name == "session_meta.json" else "episode",
        "meta": meta,
        "result": result,
        "eval_summary": {
            "state": result.get("outcome") or eval_summary.get("state"),
            "successes": eval_summary.get("successes", 0),
            "failures": eval_summary.get("failures", 0),
            "stop_reason": eval_summary.get("stop_reason", ""),
        },
        "counts": counts,
        "cameras": cameras,
        "manifest": _read_segment_manifest(path),
        "length": length,
        "samples": samples[:length],
        "frames": {camera: rows[:length] for camera, rows in frames.items()},
    }


def _episode_frame_response(handler: BaseHTTPRequestHandler, episode_name: str, camera: str, frame: str) -> None:
    try:
        episode_dir = _safe_episode_dir(episode_name)
        meta_path = _recording_meta_path(episode_dir)
        meta = _read_json_file(meta_path) if meta_path is not None else {}
        camera_specs = {str(cam["name"]): cam for cam in _camera_specs_for(episode_dir, meta)}
        if camera not in camera_specs or "/" in frame or "\\" in frame or frame.startswith("."):
            raise ValueError("invalid frame path")
        camera_spec = camera_specs[camera]
        frame_ref = frame
        if frame_ref.isdigit():
            rows = _read_jsonl_file(episode_dir / str(camera_spec["frames_file"]))
            idx = int(frame_ref)
            if idx < 0 or idx >= len(rows):
                raise FileNotFoundError(f"frame index {idx} out of range")
            row = rows[idx]
            frame_ref = str(row.get("path") or row.get("frame") or "")
            if not frame_ref:
                raise FileNotFoundError(f"frame index {idx} has no path")
        frame_path = (episode_dir / frame_ref).resolve()
        if not frame_path.exists() and "/" not in frame_ref:
            frame_path = (episode_dir / str(camera_spec["frames_dir"]) / frame_ref).resolve()
        if episode_dir.resolve() not in frame_path.parents:
            raise ValueError("invalid frame path")
        if frame_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("invalid frame type")
        data = frame_path.read_bytes()
    except FileNotFoundError as exc:
        _json_response(handler, 404, {"error": str(exc)})
        return
    except BaseException as exc:
        _json_response(handler, 400, {"error": str(exc)})
        return
    content_type = "image/png" if frame_path.suffix.lower() == ".png" else "image/jpeg"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "public, max-age=3600")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _jpeg_response(handler: BaseHTTPRequestHandler, jpeg: bytes | None) -> None:
    if jpeg is None:
        _json_response(handler, 404, {"error": "no success overlay yet"})
        return
    handler.send_response(200)
    handler.send_header("Content-Type", "image/jpeg")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(jpeg)))
    handler.end_headers()
    handler.wfile.write(jpeg)


def _success_mjpeg_stream(handler: BaseHTTPRequestHandler, controller: SO101Controller) -> None:
    boundary = b"successframe"
    handler.send_response(200)
    handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    last_jpeg: bytes | None = None
    try:
        while True:
            with controller.success_condition:
                controller.success_condition.wait(timeout=1.0)
                jpeg = controller.success_overlay_jpeg
            if jpeg is None or jpeg == last_jpeg:
                continue
            last_jpeg = jpeg
            handler.wfile.write(b"--" + boundary + b"\r\n")
            handler.wfile.write(b"Content-Type: image/jpeg\r\n")
            handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
            handler.wfile.write(jpeg)
            handler.wfile.write(b"\r\n")
    except (BrokenPipeError, ConnectionResetError):
        return


def _camera_mjpeg_proxy(handler: BaseHTTPRequestHandler, cam: CameraConfig) -> None:
    try:
        with requests.get(cam.url, stream=True, timeout=(3.0, 30.0)) as resp:
            resp.raise_for_status()
            handler.send_response(200)
            handler.send_header(
                "Content-Type",
                resp.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"),
            )
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            for chunk in resp.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                handler.wfile.write(chunk)
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        return
    except Exception as exc:
        try:
            _json_response(handler, 502, {"error": f"camera proxy failed: {exc}"})
        except (BrokenPipeError, ConnectionResetError):
            return


def _camera_jpeg_snapshot(handler: BaseHTTPRequestHandler, cam: CameraConfig) -> None:
    try:
        jpeg = read_mjpeg_jpeg(cam.url, timeout_s=2.0)
        handler.send_response(200)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Cache-Control", "no-store, max-age=0")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Content-Length", str(len(jpeg)))
        handler.end_headers()
        handler.wfile.write(jpeg)
    except (BrokenPipeError, ConnectionResetError):
        return
    except Exception as exc:
        try:
            _json_response(handler, 502, {"error": f"camera snapshot failed: {exc}"})
        except (BrokenPipeError, ConnectionResetError):
            return


def make_handler(controller: SO101Controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/test":
                body = TEST_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/episodes":
                body = EPISODE_VIEWER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/episodes":
                _json_response(self, 200, {"episodes": [_episode_summary(path) for path in _recording_dirs()]})
            elif parsed.path == "/api/episode":
                q = parse_qs(parsed.query)
                name = q.get("name", ["latest"])[0]
                try:
                    _json_response(self, 200, _episode_detail(_safe_episode_dir(name)))
                except FileNotFoundError as exc:
                    _json_response(self, 404, {"error": str(exc)})
                except BaseException as exc:
                    _json_response(self, 400, {"error": str(exc)})
            elif parsed.path == "/api/segments":
                q = parse_qs(parsed.query)
                name = q.get("source", ["latest"])[0]
                try:
                    manifest = _read_segment_manifest(_safe_episode_dir(name))
                    _json_response(self, 200, manifest)
                except FileNotFoundError as exc:
                    _json_response(self, 404, {"error": str(exc)})
                except BaseException as exc:
                    _json_response(self, 400, {"error": str(exc)})
            elif parsed.path == "/episode_frame":
                q = parse_qs(parsed.query)
                _episode_frame_response(
                    self,
                    q.get("name", ["latest"])[0],
                    q.get("camera", [""])[0],
                    q.get("frame", [""])[0],
                )
            elif parsed.path == "/api/status":
                q = parse_qs(parsed.query)
                log_limit = int(q.get("log_limit", [str(DEFAULT_STATUS_LOG_LIMIT)])[0])
                _json_response(self, 200, controller.status(log_limit=log_limit))
            elif parsed.path == "/api/health":
                try:
                    _json_response(self, 200, {"policy": controller.health()})
                except BaseException as exc:
                    _json_response(self, 500, {"error": str(exc)})
            elif parsed.path == "/api/success.jpg":
                _jpeg_response(self, controller.success_overlay_jpeg)
            elif parsed.path == "/api/success.mjpg":
                _success_mjpeg_stream(self, controller)
            elif parsed.path.startswith("/camera/") and parsed.path.endswith(".mjpg"):
                name = parsed.path.removeprefix("/camera/").removesuffix(".mjpg")
                try:
                    _camera_mjpeg_proxy(self, controller._camera_from_spec(name))
                except ValueError as exc:
                    _json_response(self, 404, {"error": str(exc)})
            elif parsed.path.startswith("/camera/") and parsed.path.endswith(".jpg"):
                name = parsed.path.removeprefix("/camera/").removesuffix(".jpg")
                try:
                    _camera_jpeg_snapshot(self, controller._camera_from_spec(name))
                except ValueError as exc:
                    _json_response(self, 404, {"error": str(exc)})
            elif parsed.path.startswith("/cam") and parsed.path.endswith(".mjpg"):
                name = parsed.path.strip("/").removesuffix(".mjpg")
                try:
                    _camera_mjpeg_proxy(self, controller._camera_from_spec(name))
                except ValueError as exc:
                    _json_response(self, 404, {"error": str(exc)})
            elif parsed.path.startswith("/cam") and parsed.path.endswith(".jpg"):
                name = parsed.path.strip("/").removesuffix(".jpg")
                try:
                    _camera_jpeg_snapshot(self, controller._camera_from_spec(name))
                except ValueError as exc:
                    _json_response(self, 404, {"error": str(exc)})
            else:
                _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                _json_response(self, 400, {"error": "invalid json"})
                return
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/eval/start":
                    controller.start_eval(
                        instruction=str(data.get("instruction", DEFAULT_INSTRUCTION)),
                        run_duration_s=float(data.get("run_duration_s", DEFAULT_EVAL_RUN_DURATION_S)),
                        attempt_duration_s=float(data.get("attempt_duration_s", DEFAULT_EVAL_ATTEMPT_DURATION_S)),
                        max_consecutive_failures=int(
                            data.get("max_consecutive_failures", DEFAULT_EVAL_MAX_CONSECUTIVE_FAILURES)
                        ),
                        record_fps=float(data.get("record_fps", DEFAULT_EVAL_RECORD_FPS)),
                        exec_steps=int(data.get("exec_steps", DEFAULT_EXEC_STEPS)),
                        max_step_deg=float(data.get("max_step_deg", DEFAULT_MAX_STEP_DEG)),
                        hz=float(data.get("hz", DEFAULT_HZ)),
                        allow_teleop=bool(data.get("allow_teleop", True)),
                        record_episodes=bool(data.get("record_episodes", True)),
                    )
                    _json_response(self, 200, {"ok": True, "eval": controller.status().get("eval")})
                elif parsed.path == "/api/eval/resume":
                    controller.resume_eval()
                    _json_response(self, 200, {"ok": True, "eval": controller.status().get("eval")})
                elif parsed.path == "/api/eval/stop":
                    controller.stop_eval()
                    _json_response(self, 200, {"ok": True, "eval": controller.status().get("eval")})
                elif parsed.path == "/api/eval/clear":
                    status = controller.clear_eval()
                    _json_response(self, 200, {"ok": True, "eval": status})
                elif parsed.path == "/api/teleop/claim":
                    status = controller.claim_teleop(operator=str(data.get("operator", "operator")))
                    _json_response(self, 200, {"ok": True, "teleop": status})
                elif parsed.path == "/api/teleop/heartbeat":
                    status = controller.heartbeat_teleop(lease_id=str(data.get("lease_id", "")))
                    _json_response(self, 200, {"ok": True, "teleop": status})
                elif parsed.path == "/api/teleop/release":
                    status = controller.release_teleop(
                        lease_id=str(data.get("lease_id", "")),
                        outcome=str(data.get("outcome", "complete")),
                    )
                    _json_response(self, 200, {"ok": True, "teleop": status})
                elif parsed.path == "/api/hf_teleop/start":
                    status = controller.start_hf_teleop(
                        duration_s=float(data.get("duration_s", 10.0)),
                        fps=int(data.get("fps", 30)),
                    )
                    _json_response(self, 200, {"ok": True, "hf_teleop": status})
                elif parsed.path == "/api/hf_teleop/stop":
                    status = controller.stop_hf_teleop()
                    _json_response(self, 200, {"ok": True, "hf_teleop": status})
                elif parsed.path == "/api/start":
                    controller.start_policy(
                        instruction=str(data.get("instruction", DEFAULT_INSTRUCTION)),
                        duration_s=float(data.get("duration_s", DEFAULT_DURATION_S)),
                        exec_steps=int(data.get("exec_steps", DEFAULT_EXEC_STEPS)),
                        max_step_deg=float(data.get("max_step_deg", DEFAULT_MAX_STEP_DEG)),
                        hz=float(data.get("hz", DEFAULT_HZ)),
                    )
                    _json_response(self, 200, {"ok": True})
                elif parsed.path == "/api/stop":
                    if controller._eval_running():
                        controller.stop_eval()
                    else:
                        controller.stop_policy()
                    _json_response(self, 200, {"ok": True})
                elif parsed.path == "/api/home":
                    controller.start_home(
                        max_step_deg=float(data.get("max_step_deg", HOME_STEP_DEG)),
                        hz=float(data.get("hz", DEFAULT_HZ)),
                    )
                    _json_response(self, 200, {"ok": True})
                elif parsed.path == "/api/save_home":
                    home = controller.save_current_as_home()
                    _json_response(self, 200, {"ok": True, "home": home})
                elif parsed.path == "/api/connect":
                    joints = controller.read_station_joints()
                    _json_response(self, 200, {"ok": True, **joints})
                elif parsed.path == "/api/disconnect":
                    controller.disconnect()
                    _json_response(self, 200, {"ok": True})
                elif parsed.path == "/api/nudge":
                    state = controller.nudge(
                        str(data["joint"]),
                        float(data["delta"]),
                        lease_id=str(data.get("lease_id", "")),
                    )
                    _json_response(self, 200, {"ok": True, "state": state})
                elif parsed.path == "/api/gripper":
                    state = controller.set_gripper(
                        float(data["value"]),
                        lease_id=str(data.get("lease_id", "")),
                    )
                    _json_response(self, 200, {"ok": True, "state": state})
                elif parsed.path == "/api/record/start":
                    result = controller.start_recording(
                        duration_s=float(data.get("duration_s", DEFAULT_RECORD_DURATION_S)),
                        fps=float(data.get("fps", DEFAULT_RECORD_FPS)),
                        cameras=list(data.get("cameras", [])),
                        task=str(data.get("task", DEFAULT_INSTRUCTION)),
                        name_prefix=str(data.get("name_prefix", "so101_recording")),
                        extra_meta=data.get("extra_meta") if isinstance(data.get("extra_meta"), dict) else None,
                        capture_mode=str(data.get("capture_mode", "policy_execute")),
                    )
                    _json_response(self, 200, {"ok": True, **result})
                elif parsed.path == "/api/record/stop":
                    controller.stop_recording()
                    _json_response(self, 200, {"ok": True})
                elif parsed.path == "/api/segments/save":
                    source_dir = _safe_episode_dir(str(data.get("source", "latest")))
                    manifest = _write_segment_manifest(source_dir, data.get("segments", []))
                    _json_response(self, 200, {"ok": True, **manifest})
                elif parsed.path == "/api/busyboard/extract":
                    summary = controller.extract_busyboard_segments(
                        source_name=str(data.get("source", "latest")),
                        segments=data.get("segments", []),
                        cameras=[str(camera) for camera in data.get("cameras", [])],
                        output_dataset_name=str(data.get("output_dataset_name", "")),
                    )
                    _json_response(self, 200, {"ok": True, **summary})
                elif parsed.path == "/api/datasets/download":
                    summary = _download_dataset_for_editing(
                        repo_id=str(data.get("repo_id", "")),
                        overwrite=bool(data.get("overwrite", False)),
                    )
                    _json_response(self, 200, {"ok": True, **summary})
                elif parsed.path == "/api/dataset/compress":
                    status = controller.start_dataset_compression(
                        repo_id=str(data.get("repo_id", "")),
                        dataset_name=str(data.get("dataset_name", "")),
                        root=str(data.get("root", "")),
                        episodes_root_name=str(data.get("episodes_root_name", "")),
                        upload=bool(data.get("upload", False)),
                        private=bool(data.get("private", False)),
                        include_failures=bool(data.get("include_failures", False)),
                        overwrite=bool(data.get("overwrite", False)),
                    )
                    _json_response(self, 200, {"ok": True, "dataset": status})
                elif parsed.path == "/api/sam3/preview":
                    if controller.is_policy_running() or controller._eval_running():
                        _json_response(self, 409, {"error": "SAM3 preview disabled while policy/eval is running"})
                        return
                    result = controller.preview_success_sam3(
                        prompt=str(data.get("prompt", SUCCESS_CONTAINER_SAM3_PROMPT)),
                        min_score=float(data.get("min_score", SUCCESS_CONTAINER_SAM3_MIN_SCORE)),
                        camera=data.get("camera", "front"),
                    )
                    _json_response(self, 200, result)
                elif parsed.path == "/api/success/sam3":
                    status = controller.configure_success_sam3(
                        prompt=str(data.get("prompt", SUCCESS_CONTAINER_SAM3_PROMPT)),
                        min_score=float(data.get("min_score", SUCCESS_CONTAINER_SAM3_MIN_SCORE)),
                    )
                    _json_response(self, 200, {"ok": True, "success_tracking": status})
                elif parsed.path == "/api/success/reset":
                    status = controller.reset_success_tracking()
                    _json_response(self, 200, {"ok": True, "success_tracking": status})
                else:
                    _json_response(self, 404, {"error": "not found"})
            except BaseException as exc:
                controller.set_error(exc)
                _json_response(self, 500, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="blupe_follower")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-id", default="blupe_leader")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8202", help="Local policy runner base URL.")
    parser.add_argument("--molmo-url", default=None, help="Deprecated alias for --policy-url.")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Semantic camera mapping NAME=URL. Repeat for front, side, wrist. Defaults to local cam0/cam1/cam2 MJPEG.",
    )
    parser.add_argument(
        "--policy-camera",
        action="append",
        default=[],
        choices=SEMANTIC_CAMERA_NAMES,
        help=(
            "Camera name sent to the policy runner, in order. Repeat to match the checkpoint's trained "
            "image keys. Defaults to SO101_POLICY_CAMERAS or front,side."
        ),
    )
    parser.add_argument("--success-fps", type=float, default=DEFAULT_SUCCESS_FPS)
    parser.add_argument("--no-success-tracking", action="store_true")
    args = parser.parse_args()
    try:
        camera_configs = parse_camera_config_specs(args.camera, SEMANTIC_CAMERA_NAMES)
    except ValueError as exc:
        parser.error(str(exc))

    controller = SO101Controller(
        args.robot_port,
        args.robot_id,
        args.molmo_url or args.policy_url,
        leader_port=args.leader_port,
        leader_id=args.leader_id,
        camera_configs=camera_configs,
        policy_camera_names=args.policy_camera or DEFAULT_POLICY_CAMERA_NAMES,
        success_enabled=not args.no_success_tracking,
        success_fps=args.success_fps,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(controller))
    print(f"SO101 web UI listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
