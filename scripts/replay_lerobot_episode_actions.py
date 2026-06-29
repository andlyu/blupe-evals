#!/usr/bin/env python3
"""Replay one LeRobot episode's action trajectory on an SO101 arm.

This reads only LeRobot parquet action/state rows. It does not load camera
videos or images. Hardware motion is disabled unless --execute is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blupe_evals.station.joint_conventions import (
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
    policy_action_to_robot_target,
)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DATA_COLUMNS = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]


def _fmt(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(v):.2f}" for v in values) + "]"


def _resolve_dataset_path(repo_id_or_path: str, *, revision: str | None) -> Path:
    path = Path(repo_id_or_path).expanduser()
    if path.exists():
        return path

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Install huggingface_hub or pass a local dataset snapshot path.") from exc

    return Path(
        snapshot_download(
            repo_id=repo_id_or_path,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[
                "data/**/*.parquet",
                "meta/episodes/**/*.parquet",
                "meta/tasks.parquet",
                "README.md",
            ],
        )
    )


def _load_episode_rows(dataset_root: Path, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("Install pyarrow to read LeRobot parquet action rows.") from exc

    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no LeRobot data parquet files under {dataset_root / 'data'}")

    states: list[list[float]] = []
    actions: list[list[float]] = []
    timestamps: list[float] = []
    frame_indices: list[int] = []
    for path in data_files:
        table = pq.read_table(path, columns=DATA_COLUMNS)
        rows = table.to_pydict()
        for idx, ep_value in enumerate(rows["episode_index"]):
            if int(ep_value) != episode:
                continue
            states.append(rows["observation.state"][idx])
            actions.append(rows["action"][idx])
            timestamps.append(float(rows["timestamp"][idx]))
            frame_indices.append(int(rows["frame_index"][idx]))

    if not actions:
        raise ValueError(f"episode {episode} has no rows in {dataset_root}")

    order = np.argsort(np.asarray(frame_indices, dtype=np.int64))
    return (
        np.asarray(states, dtype=np.float32)[order],
        np.asarray(actions, dtype=np.float32)[order],
        np.asarray(timestamps, dtype=np.float32)[order],
        np.asarray(frame_indices, dtype=np.int64)[order],
    )


def _policy_to_robot(actions: np.ndarray, *, convention: str) -> np.ndarray:
    if convention == "robot":
        return np.asarray(actions, dtype=np.float32)
    if convention != "policy":
        raise ValueError(f"unsupported action convention {convention!r}")
    return np.stack(
        [
            policy_action_to_robot_target(
                action,
                policy_to_robot_signs=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
                policy_to_robot_offsets_deg=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
            )
            for action in np.asarray(actions, dtype=np.float32)
        ]
    )


def _episode_fps(timestamps: np.ndarray, fallback: float) -> float:
    if len(timestamps) < 2:
        return fallback
    deltas = np.diff(timestamps.astype(np.float64))
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(deltas) == 0:
        return fallback
    return float(1.0 / np.median(deltas))


def _connect_robot(port: str, robot_id: str):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        max_relative_target=None,
        disable_torque_on_disconnect=False,
    )
    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)
    return robot


def _read_state(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.asarray([obs[f"{joint}.pos"] for joint in JOINTS], dtype=np.float32)


def _send_state(robot, target: np.ndarray) -> None:
    robot.send_action({f"{joint}.pos": float(value) for joint, value in zip(JOINTS, target)})


def _step_toward(
    robot,
    target: np.ndarray,
    *,
    max_step_deg: float,
    hz: float,
    tolerance_deg: float,
    timeout_s: float,
) -> np.ndarray:
    period = 1.0 / hz if hz > 0 else 0.0
    start = time.monotonic()
    while True:
        measured = _read_state(robot)
        err = target - measured
        if float(np.max(np.abs(err))) <= tolerance_deg:
            _send_state(robot, target)
            return _read_state(robot)
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"preposition timed out; measured={_fmt(measured)} target={_fmt(target)}")
        step = np.clip(err, -max_step_deg, max_step_deg) if max_step_deg > 0 else err
        _send_state(robot, measured + step)
        if period > 0:
            time.sleep(period)


def _replay(robot, targets: np.ndarray, *, hz: float, max_step_deg: float, log_every: int) -> None:
    period = 1.0 / hz if hz > 0 else 0.0
    cur_cmd = _read_state(robot)
    for idx, model_target in enumerate(targets):
        if max_step_deg > 0:
            target = cur_cmd + np.clip(model_target - cur_cmd, -max_step_deg, max_step_deg)
        else:
            target = model_target
        _send_state(robot, target)
        if idx % max(1, log_every) == 0 or idx == len(targets) - 1:
            measured = _read_state(robot)
            print(f"frame={idx:04d} target={_fmt(target)} measured={_fmt(measured)}", flush=True)
        cur_cmd = target
        if period > 0:
            time.sleep(period)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="andlyu/move_blue_ball_training_v21")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 replays all remaining frames.")
    parser.add_argument("--hz", type=float, default=0.0, help="0 uses median timestamp FPS.")
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--action-convention", choices=("policy", "robot"), default="policy")
    parser.add_argument("--max-step-deg", type=float, default=2.0, help="0 disables per-frame clipping.")
    parser.add_argument("--preposition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preposition-step-deg", type=float, default=2.0)
    parser.add_argument("--preposition-timeout-s", type=float, default=45.0)
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="blupe_follower")
    parser.add_argument("--log-every", type=int, default=15)
    parser.add_argument("--execute", action="store_true", help="Actually command the robot.")
    args = parser.parse_args()

    dataset_root = _resolve_dataset_path(args.dataset, revision=args.revision)
    states, actions, timestamps, frame_indices = _load_episode_rows(dataset_root, args.episode)
    robot_targets = _policy_to_robot(actions, convention=args.action_convention)

    start = max(0, int(args.start_frame))
    stop = len(robot_targets) if args.max_frames <= 0 else min(len(robot_targets), start + int(args.max_frames))
    if start >= stop:
        raise ValueError(f"empty replay slice start={start} stop={stop} len={len(robot_targets)}")

    states_robot = _policy_to_robot(states, convention=args.action_convention)
    selected_targets = robot_targets[start:stop]
    fps = float(args.hz) if args.hz > 0 else _episode_fps(timestamps, args.fallback_fps)

    print(f"dataset_root={dataset_root}")
    print(f"episode={args.episode} frames={len(robot_targets)} selected={start}:{stop} fps={fps:.3f}")
    print(f"action_convention={args.action_convention} max_step_deg={args.max_step_deg}")
    print(f"frame_indices selected={int(frame_indices[start])}..{int(frame_indices[stop - 1])}")
    print(f"first_state_robot={_fmt(states_robot[start])}")
    print(f"first_target_robot={_fmt(selected_targets[0])}")
    print(f"last_target_robot={_fmt(selected_targets[-1])}")
    print(f"target_min={_fmt(selected_targets.min(axis=0))}")
    print(f"target_max={_fmt(selected_targets.max(axis=0))}")
    print(f"max_abs_frame_delta={_fmt(np.max(np.abs(np.diff(selected_targets, axis=0)), axis=0)) if len(selected_targets) > 1 else _fmt(np.zeros(len(JOINTS)))}")

    if not args.execute:
        print("dry-run only; pass --execute to command the SO101 arm")
        return 0

    robot = _connect_robot(args.robot_port, args.robot_id)
    try:
        measured = _read_state(robot)
        print(f"connected measured={_fmt(measured)}", flush=True)
        if args.preposition:
            print(f"preposition target={_fmt(selected_targets[0])}", flush=True)
            measured = _step_toward(
                robot,
                selected_targets[0],
                max_step_deg=float(args.preposition_step_deg),
                hz=fps,
                tolerance_deg=float(args.tolerance_deg),
                timeout_s=float(args.preposition_timeout_s),
            )
            print(f"preposition reached measured={_fmt(measured)}", flush=True)
        print("replay start", flush=True)
        _replay(
            robot,
            selected_targets,
            hz=fps,
            max_step_deg=float(args.max_step_deg),
            log_every=int(args.log_every),
        )
        print(f"replay complete measured={_fmt(_read_state(robot))}", flush=True)
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
