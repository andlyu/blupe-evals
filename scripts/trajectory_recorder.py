from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n")


class TrajectoryRecorder:
    """Raw state/action spool for one policy rollout.

    This intentionally writes a simple pending trajectory first. A separate success gate
    later decides whether to append it to a LeRobotDataset.
    """

    def __init__(
        self,
        trajectory_dir: Path,
        *,
        run_id: str,
        policy: str,
        task: str,
        joint_names: list[str],
        units: str = "radians",
    ):
        self.trajectory_dir = trajectory_dir
        self.run_id = run_id
        self.policy = policy
        self.task = task
        self.joint_names = joint_names
        self.units = units
        self.steps_path = trajectory_dir / "steps.jsonl"
        self.meta_path = trajectory_dir / "episode.json"
        self.started_mono = time.monotonic()
        self.rollout_s = 0.0
        self.step_idx = 0
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta(status="recording")

    @classmethod
    def from_env(cls, *, policy: str, joint_names: list[str]) -> "TrajectoryRecorder | None":
        enabled = os.environ.get("BLUPE_COLLECT_TRAJECTORY", "").lower() in {"1", "true", "yes", "on"}
        trajectory_dir = os.environ.get("BLUPE_TRAJECTORY_DIR", "")
        if not enabled and not trajectory_dir:
            return None
        run_id = os.environ.get("BLUPE_TRAJECTORY_RUN_ID") or time.strftime("run_%Y%m%d_%H%M%S")
        root = Path(os.environ.get("BLUPE_TRAJECTORY_ROOT", "trajectories/pending"))
        out_dir = Path(trajectory_dir) if trajectory_dir else root / run_id
        return cls(
            out_dir,
            run_id=run_id,
            policy=os.environ.get("BLUPE_TRAJECTORY_POLICY") or policy,
            task=os.environ.get("BLUPE_TRAJECTORY_TASK") or "",
            joint_names=joint_names,
            units=os.environ.get("BLUPE_TRAJECTORY_UNITS") or "radians",
        )

    def _write_meta(self, *, status: str, error: str = "") -> None:
        payload = {
            "format": "blupe_policy_trajectory",
            "format_version": 1,
            "run_id": self.run_id,
            "policy": self.policy,
            "task": self.task,
            "status": status,
            "error": error,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "duration_s": round(time.monotonic() - self.started_mono, 6),
            "rollout_duration_s": round(self.rollout_s, 6),
            "steps": self.step_idx,
            "joint_names": self.joint_names,
            "units": self.units,
            "steps_file": self.steps_path.name,
        }
        self.meta_path.write_text(json.dumps(payload, indent=2) + "\n")

    def record_step(
        self,
        *,
        observation_state: Any,
        action: Any,
        gripper: float | None,
        desired_action: Any | None = None,
        observation_gripper: float | None = None,
        dt_s: float | None = None,
    ) -> None:
        now = time.monotonic()
        rollout_dt_s = max(0.0, float(dt_s)) if dt_s is not None else 0.0
        payload = {
            "step": self.step_idx,
            "timestamp_s": round(self.rollout_s, 6),
            "rollout_dt_s": round(rollout_dt_s, 6),
            "wall_elapsed_s": round(now - self.started_mono, 6),
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "observation.state": observation_state,
            "observation.gripper": observation_gripper,
            "action": action,
            "action.desired": desired_action,
            "gripper": gripper,
            "dt_s": dt_s,
        }
        _append_jsonl(self.steps_path, payload)
        self.step_idx += 1
        self.rollout_s += rollout_dt_s

    def finalize(self, *, status: str = "complete", error: str = "") -> None:
        self._write_meta(status=status, error=error)
