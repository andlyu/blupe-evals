"""Joint convention transforms for SO100/SO101 policies and calibrated robots."""

from __future__ import annotations

import json
import os

import numpy as np


# MolmoAct2-SO100_101 was trained with the LeRobot pre-PR #777 convention.
# LeRobot 0.5.x SO100/SO101 calibrations use the newer convention.
# robot_v3 = policy_v2 * sign + offset
DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG = np.array([0, 90, -90, 0, 0, 0], dtype=np.float32)


def load_joint_array_env(env_name: str, default: np.ndarray, *, joint_count: int) -> np.ndarray:
    raw = os.environ.get(env_name)
    values = default if raw is None else np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape != (joint_count,):
        raise ValueError(f"{env_name} must have {joint_count} values")
    return values


def validate_policy_to_robot_signs(signs: np.ndarray, *, env_name: str = "policy_to_robot_signs") -> None:
    if np.any(np.asarray(signs, dtype=np.float32) == 0):
        raise ValueError(f"{env_name} cannot contain zero values")


def robot_state_to_policy_state(
    state: np.ndarray,
    *,
    policy_to_robot_signs: np.ndarray,
    policy_to_robot_offsets_deg: np.ndarray,
) -> np.ndarray:
    """Convert calibrated robot joint degrees into the policy training convention."""

    return (np.asarray(state, dtype=np.float32) - policy_to_robot_offsets_deg) / policy_to_robot_signs


def policy_action_to_robot_target(
    action: np.ndarray,
    *,
    policy_to_robot_signs: np.ndarray,
    policy_to_robot_offsets_deg: np.ndarray,
) -> np.ndarray:
    """Convert policy action degrees into calibrated robot target degrees."""

    return np.asarray(action, dtype=np.float32) * policy_to_robot_signs + policy_to_robot_offsets_deg
