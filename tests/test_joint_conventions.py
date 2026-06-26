import numpy as np
import pytest

from blupe_evals.station.joint_conventions import (
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
    load_joint_array_env,
    policy_action_to_robot_target,
    robot_state_to_policy_state,
    validate_policy_to_robot_signs,
)


def test_so101_defaults_convert_jetson_home_to_molmoact2_policy_convention():
    robot_home = np.array([0, -90, 70, 0, -45, 0], dtype=np.float32)

    policy_home = robot_state_to_policy_state(
        robot_home,
        policy_to_robot_signs=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
        policy_to_robot_offsets_deg=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    )

    np.testing.assert_allclose(policy_home, [0, 180, 160, 0, -45, 0])


def test_so101_defaults_convert_policy_action_back_to_robot_convention():
    policy_action = np.array([-0.52734375, 189.140625, 181.40625, 60.64453125, -3.603515625, 1.0971787])

    robot_target = policy_action_to_robot_target(
        policy_action,
        policy_to_robot_signs=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
        policy_to_robot_offsets_deg=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    )

    np.testing.assert_allclose(
        robot_target,
        [-0.52734375, -99.140625, 91.40625, 60.64453125, -3.603515625, 1.0971787],
    )


def test_so101_policy_robot_conversion_round_trips():
    robot_state = np.array([12.5, -80.0, 65.5, 4.0, -47.0, 13.0], dtype=np.float32)

    policy_state = robot_state_to_policy_state(
        robot_state,
        policy_to_robot_signs=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
        policy_to_robot_offsets_deg=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    )
    round_trip = policy_action_to_robot_target(
        policy_state,
        policy_to_robot_signs=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_SIGNS,
        policy_to_robot_offsets_deg=DEFAULT_SO101_POLICY_TO_ROBOT_JOINT_OFFSETS_DEG,
    )

    np.testing.assert_allclose(round_trip, robot_state)


def test_load_joint_array_env_validates_shape(monkeypatch):
    monkeypatch.setenv("TEST_JOINT_ARRAY", "[1,2]")

    with pytest.raises(ValueError, match="TEST_JOINT_ARRAY must have 6 values"):
        load_joint_array_env("TEST_JOINT_ARRAY", np.zeros(6, dtype=np.float32), joint_count=6)


def test_validate_policy_to_robot_signs_rejects_zero():
    with pytest.raises(ValueError, match="cannot contain zero"):
        validate_policy_to_robot_signs(np.array([1, 0, 1], dtype=np.float32))
