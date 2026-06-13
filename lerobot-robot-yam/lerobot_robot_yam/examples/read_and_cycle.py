"""Read YAM positions through LeRobot, then cycle the gripper.

This script talks only to ``YamFollower``. Start ``yam_serve`` first.
"""

from __future__ import annotations

import argparse
import time

from lerobot_robot_yam import YamFollower, YamFollowerConfig


def _format_obs(obs: dict, names: list[str]) -> str:
    return " ".join(f"{name}={obs[f'{name}.pos']:.4f}" for name in names)


def _read_positions(robot: YamFollower, names: list[str], seconds: float, fps: float) -> None:
    period = 1.0 / fps
    start = time.monotonic()
    next_tick = start
    print(f"[read] printing positions for {seconds:.1f}s at {fps:.1f}Hz", flush=True)

    while True:
        now = time.monotonic()
        if now - start >= seconds:
            break
        if now < next_tick:
            time.sleep(next_tick - now)
            continue

        obs = robot.get_observation()
        print(f"[read] t={now - start:5.2f}s {_format_obs(obs, names)}", flush=True)
        next_tick += period


def _current_arm_action(obs: dict, joints: list[str]) -> dict[str, float]:
    return {f"{joint}.pos": float(obs[f"{joint}.pos"]) for joint in joints}


def _drive_gripper_phase(
    robot: YamFollower,
    config: YamFollowerConfig,
    names: list[str],
    target: float,
    label: str,
    duration: float,
    fps: float,
) -> None:
    period = 1.0 / fps
    start = time.monotonic()
    next_tick = start
    next_print = start

    while True:
        now = time.monotonic()
        if now - start >= duration:
            break
        if now < next_tick:
            time.sleep(next_tick - now)
            continue

        obs = robot.get_observation()
        action = _current_arm_action(obs, list(config.joints))
        action[f"{config.gripper}.pos"] = target
        robot.send_action(action)

        if now >= next_print:
            print(
                f"[gripper] {label} target={target:.3f} "
                f"observed={obs[f'{config.gripper}.pos']:.4f} {_format_obs(obs, names[:-1])}",
                flush=True,
            )
            next_print = now + 0.25

        next_tick += period


def _cycle_gripper(
    robot: YamFollower,
    config: YamFollowerConfig,
    names: list[str],
    low: float,
    high: float,
    cycles: int,
    phase_time: float,
    fps: float,
) -> None:
    print(
        f"[gripper] cycling between {low:.3f} and {high:.3f} for {cycles} cycles",
        flush=True,
    )
    for cycle in range(1, cycles + 1):
        print(f"[gripper] cycle {cycle}/{cycles}: high", flush=True)
        _drive_gripper_phase(robot, config, names, high, "high", phase_time, fps)
        print(f"[gripper] cycle {cycle}/{cycles}: low", flush=True)
        _drive_gripper_phase(robot, config, names, low, "low", phase_time, fps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-seconds", type=float, default=5.0)
    parser.add_argument("--read-fps", type=float, default=5.0)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=0.9)
    parser.add_argument("--phase-time", type=float, default=1.5)
    parser.add_argument("--command-fps", type=float, default=20.0)
    parser.add_argument("--keep-serve", action="store_true", help="disconnect without asking serve to torque off")
    args = parser.parse_args()

    config = YamFollowerConfig(
        serve_host=args.host,
        serve_port=args.port,
        connect_timeout=args.connect_timeout,
        disable_torque_on_disconnect=not args.keep_serve,
    )
    robot = YamFollower(config)
    names = list(config.joints) + [config.gripper]

    robot.connect()
    try:
        print(f"[connect] start_joints={robot.start_joints}", flush=True)
        _read_positions(robot, names, args.read_seconds, args.read_fps)
        _cycle_gripper(robot, config, names, args.low, args.high, args.cycles, args.phase_time, args.command_fps)
        final_obs = robot.get_observation()
        print(f"[done] final {_format_obs(final_obs, names)}", flush=True)
    finally:
        robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
