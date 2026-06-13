"""Serve a real i2rt YAM arm to the LeRobot ``yam_follower`` plugin.

Run this with the i2rt Python environment, not the LeRobot environment:

    PYTHONPATH=$HOME/i2rt:$HOME/lerobot-robot-yam \
      $HOME/i2rt/.venv/bin/python -m lerobot_robot_yam.yam_serve --channel can0

Wire protocol is newline-delimited JSON:

* server -> client once: ``{"start_joints": [6 arm joints]}``
* client -> server read: ``{"obs": true}``
* server -> client read: ``{"joints": [6 arm joints, gripper]}``
* client -> server move: ``{"q": [6 arm joints], "g": 0..1}``
* client -> server stop: ``{"shutdown": true}``

Arm joints are radians. The gripper is normalized 0..1 in i2rt command space.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

import numpy as np

N_ARM_JOINTS = 6
DEFAULT_MAX_VEL = 0.6
DEFAULT_GRIPPER_LEAD = 0.9 * 6 / 71


def disable_motorchain(robot: Any) -> None:
    """Stop i2rt's control thread, turn motor torque off, then close CAN."""
    try:
        robot._stop_event.set()
        robot._server_thread.join(timeout=2.0)
    except Exception as exc:
        print(f"[off] stop thread: {exc}", flush=True)

    try:
        robot.motor_chain.running = False
    except Exception:
        pass

    chain = getattr(robot, "motor_chain", None)
    if chain is None:
        return

    try:
        n_motors = len(chain)
    except Exception:
        n_motors = N_ARM_JOINTS + 1

    try:
        motor_interface = chain.motor_interface
        for motor_id in range(1, n_motors + 1):
            try:
                motor_interface.motor_off(motor_id)
            except Exception as exc:
                print(f"[off] motor_off {motor_id} error: {exc}", flush=True)
    except Exception as exc:
        print(f"[off] disable error: {exc}", flush=True)

    try:
        chain.close()
    except Exception:
        pass

    print("[off] all motors off; arm is limp.", flush=True)


class _FakeRobot:
    """No-hardware robot used for protocol smoke tests."""

    def __init__(self) -> None:
        self._q = np.zeros(N_ARM_JOINTS + 1)

        class _MotorInterface:
            def motor_off(self, motor_id: int) -> None:
                pass

        class _MotorChain:
            running = True
            motor_interface = _MotorInterface()

            def __len__(self) -> int:
                return N_ARM_JOINTS + 1

            def close(self) -> None:
                self.running = False

        self.motor_chain = _MotorChain()

    def get_joint_pos(self) -> np.ndarray:
        return self._q.copy()

    def command_joint_pos(self, command: np.ndarray) -> None:
        self._q = np.asarray(command, dtype=float)

    def get_robot_info(self) -> dict[str, int]:
        return {"gripper_index": N_ARM_JOINTS}


def _float_list(values: np.ndarray) -> list[float]:
    return [float(x) for x in values]


def _make_robot(args: argparse.Namespace) -> Any:
    if args.fake:
        print("[serve] FAKE mode; no hardware will be touched.", flush=True)
        return _FakeRobot()

    from i2rt.robots.get_robot import get_yam_robot

    kwargs: dict[str, Any] = {}
    if args.gripper_limits is not None:
        kwargs["gripper_limits_override"] = np.asarray(args.gripper_limits, dtype=float)

    print(f"[serve] connecting real YAM on {args.channel} ...", flush=True)
    return get_yam_robot(args.channel, **kwargs)


def _handle_client(
    robot: Any,
    conn: socket.socket,
    max_vel: float,
    gripper_lead: float,
    gripper_index: int | None,
) -> bool:
    f = conn.makefile("rwb")
    last = np.asarray(robot.get_joint_pos(), dtype=float)[:N_ARM_JOINTS].copy()
    f.write((json.dumps({"start_joints": _float_list(last)}) + "\n").encode())
    f.flush()

    last_t: float | None = None
    n_received = 0
    should_shutdown = False

    for line in f:
        if not line.strip():
            continue

        msg = json.loads(line.decode())
        if msg.get("shutdown"):
            print("[serve] shutdown requested.", flush=True)
            should_shutdown = True
            break

        if msg.get("obs"):
            current = np.asarray(robot.get_joint_pos(), dtype=float)
            f.write((json.dumps({"joints": _float_list(current)}) + "\n").encode())
            f.flush()
            continue

        q = msg.get("q")
        if q is None:
            continue

        now = time.monotonic()
        dt = 0.02 if last_t is None else min(max(now - last_t, 1e-4), 0.1)
        last_t = now

        step = max_vel * dt
        target = np.asarray(q[:N_ARM_JOINTS], dtype=float)
        last = last + np.clip(target - last, -step, step)

        command = np.asarray(robot.get_joint_pos(), dtype=float).copy()
        command[:N_ARM_JOINTS] = last

        g = msg.get("g")
        if g is not None and gripper_index is not None and len(command) > gripper_index:
            desired = min(max(float(g), 0.0), 1.0)
            actual = float(command[gripper_index])
            if desired > actual:
                command[gripper_index] = min(desired, actual + gripper_lead)
            else:
                command[gripper_index] = max(desired, actual - gripper_lead)

        robot.command_joint_pos(command)

        timestamp = msg.get("t")
        if timestamp is not None:
            f.write((json.dumps({"ack": timestamp}) + "\n").encode())
            f.flush()

        n_received += 1
        if n_received % 25 == 0:
            print(f"[serve] {n_received} cmds; arm={np.round(last, 3)}", flush=True)

    return should_shutdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--max-vel", type=float, default=DEFAULT_MAX_VEL)
    parser.add_argument("--gripper-lead", type=float, default=DEFAULT_GRIPPER_LEAD)
    parser.add_argument("--gripper-limits", nargs=2, type=float, default=None, metavar=("CLOSED", "OPEN"))
    parser.add_argument("--fake", action="store_true", help="run protocol server without touching hardware")
    args = parser.parse_args()

    robot = _make_robot(args)
    info = robot.get_robot_info() if hasattr(robot, "get_robot_info") else {}
    gripper_index = info.get("gripper_index", N_ARM_JOINTS)
    print(
        f"[serve] gripper index={gripper_index}; gripper lead={args.gripper_lead:.3f}; "
        "units: arm radians, gripper 0..1",
        flush=True,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[serve] listening on {args.host}:{args.port}; Ctrl-C turns torque off.", flush=True)

    try:
        while True:
            conn, addr = srv.accept()
            print(f"[serve] client {addr} connected.", flush=True)
            try:
                if _handle_client(robot, conn, args.max_vel, args.gripper_lead, gripper_index):
                    break
            except Exception as exc:
                print(f"[serve] client stream ended: {exc}", flush=True)
            finally:
                conn.close()
                print("[serve] client disconnected; holding last pose.", flush=True)
    except KeyboardInterrupt:
        print("[serve] interrupted.", flush=True)
    finally:
        disable_motorchain(robot)
        srv.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
