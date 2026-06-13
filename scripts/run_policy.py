#!/usr/bin/env python3
"""Run a `run(robot, stop)` policy against a local YAM serve.

This is the robot-side policy runner used by the relay agent. It connects to the
serve on 127.0.0.1:5599, wraps it in the same velocity-clamped policy seam used by
the eval loop, and keeps policy execution on the robot computer.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

N_ARM = 6
DEFAULT_MAX_VEL = 0.6


@dataclass
class Observation:
    joint_pos: np.ndarray
    ee_pos: np.ndarray
    gripper: float = 0.0


def load_run(spec: str) -> Callable:
    """Load '/path/to/policy.py:run' or 'module.path:run'."""
    path, _, attr = spec.partition(":")
    attr = attr or "run"
    if path.endswith(".py"):
        p = Path(path).expanduser()
        mod_spec = importlib.util.spec_from_file_location(p.stem, p)
        if mod_spec is None or mod_spec.loader is None:
            raise RuntimeError(f"cannot load policy {spec!r}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(path)
    return getattr(mod, attr)


class ServeRobot:
    """Robot adapter over the newline-JSON YAM serve protocol."""

    def __init__(self, host: str, port: int, timeout: float):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rwb")
        line = self.file.readline()
        if not line:
            raise RuntimeError("serve closed before start_joints")
        hello = json.loads(line.decode())
        self.start_joints = np.asarray(hello["start_joints"], dtype=float)[:N_ARM]
        self._last_gripper = 1.0

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()

    def read(self) -> Observation:
        self.file.write(b'{"obs":true}\n')
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise RuntimeError("serve closed during obs")
        joints = np.asarray(json.loads(line.decode())["joints"], dtype=float)
        if len(joints) > N_ARM:
            self._last_gripper = float(joints[N_ARM])
        return Observation(joint_pos=joints[:N_ARM].copy(), ee_pos=np.zeros(3), gripper=self._last_gripper)

    def command(self, joint_pos, gripper=None) -> None:
        msg = {"q": [float(x) for x in np.asarray(joint_pos, dtype=float)[:N_ARM]]}
        if gripper is not None:
            self._last_gripper = float(gripper)
            msg["g"] = self._last_gripper
        self.file.write((json.dumps(msg) + "\n").encode())
        self.file.flush()


class SafeRobot:
    """Velocity-clamped, killable policy wrapper."""

    def __init__(self, robot: ServeRobot, max_vel: float):
        self.robot = robot
        self.max_vel = max_vel
        self._armed = False
        self._last_t = None
        self._last = np.asarray(robot.read().joint_pos, dtype=float)[:N_ARM].copy()

    def arm(self) -> None:
        self._last = np.asarray(self.robot.read().joint_pos, dtype=float)[:N_ARM].copy()
        self._last_t = None
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def read(self) -> Observation:
        return self.robot.read()

    def command(self, joint_pos, gripper=None) -> None:
        if not self._armed:
            return
        now = time.monotonic()
        dt = (now - self._last_t) if self._last_t is not None else 0.005
        dt = min(max(dt, 1e-4), 0.1)
        self._last_t = now
        step = self.max_vel * dt
        desired = np.asarray(joint_pos, dtype=float)[:N_ARM]
        self._last = self._last + np.clip(desired - self._last, -step, step)
        self.robot.command(self._last, gripper)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="Python callable: path.py:run or module.path:run")
    ap.add_argument("--serve-host", default="127.0.0.1")
    ap.add_argument("--serve-port", type=int, default=5599)
    ap.add_argument("--connect-timeout", type=float, default=5.0)
    ap.add_argument("--max-vel", type=float, default=DEFAULT_MAX_VEL)
    ap.add_argument("--max-runtime", type=float, default=0.0, help="seconds; 0 disables")
    args = ap.parse_args()

    stop_evt = Event()

    def _stop(signum, _frame):
        print(f"[runner] stop signal {signum}", flush=True)
        stop_evt.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    run = load_run(args.policy)
    robot = ServeRobot(args.serve_host, args.serve_port, args.connect_timeout)
    safe = SafeRobot(robot, args.max_vel)
    started = time.monotonic()

    def stop() -> bool:
        if args.max_runtime and time.monotonic() - started >= args.max_runtime:
            print("[runner] max runtime reached", flush=True)
            return True
        return stop_evt.is_set()

    print(f"[runner] connected start_joints={np.round(robot.start_joints, 4).tolist()}", flush=True)
    print(f"[runner] policy={args.policy} serve={args.serve_host}:{args.serve_port}", flush=True)
    safe.arm()
    try:
        run(safe, stop)
    finally:
        safe.disarm()
        robot.close()
    print("[runner] policy complete", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[runner] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
