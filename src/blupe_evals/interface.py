"""User ⇄ blupe-evals surface — the user brings a ROBOT and a POLICY LOOP.

The robot is ALWAYS the user's; we only wrap it. blupe-evals owns ONE decision:
who drives the arm right now — us (teleop, for setup/reset) or the user's policy
loop. We hand off and reclaim; we never sit inside the robot↔policy interaction.

The user provides exactly two things:

  1. Robot   — how to READ and COMMAND their arm (a thin adapter over their stack).
               `command` is a RAW joint write; we wrap it with safety (SafeRobot).

  2. run loop — `run(robot, stop)`: their whole rollout. They own the robot↔policy
               interaction (query cadence, chunking, horizon). They read `robot`
               and command `robot`, and return when `stop()` becomes True.

                   def run(robot, stop):          # USER writes this
                       while not stop():
                           obs = robot.read()
                           act = my_model(obs)     # their cadence / chunking
                           robot.command(act)      # passes through OUR safety

We add, AROUND their robot, only what must be ours (see console.py):
  - SAFETY we never cede: every command is rate-clamped, and we can DISARM the arm
    instantly — the kill switch — so a misbehaving loop is stopped even if it
    ignores `stop()`.
  - the gate: start(run) hands off, stop() cuts + reclaims + holds, plus teleop.
  - sim: the same `run` runs against a simulated copy of their arm first.

No action format, no chunk shape, no us-in-the-loop — that complexity stays in
the user's `run`.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class Observation:
    """What `Robot.read()` returns — handed to the user's run loop as-is."""

    joint_pos: np.ndarray                       # (n_arm,) arm joints, rad, robot's own order
    gripper: float                              # opening, normalized 0=closed .. 1=open
    ee_pos: np.ndarray                          # (3,) end-effector xyz, base frame, m (FK)
    images: dict[str, np.ndarray] = field(default_factory=dict)  # name -> HxWx3 uint8 RGB


@runtime_checkable
class Robot(Protocol):
    """The user's arm as blupe-evals needs it.

    `command` is a RAW joint write — we wrap it with safety (SafeRobot) and can
    disarm it. The robot stays the user's; we only interpose."""

    def read(self) -> Observation: ...
    def command(self, joint_pos: np.ndarray, gripper: Optional[float] = None) -> None: ...
    def info(self) -> dict: ...                 # {n_arm, gripper_index, joint_limits, ...}
    def home(self) -> None: ...                 # ease to a known safe pose, then hold


# The policy plug is just the user's run loop — a callable, no schema to bless:
#   run(robot, stop) -> None   where stop() -> bool tells it to return.
RunLoop = Callable[["Robot", Callable[[], bool]], None]
