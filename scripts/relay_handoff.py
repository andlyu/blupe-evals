"""Pure relay handoff rules shared by the Mac <-> Quest bridge and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReleasedLink:
    connect_arm: bool
    link: Any | None
    released: bool


def release_teleop_link_for_policy(connect_arm: bool, link: Any | None) -> ReleasedLink:
    """Let robot-side policy own the serve without changing logical CONNECT state.

    The teleop socket and robot-side policy runner cannot both own the serve. Releasing
    the socket is therefore correct, but CONNECT must stay logically on so the bridge can
    reconnect teleop after POLICY finishes.
    """
    if link is None:
        return ReleasedLink(connect_arm=connect_arm, link=None, released=False)
    link.close(shutdown=False)
    return ReleasedLink(connect_arm=connect_arm, link=None, released=True)


def should_attach_teleop_link(
    connect_arm: bool,
    relay_arm_status: str,
    state: str,
    link: Any | None,
) -> bool:
    """Return true when the bridge should attach to the serve tunnel.

    CONNECT means the robot-side serve is available. The Mac should only own the command
    socket while an action that sends commands is active.
    """
    return bool(
        connect_arm
        and relay_arm_status == "serve_ready"
        and mode_owns_command_socket(state)
        and link is None
    )


def mode_owns_command_socket(state: str) -> bool:
    """Only active motion modes may own the Mac->serve command socket."""
    return state in ("TELEOP", "GO_HOME")


def should_release_command_link_for_state(state: str, link: Any | None) -> bool:
    """IDLE/POLICY must not keep a teleop command socket open."""
    return link is not None and not mode_owns_command_socket(state)


def can_switch_state(current_state: str, target_state: str, policy_active: bool) -> bool:
    """Freeze mode switching while robot-side policy is starting/running."""
    if current_state == "POLICY" and policy_active and target_state != "IDLE":
        return False
    return True


def relay_result(data: dict[str, Any]) -> str:
    """Normalize relay command responses into the human/result string."""
    return str(((data.get("data") or {}).get("result") or data.get("err") or ""))


def policy_start_accepted(data: dict[str, Any]) -> bool:
    result = relay_result(data)
    return bool(
        data.get("ok")
        and ("POLICY started" in result or "policy already running" in result)
    )


def should_retry_policy_start(
    data: dict[str, Any],
    attempt: int,
    max_attempts: int,
    relay_arm_status: str,
) -> bool:
    """Retry the known handoff race after teleop releases the serve socket."""
    return bool(
        attempt + 1 < max_attempts
        and relay_arm_status == "serve_ready"
        and "serve is off" in relay_result(data)
    )
