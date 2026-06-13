from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.robot import RobotConfig


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerConfig(RobotConfig):
    """YAM follower that drives the arm through a running `yam_real_serve.py`.

    Unlike the Dynamixel/Damiao followers, this robot owns NO motor bus: it opens a
    TCP socket to the serve (which lives in i2rt's own venv and holds all the safety:
    velocity clamp, hold-on-disconnect, torque-off). So lerobot and i2rt never share an
    interpreter — sidestepping the numpy 1.x/2.x conflict between the two stacks.
    """

    # The yam_real_serve.py endpoint (default: same machine). Start the serve first.
    serve_host: str = "127.0.0.1"
    serve_port: int = 5599
    connect_timeout: float = 5.0

    # Arm joints in i2rt CAN order (base -> wrist); the gripper is appended separately.
    joints: list[str] = field(
        default_factory=lambda: ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    )
    gripper: str = "gripper"

    # UNITS: radians for the arm, normalized 0..1 for the gripper (0=closed, 1=open) —
    # passed through to/from the serve verbatim. (reBot exposes degrees; we keep YAM's
    # native i2rt units to avoid a conversion bug — the #1 silent-failure risk.)

    # Safety: cap |target - current| per send_action, in radians. None = rely on the
    # serve's own velocity clamp (which is always active regardless).
    max_relative_target: float | None = None

    disable_torque_on_disconnect: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
