"""LeRobot plugin: i2rt YAM as a follower robot.

Importing this package registers the `yam_follower` robot type (via the
`@RobotConfig.register_subclass` decorator on the config), so
`make_robot_from_config({"type": "yam_follower", ...})` and the
`--robot.type=yam_follower` CLI both resolve to YamFollower.
"""

try:
    from .config_yam_follower import YamFollowerConfig
    from .yam_follower import YamFollower
except ModuleNotFoundError as exc:
    if exc.name != "lerobot":
        raise
    YamFollowerConfig = None
    YamFollower = None

__all__ = ["YamFollowerConfig", "YamFollower"]
