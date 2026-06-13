import json
import logging
import socket
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_yam_follower import YamFollowerConfig

logger = logging.getLogger(__name__)


class YamFollower(Robot):
    """i2rt YAM as a LeRobot follower, driven through `yam_real_serve.py` over TCP.

    Protocol (newline-JSON, the serve's wire format):
      serve -> us, once on connect:   {"start_joints": [q1..q6]}    (radians, seed)
      us -> serve, observation pull:  {"obs": true}  ->  {"joints": [q1..q6, g]}
      us -> serve, action:            {"q": [q1..q6], "g": 0..1}
      us -> serve, on disconnect:     {"shutdown": true}            (torque off)

    All motion safety is robot-side in the serve; this class is a thin client.
    """

    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig):
        super().__init__(config)
        self.config = config
        self.motor_names = list(config.joints) + [config.gripper]   # 6 arm + gripper
        self._sock: socket.socket | None = None
        self._f = None
        self.start_joints = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._sock is not None and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        try:
            self._sock = socket.create_connection(
                (self.config.serve_host, self.config.serve_port), timeout=self.config.connect_timeout
            )
        except OSError as e:
            raise DeviceNotConnectedError(
                f"could not reach yam_real_serve at {self.config.serve_host}:{self.config.serve_port} "
                f"({e}). Start the serve on the robot computer first."
            )
        self._f = self._sock.makefile("rwb")
        line = self._f.readline()                       # the no-jump handshake
        if not line:
            self._sock.close()
            self._sock = None
            raise DeviceNotConnectedError("serve sent no start_joints handshake (arm not ready?)")
        self.start_joints = json.loads(line.decode()).get("start_joints")
        for cam in self.cameras.values():
            cam.connect()
        logger.info(f"{self} connected; start_joints={self.start_joints}")

    @property
    def is_calibrated(self) -> bool:
        return True                                     # i2rt arm is pre-calibrated

    def calibrate(self) -> None:
        pass                                            # no-op: i2rt owns calibration

    def configure(self) -> None:
        pass

    def _read_joints(self) -> list[float]:
        """Pull the arm's current joints from the serve (radians + gripper 0..1)."""
        self._f.write((json.dumps({"obs": True}) + "\n").encode())
        self._f.flush()
        while True:
            line = self._f.readline()
            if not line:
                raise DeviceNotConnectedError("serve closed during observation")
            msg = json.loads(line.decode())
            if "joints" in msg:                         # skip any interleaved acks
                return msg["joints"]

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        q = self._read_joints()
        obs: dict = {f"{m}.pos": float(q[i]) for i, m in enumerate(self.motor_names)}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        goal = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}
        current: dict[str, float] | None = None

        missing_joints = [joint for joint in self.config.joints if joint not in goal]
        if missing_joints or self.config.max_relative_target is not None:
            q = self._read_joints()
            current = {m: float(q[i]) for i, m in enumerate(self.motor_names)}
            for joint in missing_joints:
                goal[joint] = current[joint]

        if self.config.max_relative_target is not None:   # extra cap on top of serve clamp
            cap = self.config.max_relative_target
            for m in goal:
                c = current.get(m, goal[m]) if current is not None else goal[m]
                goal[m] = max(c - cap, min(c + cap, goal[m]))

        msg: dict = {"q": [goal[j] for j in self.config.joints]}
        if self.config.gripper in goal:
            msg["g"] = goal[self.config.gripper]
        self._f.write((json.dumps(msg) + "\n").encode())
        self._f.flush()
        return {f"{m}.pos": goal[m] for m in goal}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            if self.config.disable_torque_on_disconnect:
                self._f.write((json.dumps({"shutdown": True}) + "\n").encode())
                self._f.flush()
        except OSError:
            pass
        try:
            self._sock.close()
        finally:
            self._sock, self._f = None, None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
