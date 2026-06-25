"""Semantic camera configuration used by station and LeLab integration code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    id: int
    name: str
    url: str

    @property
    def lerobot_key(self) -> str:
        return f"observation.images.{self.name}"


def default_camera_configs(names: tuple[str, ...] = ("front", "side", "wrist")) -> list[CameraConfig]:
    return [
        CameraConfig(idx, name, f"http://127.0.0.1:8080/cam{idx}.mjpg")
        for idx, name in enumerate(names)
    ]


def parse_camera_config_specs(specs: list[str], default_names: tuple[str, ...] = ("front", "side", "wrist")) -> list[CameraConfig]:
    if not specs:
        return default_camera_configs(default_names)
    configs: list[CameraConfig] = []
    for idx, spec in enumerate(specs):
        if "=" not in spec:
            raise ValueError("--camera must be NAME=URL, for example --camera front=http://127.0.0.1:8080/cam0.mjpg")
        name, url = spec.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"invalid camera name: {name!r}")
        if not url:
            raise ValueError(f"missing URL for camera {name!r}")
        configs.append(CameraConfig(idx, name, url))
    names = [cam.name for cam in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate camera names in --camera: {names}")
    return configs
