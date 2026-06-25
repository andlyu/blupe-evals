"""Station registry used by a central LeLab control surface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StationConfig:
    id: str
    name: str
    base_url: str
    robot_type: str = "so101_follower"
    cameras: list[str] = field(default_factory=lambda: ["front", "side", "wrist"])

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.normalized_base_url,
            "robot_type": self.robot_type,
            "cameras": list(self.cameras),
        }


def load_station_configs(path: Path) -> list[StationConfig]:
    payload = json.loads(path.read_text())
    raw_stations = payload.get("stations", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_stations, list):
        raise ValueError("station config must be a JSON list or an object with a stations list")
    stations: list[StationConfig] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_stations, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"station {idx} must be an object")
        station_id = str(raw.get("id") or raw.get("name") or f"station-{idx}").strip()
        if not station_id:
            raise ValueError(f"station {idx} needs an id")
        if station_id in seen:
            raise ValueError(f"duplicate station id: {station_id}")
        base_url = str(raw.get("base_url") or raw.get("url") or "").strip()
        if not base_url:
            raise ValueError(f"station {station_id} needs base_url")
        cameras_raw = raw.get("cameras") or ["front", "side", "wrist"]
        if not isinstance(cameras_raw, list):
            raise ValueError(f"station {station_id} cameras must be a list")
        stations.append(
            StationConfig(
                id=station_id,
                name=str(raw.get("name") or station_id),
                base_url=base_url,
                robot_type=str(raw.get("robot_type") or "so101_follower"),
                cameras=[str(camera) for camera in cameras_raw],
            )
        )
        seen.add(station_id)
    return stations
