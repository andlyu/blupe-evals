import json

import pytest

from blupe_evals.station import load_station_configs
from scripts.lelab_station_hub import _station_recording_preset


def test_load_station_configs_from_object(tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "stations": [
                    {
                        "id": "so101-1",
                        "name": "Station One",
                        "base_url": "http://jetson.local:8091/",
                        "cameras": ["front", "side"],
                    }
                ]
            }
        )
    )

    stations = load_station_configs(path)

    assert len(stations) == 1
    assert stations[0].id == "so101-1"
    assert stations[0].normalized_base_url == "http://jetson.local:8091"
    assert stations[0].as_public_dict()["cameras"] == ["front", "side"]


def test_load_station_configs_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            [
                {"id": "dup", "base_url": "http://a"},
                {"id": "dup", "base_url": "http://b"},
            ]
        )
    )

    with pytest.raises(ValueError, match="duplicate station id"):
        load_station_configs(path)


def test_station_recording_preset_uses_hub_camera_urls(tmp_path, monkeypatch):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "stations": [
                    {
                        "id": "so101-1",
                        "name": "Station One",
                        "base_url": "http://jetson.local:8091",
                        "cameras": ["front", "wrist"],
                    }
                ]
            }
        )
    )
    station = load_station_configs(path)[0]

    def fake_get_json(station_arg, path_arg):
        assert station_arg == station
        assert path_arg == "/api/status"
        return {
            "robots": [
                {"role": "follower", "type": "so101_follower", "id": "blupe_follower", "port": "/dev/ttyACM0"},
                {"role": "leader", "type": "so101_leader", "id": "blupe_leader", "port": "/dev/ttyACM1"},
            ],
            "robot_profiles": [
                {
                    "id": "blupe_so101",
                    "name": "BluPe SO101",
                    "follower": {"id": "blupe_follower", "port": "/dev/ttyACM0"},
                    "leader": {"id": "blupe_leader", "port": "/dev/ttyACM1"},
                    "cameras": ["front", "wrist"],
                }
            ],
            "cameras": [
                {
                    "name": "front",
                    "url": "http://127.0.0.1:8080/0",
                    "frames_dir": "front",
                    "frames_file": "front/frames.jsonl",
                    "lerobot_key": "observation.images.front",
                }
            ]
        }

    monkeypatch.setattr("scripts.lelab_station_hub._station_get_json", fake_get_json)

    preset = _station_recording_preset(station)

    assert preset["station"]["id"] == "so101-1"
    assert preset["robot_profiles"][0]["id"] == "blupe_so101"
    assert preset["robots"][0]["role"] == "follower"
    assert preset["robots"][1]["role"] == "leader"
    assert preset["defaults"]["capture_mode"] == "continuous"
    assert preset["cameras"][0]["name"] == "front"
    assert preset["cameras"][0]["source_url"] == "http://127.0.0.1:8080/0"
    assert preset["cameras"][0]["stream_url"] == "/api/stations/so101-1/camera/front.mjpg"
    assert preset["cameras"][1]["name"] == "wrist"
    assert preset["cameras"][1]["lerobot_key"] == "observation.images.wrist"
