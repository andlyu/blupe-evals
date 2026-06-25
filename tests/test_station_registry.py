import json

import pytest

from blupe_evals.station import load_station_configs


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
