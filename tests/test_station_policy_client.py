import numpy as np
import pytest

from blupe_evals.station import default_camera_configs, parse_camera_config_specs
from blupe_evals.station.policy_client import HttpPolicyClient


def test_default_camera_configs_use_semantic_lerobot_keys():
    cameras = default_camera_configs()

    assert [cam.name for cam in cameras] == ["front", "side", "wrist"]
    assert [cam.lerobot_key for cam in cameras] == [
        "observation.images.front",
        "observation.images.side",
        "observation.images.wrist",
    ]


def test_parse_camera_config_specs_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate camera names"):
        parse_camera_config_specs(["front=http://a", "front=http://b"])


def test_http_policy_client_act_posts_named_images(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"actions": [[1, 2, 3, 4, 5, 6]]}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("blupe_evals.station.policy_client.requests.post", fake_post)
    client = HttpPolicyClient("http://policy.local/", timeout_s=12.0)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    actions = client.act(
        images={"front": image, "side": image},
        state=np.zeros(6, dtype=np.float32),
        instruction="test task",
        joints=["j0", "j1", "j2", "j3", "j4", "j5"],
    )

    assert actions.shape == (1, 6)
    assert captured["url"] == "http://policy.local/act"
    assert captured["timeout"] == 12.0
    assert captured["json"]["camera_order"] == ["front", "side"]
    assert set(captured["json"]["images"]) == {"front", "side"}
    assert captured["json"]["images"]["front"]["encoding"] == "jpeg_base64"
