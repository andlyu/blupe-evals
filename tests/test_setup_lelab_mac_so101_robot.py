import pytest

from scripts.setup_lelab_mac_so101_robot import parse_cameras


def test_parse_cameras_uses_browser_rebindable_native_mac_defaults() -> None:
    cameras = parse_cameras(["front=0", "side=1", "wrist=2"])

    assert cameras == [
        {
            "id": "front",
            "name": "front",
            "type": "opencv",
            "camera_index": 0,
            "device_id": "",
            "width": 640,
            "height": 360,
            "fps": 30,
            "fourcc": "MJPG",
        },
        {
            "id": "side",
            "name": "side",
            "type": "opencv",
            "camera_index": 1,
            "device_id": "",
            "width": 640,
            "height": 360,
            "fps": 30,
            "fourcc": "MJPG",
        },
        {
            "id": "wrist",
            "name": "wrist",
            "type": "opencv",
            "camera_index": 2,
            "device_id": "",
            "width": 640,
            "height": 360,
            "fps": 30,
            "fourcc": "MJPG",
        },
    ]


def test_parse_cameras_rejects_missing_index_mapping() -> None:
    with pytest.raises(ValueError, match="NAME=INDEX"):
        parse_cameras(["front"])
