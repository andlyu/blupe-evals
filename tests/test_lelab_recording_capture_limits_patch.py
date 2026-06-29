from scripts.apply_lelab_recording_capture_limits_patch import (
    CREATE_CONFIG_OLD,
    HELPER_BLOCK,
    IMPORT_OLD,
    patch_text,
)


class _Request:
    def __init__(self, fps: int = 30, streaming_encoding: bool = True) -> None:
        self.fps = fps
        self.streaming_encoding = streaming_encoding
        self.cameras = {
            "front": {"fps": 30},
            "side": {"fps": 30},
            "wrist": {"fps": 30},
        }


class _Logger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


def _helper_namespace() -> dict:
    namespace: dict = {"logger": _Logger()}
    exec("import os\nRecordingRequest = object\n" + HELPER_BLOCK, namespace)
    return namespace


def _source() -> str:
    return "\n".join(
        [
            IMPORT_OLD,
            "class RecordingRequest:",
            "    pass",
            "",
            CREATE_CONFIG_OLD + "    return RecordConfig()\n",
        ]
    )


def test_recording_capture_limits_patch_adds_env_controls() -> None:
    patched, changed = patch_text(_source())

    assert changed is True
    assert "import os" in patched
    assert "LELAB_RECORD_MAX_FPS" in patched
    assert "LELAB_RECORD_CAP_CAMERA_FPS" in patched
    assert "LELAB_RECORD_STREAMING_ENCODING" in patched
    assert "request = _apply_recording_capture_limits(request)" in patched


def test_recording_capture_limits_patch_keeps_all_cameras_and_does_not_cap_camera_fps_by_default() -> None:
    patched, _ = patch_text(_source())

    assert "filtered" not in HELPER_BLOCK
    assert "LELAB_RECORD_CAMERAS" not in patched
    assert 'cap_camera_fps = _env_bool("LELAB_RECORD_CAP_CAMERA_FPS", False)' in patched
    assert 'camera_data["fps"] = max_fps' in patched
    assert "if cap_camera_fps:" in patched
    assert "for camera_name, camera_data in request.cameras.items()" in patched


def test_recording_capture_limits_patch_is_idempotent() -> None:
    patched, changed = patch_text(_source())
    patched_again, changed_again = patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_recording_capture_limits_caps_dataset_fps_not_camera_fps_by_default(monkeypatch) -> None:
    monkeypatch.setenv("LELAB_RECORD_MAX_FPS", "15")
    monkeypatch.delenv("LELAB_RECORD_CAP_CAMERA_FPS", raising=False)
    request = _Request()

    _helper_namespace()["_apply_recording_capture_limits"](request)

    assert request.fps == 15
    assert request.cameras == {
        "front": {"fps": 30},
        "side": {"fps": 30},
        "wrist": {"fps": 30},
    }


def test_recording_capture_limits_caps_camera_fps_only_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LELAB_RECORD_MAX_FPS", "15")
    monkeypatch.setenv("LELAB_RECORD_CAP_CAMERA_FPS", "1")
    request = _Request()
    request.cameras["metadata"] = "not-a-camera-dict"

    _helper_namespace()["_apply_recording_capture_limits"](request)

    assert request.fps == 15
    assert request.cameras["front"]["fps"] == 15
    assert request.cameras["side"]["fps"] == 15
    assert request.cameras["wrist"]["fps"] == 15
    assert request.cameras["metadata"] == "not-a-camera-dict"


def test_recording_capture_limits_ignores_invalid_max_fps(monkeypatch) -> None:
    monkeypatch.setenv("LELAB_RECORD_MAX_FPS", "fast")
    request = _Request()

    _helper_namespace()["_apply_recording_capture_limits"](request)

    assert request.fps == 30
    assert request.cameras["front"]["fps"] == 30


def test_recording_capture_limits_can_disable_streaming_encoding(monkeypatch) -> None:
    monkeypatch.setenv("LELAB_RECORD_STREAMING_ENCODING", "0")
    request = _Request(streaming_encoding=True)

    _helper_namespace()["_apply_recording_capture_limits"](request)

    assert request.streaming_encoding is False
