from pathlib import Path


CAMERA_RELAY_SOURCE = Path("YAM_control/camera_relay.py").read_text()
SO101_SOURCE = Path("scripts/so101_web_intervene.py").read_text()
START_SCRIPT_SOURCE = Path("scripts/start_jetson_so101_remote_policy.sh").read_text()


def test_camera_relay_reports_and_rejects_stale_frames() -> None:
    assert "last_frame_mono" in CAMERA_RELAY_SOURCE
    assert "def fresh_jpeg(self):" in CAMERA_RELAY_SOURCE
    assert "stale frame age=" in CAMERA_RELAY_SOURCE
    assert "self.send_error(503" in CAMERA_RELAY_SOURCE
    assert '"ok": all(cam.status()["fresh"] for cam in CAMS.values())' in CAMERA_RELAY_SOURCE


def test_camera_relay_reopens_after_repeated_read_failures() -> None:
    assert "--reopen-after-failures" in CAMERA_RELAY_SOURCE
    assert "self.consecutive_failures += 1" in CAMERA_RELAY_SOURCE
    assert "self._reopen_capture(self.error)" in CAMERA_RELAY_SOURCE
    assert "--height" in CAMERA_RELAY_SOURCE
    assert "default=360" in CAMERA_RELAY_SOURCE


def test_eval_proxy_drops_stale_mjpeg_sources() -> None:
    assert "MJPEG_CAMERA_STALE_TIMEOUT_S" in SO101_SOURCE
    assert "self.last_frame_mono: float | None = None" in SO101_SOURCE
    assert "stale MJPEG frame from" in SO101_SOURCE
    assert "def _drop_mjpeg_camera" in SO101_SOURCE
    assert "reset mjpeg camera" in SO101_SOURCE


def test_so101_launcher_owns_camera_relay_and_semantic_routes() -> None:
    assert "SO101_CAMERA_RELAY_ENABLED" in START_SCRIPT_SOURCE
    assert 'YAM_control/camera_relay.py' in START_SCRIPT_SOURCE
    assert "CAMERA_RELAY_DEVICES" in START_SCRIPT_SOURCE
    assert "front=http://127.0.0.1:${CAMERA_RELAY_PORT}/2" in START_SCRIPT_SOURCE
    assert "side=http://127.0.0.1:${CAMERA_RELAY_PORT}/1" in START_SCRIPT_SOURCE
    assert "wrist=http://127.0.0.1:${CAMERA_RELAY_PORT}/0" in START_SCRIPT_SOURCE
    assert "/health" in START_SCRIPT_SOURCE
    assert "Camera relay did not become healthy" in START_SCRIPT_SOURCE
