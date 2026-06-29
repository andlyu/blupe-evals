from pathlib import Path


CAMERA_RELAY_SOURCE = Path("YAM_control/camera_relay.py").read_text()
SO101_SOURCE = Path("scripts/so101_web_intervene.py").read_text()
START_SCRIPT_SOURCE = Path("scripts/start_jetson_so101_remote_policy.sh").read_text()
FAST_LAUNCHER_SOURCE = Path("scripts/launch_so101_eval_ui.sh").read_text()
COMPRESS_SOURCE = Path("scripts/compress_so101_episodes.py").read_text()


def test_camera_relay_reports_and_rejects_stale_frames() -> None:
    assert "CAP_AVFOUNDATION" in CAMERA_RELAY_SOURCE
    assert "cv2.VideoCapture(self.dev, CAPTURE_BACKEND)" in CAMERA_RELAY_SOURCE
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


def test_fast_eval_ui_launcher_starts_ui_without_policy_health_gate() -> None:
    assert 'UI_URL="${SO101_UI_URL:-http://localhost:${UI_PORT}/#setup}"' in FAST_LAUNCHER_SOURCE
    assert 'CAMERA_READY_TIMEOUT_S="${SO101_CAMERA_READY_TIMEOUT_S:-5}"' in FAST_LAUNCHER_SOURCE
    assert 'POLICY_URL="${SO101_POLICY_URL:-http://127.0.0.1:8202}"' in FAST_LAUNCHER_SOURCE
    assert 'PYTHON="$(command -v "$PYTHON")"' in FAST_LAUNCHER_SOURCE
    assert "cd \"$REPO_ROOT\"" in FAST_LAUNCHER_SOURCE
    assert 'screen -c "$screenrc" -L -dmS "$name" "$@"' in FAST_LAUNCHER_SOURCE
    assert "logfile %s" in FAST_LAUNCHER_SOURCE
    assert "logfile flush 1" in FAST_LAUNCHER_SOURCE
    assert 'REPLACE_PORTS="${SO101_REPLACE_PORTS:-1}"' in FAST_LAUNCHER_SOURCE
    assert 'lsof -tiTCP:"$port" -sTCP:LISTEN' in FAST_LAUNCHER_SOURCE
    assert "kill -KILL $pids" in FAST_LAUNCHER_SOURCE
    assert "Could not stop ${label} listener" in FAST_LAUNCHER_SOURCE
    assert 'stop_port_listener "$UI_PORT" "SO101 eval UI"' in FAST_LAUNCHER_SOURCE
    assert 'stop_port_listener "$CAMERA_RELAY_PORT" "SO101 camera relay"' in FAST_LAUNCHER_SOURCE
    assert 'wait_http "http://127.0.0.1:${UI_PORT}/api/status?log_limit=1"' in FAST_LAUNCHER_SOURCE
    assert 'http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/health' in FAST_LAUNCHER_SOURCE
    assert "front=http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/2" in FAST_LAUNCHER_SOURCE
    assert "side=http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/1" in FAST_LAUNCHER_SOURCE
    assert "wrist=http://${CAMERA_RELAY_HOST}:${CAMERA_RELAY_PORT}/0" in FAST_LAUNCHER_SOURCE
    assert 'post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/connect"' in FAST_LAUNCHER_SOURCE
    assert "8202/health" not in FAST_LAUNCHER_SOURCE


def test_dataset_export_can_ignore_side_and_skip_bad_episodes() -> None:
    assert "--camera" in COMPRESS_SOURCE
    assert "--skip-unusable" in COMPRESS_SOURCE
    assert "converter._episode_plan(path, args.camera, 0)" in COMPRESS_SOURCE
    assert "skipping unusable episode" in COMPRESS_SOURCE
    assert "DEFAULT_RECORD_CAMERA_NAMES" in SO101_SOURCE
    assert 'os.environ.get("SO101_RECORD_CAMERAS", "front,wrist,side")' in SO101_SOURCE
    assert "REQUIRED_RECORD_CAMERA_NAMES" in SO101_SOURCE
    assert "resolve_record_export_cameras" in SO101_SOURCE
    assert "_lerobot_dataset_cameras" in SO101_SOURCE
