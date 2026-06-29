#!/usr/bin/env python3
"""Patch installed LeLab with Jetson recording capture limit env controls.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_recording_capture_limits_patch.py

The SO101 Jetson can preview all three cameras, but recording three 640x360
camera streams at 30 FPS while controlling the arm can stall before frame 1.
This patch keeps all configured cameras and lets the launch environment reduce
recording loop pressure:

    LELAB_RECORD_MAX_FPS=15
    LELAB_RECORD_STREAMING_ENCODING=0

On macOS, AVFoundation devices can reject non-native FPS values. Camera FPS is
therefore only capped when `LELAB_RECORD_CAP_CAMERA_FPS=1` is set.
"""

from __future__ import annotations

import shutil
from pathlib import Path


IMPORT_OLD = "import logging\n"
IMPORT_NEW = "import logging\nimport os\n"

HELPER_MARKER = "\ndef create_record_config(request: RecordingRequest) -> RecordConfig:\n"
HELPER_BLOCK = r'''

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _apply_recording_capture_limits(request: RecordingRequest) -> RecordingRequest:
    """Apply Jetson recording limits without changing the saved robot config."""
    max_fps_text = os.environ.get("LELAB_RECORD_MAX_FPS", "").strip()
    cap_camera_fps = _env_bool("LELAB_RECORD_CAP_CAMERA_FPS", False)
    if max_fps_text:
        try:
            max_fps = int(max_fps_text)
        except ValueError:
            logger.warning("Ignoring invalid LELAB_RECORD_MAX_FPS=%r", max_fps_text)
        else:
            if max_fps > 0:
                if request.fps > max_fps:
                    logger.info("Capping recording dataset FPS from %s to %s", request.fps, max_fps)
                    request.fps = max_fps
                if cap_camera_fps:
                    for camera_name, camera_data in request.cameras.items():
                        if not isinstance(camera_data, dict):
                            continue
                        camera_fps = camera_data.get("fps")
                        if camera_fps is None or int(camera_fps) > max_fps:
                            logger.info(
                                "Capping recording camera %s FPS from %s to %s",
                                camera_name,
                                camera_fps,
                                max_fps,
                            )
                            camera_data["fps"] = max_fps

    original_streaming = request.streaming_encoding
    request.streaming_encoding = _env_bool("LELAB_RECORD_STREAMING_ENCODING", request.streaming_encoding)
    if request.streaming_encoding != original_streaming:
        logger.info(
            "Overriding streaming encoding from %s to %s via LELAB_RECORD_STREAMING_ENCODING",
            original_streaming,
            request.streaming_encoding,
        )

    return request
'''

CREATE_CONFIG_OLD = """def create_record_config(request: RecordingRequest) -> RecordConfig:
    \"\"\"Create a RecordConfig from the recording request\"\"\"
    # Setup calibration files
"""
CREATE_CONFIG_NEW = """def create_record_config(request: RecordingRequest) -> RecordConfig:
    \"\"\"Create a RecordConfig from the recording request\"\"\"
    request = _apply_recording_capture_limits(request)

    # Setup calibration files
"""


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_text(text: str) -> tuple[str, bool]:
    changed = False

    if "import os\n" not in text:
        if IMPORT_OLD not in text:
            raise ValueError("Could not find logging import in lelab.record")
        text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
        changed = True

    if "def _apply_recording_capture_limits(request: RecordingRequest)" not in text:
        if HELPER_MARKER not in text:
            raise ValueError("Could not find create_record_config marker in lelab.record")
        text = text.replace(HELPER_MARKER, HELPER_BLOCK + HELPER_MARKER, 1)
        changed = True
    elif "LELAB_RECORD_CAP_CAMERA_FPS" not in text:
        old_start = text.index("def _env_bool(name: str, default: bool) -> bool:")
        old_end = text.index(HELPER_MARKER)
        text = text[:old_start] + HELPER_BLOCK.strip("\n") + "\n" + text[old_end:]
        changed = True

    if "request = _apply_recording_capture_limits(request)" in text:
        return text, changed
    if CREATE_CONFIG_OLD not in text:
        raise ValueError("Could not find create_record_config body in lelab.record")
    text = text.replace(CREATE_CONFIG_OLD, CREATE_CONFIG_NEW, 1)
    changed = True
    return text, changed


def main() -> int:
    try:
        import lelab.record
    except ImportError as exc:
        raise SystemExit("Could not import lelab.record. Activate the LeLab Python environment first.") from exc

    record_path = Path(lelab.record.__file__)
    text = record_path.read_text()
    patched, changed = patch_text(text)
    if not changed:
        print(f"Already patched: {record_path}")
        return 0

    _backup(record_path)
    record_path.write_text(patched)
    print(f"Patched LeLab recording capture limits: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
