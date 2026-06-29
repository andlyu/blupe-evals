#!/usr/bin/env python3
"""Patch installed LeLab recording for Jetson stability and clearer failures.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_recording_stability_patch.py

The SO101 Jetson path records three cameras while controlling the arm at 30 Hz.
LeRobot's default streaming video encoder uses libsvtav1 with automatic thread
parallelism, which can starve the control loop on the Jetson. This patch keeps
streaming encoding enabled but caps encoder threads and increases queue depth.
It also exposes the real traceback in logs and `/recording-status` when a run
fails, since the default LeLab log line only says "Recording session failed".
"""

from __future__ import annotations

import shutil
from pathlib import Path


IMPORT_OLD = "import time\nfrom datetime import datetime\n"
IMPORT_NEW = "import time\nimport traceback\nfrom datetime import datetime\n"

DATASET_CONFIG_OLD = """        streaming_encoding=request.streaming_encoding,
    )
"""
DATASET_CONFIG_NEW = """        streaming_encoding=request.streaming_encoding,
        # Jetson SO101: three 640x360 streams plus robot control can starve when
        # SVT-AV1 uses its default parallelism. `auto` keeps hardware encoding
        # available when PyAV exposes it, and `encoder_threads=2` bounds the
        # software fallback.
        vcodec="auto",
        encoder_threads=2,
        encoder_queue_maxsize=180,
    )
"""

WORKER_LOG_OLD = """                logger.info(
                    "Recording session started: dataset=%s task=%r episodes=%d",
                    request.dataset_repo_id,
                    request.single_task,
                    request.num_episodes,
                )
"""
WORKER_LOG_NEW = """                logger.info(
                    "Recording session started: dataset=%s task=%r episodes=%d",
                    request.dataset_repo_id,
                    request.single_task,
                    request.num_episodes,
                )
                logger.info(
                    "Recording encoder settings: streaming=%s vcodec=%s encoder_threads=%s "
                    "encoder_queue_maxsize=%s fps=%s cameras=%s",
                    record_config.dataset.streaming_encoding,
                    record_config.dataset.vcodec,
                    record_config.dataset.encoder_threads,
                    record_config.dataset.encoder_queue_maxsize,
                    record_config.dataset.fps,
                    list(request.cameras.keys()),
                )
"""

EXCEPT_OLD = """            except Exception as e:
                logger.exception("Recording session failed")
                current_phase = "error"
                if recording_start_time:
                    session_end_elapsed_seconds = int(time.time() - recording_start_time)
                last_recording_info = {"success": False, "error": str(e)}
"""
EXCEPT_NEW = """            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.exception("Recording session failed")
                logger.error(
                    "Recording exception detail: %s: %s\\n%s",
                    type(e).__name__,
                    e,
                    error_traceback,
                )
                print(f"❌ RECORDING ERROR: {type(e).__name__}: {e}")
                current_phase = "error"
                if recording_start_time:
                    session_end_elapsed_seconds = int(time.time() - recording_start_time)
                last_recording_info = {
                    "success": False,
                    "dataset_repo_id": request.dataset_repo_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_traceback,
                }
"""

STATUS_OLD = """    if recording_config:
        status["dataset_repo_id"] = recording_config.dataset_repo_id

    # Add episode information if recording is active
"""
STATUS_NEW = """    if recording_config:
        status["dataset_repo_id"] = recording_config.dataset_repo_id

    if current_phase == "error" and last_recording_info and not last_recording_info.get("success"):
        status["error"] = last_recording_info.get("error", "")
        status["error_type"] = last_recording_info.get("error_type", "")
        status["traceback"] = last_recording_info.get("traceback", "")

    # Add episode information if recording is active
"""


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def _replace_required(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise ValueError(f"Could not find expected block for {label}")
    return text.replace(old, new, 1), True


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    for label, old, new in (
        ("traceback import", IMPORT_OLD, IMPORT_NEW),
        ("dataset encoder settings", DATASET_CONFIG_OLD, DATASET_CONFIG_NEW),
        ("worker encoder log", WORKER_LOG_OLD, WORKER_LOG_NEW),
        ("exception detail", EXCEPT_OLD, EXCEPT_NEW),
        ("recording status error detail", STATUS_OLD, STATUS_NEW),
    ):
        text, did_change = _replace_required(text, old, new, label)
        changed = changed or did_change
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
    print(f"Patched LeLab recording stability: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
