#!/usr/bin/env python3
"""Patch an installed LeLab for BluPe SO101 dataset recording.

Run this inside the Python environment that provides the `lelab` package:

    python scripts/apply_lelab_jetson_mjpg_patch.py

The patches are intentionally small and idempotent. They modify LeLab's
installed `record.py` in place and write a `.blupe-backup` file the first time
the script changes that file.
"""

from __future__ import annotations

import shutil
from pathlib import Path


MJPG_OLD_BLOCK = """        backend_name = camera_data.get("backend")
        backend = Cv2Backends[backend_name] if backend_name else default_backend
        fourcc = camera_data.get("fourcc") or None

        camera_configs[camera_name] = OpenCVCameraConfig(
"""

MJPG_NEW_BLOCK = """        backend_name = camera_data.get("backend")
        backend = Cv2Backends[backend_name] if backend_name else default_backend
        fourcc = camera_data.get("fourcc") or None
        if fourcc is None and backend == Cv2Backends.V4L2 and len(cameras) >= 3:
            # Three USB cameras on the Jetson need compressed capture; the UI
            # currently omits the saved robot camera fourcc from recording requests.
            fourcc = "MJPG"

        camera_configs[camera_name] = OpenCVCameraConfig(
"""

TIMEOUT_OLD_BLOCK = """            else:
                # Recording completed due to timeout - trigger re-record behavior
                logger.info("⏰ RECORDING PHASE COMPLETED DUE TO TIMEOUT - triggering re-record")
                print(
                    f"⏰ STATUS CHANGE: Recording timeout reached for episode {current_episode} - re-recording"
                )
                web_events["rerecord_episode"] = True
"""

TIMEOUT_NEW_BLOCK = """            else:
                # Recording completed because the requested episode duration elapsed.
                # That is a successful episode; only an explicit rerecord event should
                # discard the buffer and repeat the same episode.
                logger.info("⏰ RECORDING PHASE COMPLETED DUE TO TIMEOUT - proceeding to save episode")
                print(
                    f"⏰ STATUS CHANGE: Recording timeout reached for episode {current_episode} - saving episode"
                )
"""

PATCHES = (
    ("jetson 3-camera MJPG default", MJPG_OLD_BLOCK, MJPG_NEW_BLOCK),
    ("save episode on normal timeout", TIMEOUT_OLD_BLOCK, TIMEOUT_NEW_BLOCK),
)


def patch_text(text: str, record_path: Path | str = "lelab/record.py") -> tuple[str, list[str]]:
    applied: list[str] = []
    for name, old, new in PATCHES:
        if new in text:
            continue
        if old not in text:
            raise ValueError(
                f"Could not find the expected block for '{name}' in {record_path}. "
                "LeLab may have changed; inspect patches/lelab/."
            )
        text = text.replace(old, new)
        applied.append(name)
    return text, applied


def main() -> int:
    try:
        import lelab.record
    except ImportError as exc:
        raise SystemExit(
            "Could not import lelab.record. Activate the LeLab Python environment first."
        ) from exc

    record_path = Path(lelab.record.__file__)
    text = record_path.read_text()

    patched_text, applied = patch_text(text, record_path)

    if not applied:
        print(f"Already patched: {record_path}")
        return 0

    backup_path = record_path.with_suffix(record_path.suffix + ".blupe-backup")
    if not backup_path.exists():
        shutil.copy2(record_path, backup_path)

    record_path.write_text(patched_text)
    print(f"Patched: {record_path}")
    for name in applied:
        print(f"Applied: {name}")
    print(f"Backup:  {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
