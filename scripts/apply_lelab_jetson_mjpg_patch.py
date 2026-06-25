#!/usr/bin/env python3
"""Patch an installed LeLab so Jetson 3-camera recordings default to MJPG.

Run this inside the Python environment that provides the `lelab` package:

    python scripts/apply_lelab_jetson_mjpg_patch.py

The patch is intentionally small and idempotent. It modifies LeLab's installed
`record.py` in place and writes a `.blupe-backup` file the first time it runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path


OLD_BLOCK = """        backend_name = camera_data.get("backend")
        backend = Cv2Backends[backend_name] if backend_name else default_backend
        fourcc = camera_data.get("fourcc") or None

        camera_configs[camera_name] = OpenCVCameraConfig(
"""

NEW_BLOCK = """        backend_name = camera_data.get("backend")
        backend = Cv2Backends[backend_name] if backend_name else default_backend
        fourcc = camera_data.get("fourcc") or None
        if fourcc is None and backend == Cv2Backends.V4L2 and len(cameras) >= 3:
            # Three USB cameras on the Jetson need compressed capture; the UI
            # currently omits the saved robot camera fourcc from recording requests.
            fourcc = "MJPG"

        camera_configs[camera_name] = OpenCVCameraConfig(
"""


def main() -> int:
    try:
        import lelab.record
    except ImportError as exc:
        raise SystemExit(
            "Could not import lelab.record. Activate the LeLab Python environment first."
        ) from exc

    record_path = Path(lelab.record.__file__)
    text = record_path.read_text()

    if NEW_BLOCK in text:
        print(f"Already patched: {record_path}")
        return 0

    if OLD_BLOCK not in text:
        raise SystemExit(
            f"Could not find the expected camera config block in {record_path}. "
            "LeLab may have changed; inspect patches/lelab/jetson-3cam-mjpg.patch."
        )

    backup_path = record_path.with_suffix(record_path.suffix + ".blupe-backup")
    if not backup_path.exists():
        shutil.copy2(record_path, backup_path)

    record_path.write_text(text.replace(OLD_BLOCK, NEW_BLOCK))
    print(f"Patched: {record_path}")
    print(f"Backup:  {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
