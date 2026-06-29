#!/usr/bin/env python3
"""Patch installed LeRobot OpenCV camera reads to tolerate transient misses.

Run inside the Python environment that provides `lerobot`:

    python scripts/apply_lerobot_opencv_transient_read_tolerance_patch.py

The stock OpenCV background read loop aborts after a fixed count of consecutive
read failures. Under multi-camera USB load that count can be consumed in a
fraction of a second, killing an otherwise healthy recording. This patch makes
the policy time based: tolerate transient misses for a few seconds, but still
raise if a camera keeps failing.
"""

from __future__ import annotations

import shutil
from pathlib import Path


OLD_BLOCK = """        failure_count = 0
        while not self.stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e
"""

NEW_BLOCK = """        failure_count = 0
        failure_start_time = None
        max_failure_duration_s = 5.0
        while not self.stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0
                failure_start_time = None

            except DeviceNotConnectedError:
                break
            except Exception as e:
                now = time.perf_counter()
                if failure_start_time is None:
                    failure_start_time = now
                failure_count += 1
                if failure_count <= 10 or failure_count % 30 == 0:
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                if now - failure_start_time > max_failure_duration_s:
                    raise RuntimeError(
                        f"{self} exceeded maximum read failure duration "
                        f"({max_failure_duration_s:.1f}s, {failure_count} consecutive failures)."
                    ) from e
                time.sleep(0.02)
"""


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_text(text: str) -> tuple[str, bool]:
    if "max_failure_duration_s = 5.0" in text:
        return text, False
    if OLD_BLOCK not in text:
        raise ValueError("Could not find LeRobot OpenCV read-loop failure block")
    return text.replace(OLD_BLOCK, NEW_BLOCK, 1), True


def main() -> int:
    try:
        import lerobot
    except ImportError as exc:
        raise SystemExit("Could not import lerobot. Activate the LeLab Python environment first.") from exc

    package_root = Path(lerobot.__file__).resolve().parent
    target = package_root / "cameras" / "opencv" / "camera_opencv.py"
    if not target.exists():
        raise SystemExit(f"Could not find {target}")

    patched, changed = patch_text(target.read_text())
    if changed:
        _backup(target)
        target.write_text(patched)
        print(f"Patched {target}")
    else:
        print("Already patched: LeRobot OpenCV transient read tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
