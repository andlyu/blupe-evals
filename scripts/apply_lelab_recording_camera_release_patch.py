#!/usr/bin/env python3
"""Patch installed LeLab to wait longer after releasing browser camera streams.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_recording_camera_release_patch.py

On macOS, refreshing the LeLab page can reopen browser getUserMedia previews
right before recording starts. The frontend already pauses those streams before
calling `/start-recording`, but a 500 ms wait is too short for AVFoundation to
release several USB cameras reliably. This patch increases the frontend wait to
3 seconds and cache-busts the production bundle.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


DIST_ASSET_SUFFIX = "-blupe-camrelease1.js"
DIST_ASSET_QUERY = "blupe_camrelease=1"

SOURCE_OLD = """      releaseStreamsRef.current();
      await new Promise((resolve) => setTimeout(resolve, 500));
      console.log("✅ Camera streams released, proceeding with recording...");
"""
SOURCE_NEW = """      releaseStreamsRef.current();
      await new Promise((resolve) => setTimeout(resolve, 3000));
      console.log("✅ Camera streams released, proceeding with recording...");
"""

BUNDLE_OLD = (
    "k.current(),await new Promise(e=>setTimeout(e,500)),console.log("
    "`✅ Camera streams released, proceeding with recording...`)"
)
BUNDLE_NEW = (
    "k.current(),await new Promise(e=>setTimeout(e,3000)),console.log("
    "`✅ Camera streams released, proceeding with recording...`)"
)


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_source_text(text: str) -> tuple[str, bool]:
    if "setTimeout(resolve, 3000)" in text:
        return text, False
    if SOURCE_OLD not in text:
        raise ValueError("Could not find LeLab recording stream-release source block")
    return text.replace(SOURCE_OLD, SOURCE_NEW, 1), True


def patch_dist_bundle_text(text: str) -> tuple[str, bool]:
    if "setTimeout(e,3000)" in text:
        return text, False
    if BUNDLE_OLD not in text:
        raise ValueError("Could not find LeLab recording stream-release bundle block")
    return text.replace(BUNDLE_OLD, BUNDLE_NEW, 1), True


def _patch_file(path: Path, patcher) -> bool:
    text = path.read_text()
    patched, changed = patcher(text)
    if changed:
        _backup(path)
        path.write_text(patched)
    return changed


def _patch_dist_index(dist: Path) -> list[str]:
    index_path = dist / "index.html"
    if not index_path.exists():
        return []

    text = index_path.read_text()
    match = re.search(r'src="/assets/([^"]+?\.js)(?:\?[^"]*)?"', text)
    if not match:
        raise SystemExit(f"Could not find JS asset script in {index_path}")

    source_name = match.group(1)
    source_path = dist / "assets" / source_name
    if not source_path.exists():
        raise SystemExit(f"Could not find JS asset {source_path}")

    if source_name.endswith(DIST_ASSET_SUFFIX):
        target_name = source_name
        target_path = source_path
    else:
        target_name = source_name.removesuffix(".js") + DIST_ASSET_SUFFIX
        target_path = dist / "assets" / target_name

    changed: list[str] = []
    if not target_path.exists() or target_path.read_text() != source_path.read_text():
        shutil.copy2(source_path, target_path)
        changed.append(str(target_path))

    old = match.group(0)
    new = f'src="/assets/{target_name}?{DIST_ASSET_QUERY}"'
    if old != new:
        _backup(index_path)
        index_path.write_text(text.replace(old, new, 1))
        changed.append(str(index_path))

    return changed


def main() -> int:
    try:
        import lelab
    except ImportError as exc:
        raise SystemExit("Could not import lelab. Activate the LeLab Python environment first.") from exc

    package_root = Path(lelab.__file__).resolve().parents[1]
    landing_path = package_root / "frontend" / "src" / "pages" / "Landing.tsx"
    dist = package_root / "frontend" / "dist"
    dist_assets = dist / "assets"

    changed: list[str] = []
    if landing_path.exists() and _patch_file(landing_path, patch_source_text):
        changed.append(str(landing_path))

    if dist_assets.exists():
        for path in sorted(dist_assets.glob("*.js")):
            try:
                if _patch_file(path, patch_dist_bundle_text):
                    changed.append(str(path))
            except ValueError:
                continue
        changed.extend(_patch_dist_index(dist))

    if changed:
        print("Patched LeLab recording camera release wait:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Already patched: LeLab recording camera release wait")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
