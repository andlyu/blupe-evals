#!/usr/bin/env python3
"""Patch installed LeLab to repair seeded camera browser device IDs.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_camera_device_binding_patch.py

LeLab's recording UI stores two camera identifiers:

- `camera_index`: the OpenCV index used by the recorder.
- `device_id`: the browser's opaque MediaDeviceInfo deviceId used for preview.

When a robot record is seeded outside the browser, we can know the OpenCV
indices but not the browser device IDs. This patch lets the recording UI repair
saved cameras by matching their `camera_index` to `/available-cameras`, while
still preserving the existing device-id-to-index refresh for USB reorderings.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


DIST_ASSET_SUFFIX = "-blupe-camdev1.js"
DIST_ASSET_QUERY = "blupe_camdev=1"

SOURCE_OLD = """  // cv2's AVFoundation order is uniqueID-sorted, so plugging/unplugging a
  // device between sessions shifts indices. The browser device_id stays
  // stable per-origin, so use it to refresh each seeded camera's
  // camera_index — otherwise the recorder opens the wrong physical device
  // and the dropdown's "already added" check guards a stale index.
  useEffect(() => {
    if (availableCameras.length === 0 || cameras.length === 0) return;
    let changed = false;
    const refreshed = cameras.map((cam) => {
      if (!cam.device_id) return cam;
      const match = availableCameras.find((m) => m.deviceId === cam.device_id);
      if (match && match.index !== cam.camera_index) {
        changed = true;
        return { ...cam, camera_index: match.index };
      }
      return cam;
    });
    if (changed) onCamerasChange(refreshed);
    // We deliberately don't depend on `cameras`/`onCamerasChange` to avoid
    // re-running every keystroke in the camera-name input — re-syncing only
    // when the available-cameras list itself changes is sufficient.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableCameras]);
"""

SOURCE_NEW = """  // cv2's AVFoundation order is uniqueID-sorted, so plugging/unplugging a
  // device between sessions shifts indices. The browser device_id stays
  // stable per-origin, so use it to refresh each seeded camera's
  // camera_index. Robot records seeded outside the browser cannot know that
  // opaque device_id, so also bind by saved camera_index when needed.
  useEffect(() => {
    if (availableCameras.length === 0 || cameras.length === 0) return;
    let changed = false;
    const refreshed = cameras.map((cam) => {
      const matchByDeviceId = cam.device_id
        ? availableCameras.find((m) => m.deviceId === cam.device_id)
        : undefined;
      const matchByIndex =
        cam.camera_index !== undefined
          ? availableCameras.find((m) => m.index === cam.camera_index)
          : undefined;
      const match = matchByDeviceId ?? matchByIndex;
      if (!match) return cam;

      const updates: Partial<CameraConfig> = {};
      if (match.deviceId && cam.device_id !== match.deviceId) {
        updates.device_id = match.deviceId;
      }
      if (cam.camera_index !== match.index) {
        updates.camera_index = match.index;
      }
      if (Object.keys(updates).length === 0) return cam;
      changed = true;
      return { ...cam, ...updates };
    });
    if (changed) onCamerasChange(refreshed);
    // We deliberately don't depend on `cameras`/`onCamerasChange` to avoid
    // re-running every keystroke in the camera-name input — re-syncing only
    // when the available-cameras list itself changes is sufficient.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableCameras]);
"""

BUNDLE_OLD = (
    "(0,_.useEffect)(()=>{if(i.length===0||e.length===0)return;let n=!1,r=e.map(e=>{"
    "if(!e.device_id)return e;let t=i.find(t=>t.deviceId===e.device_id);return "
    "t&&t.index!==e.camera_index?(n=!0,{...e,camera_index:t.index}):e});n&&t(r)},[i]);"
)

BUNDLE_NEW = (
    "(0,_.useEffect)(()=>{if(i.length===0||e.length===0)return;let n=!1,r=e.map(e=>{"
    "let t=e.device_id?i.find(t=>t.deviceId===e.device_id):void 0,"
    "r=e.camera_index!==void 0?i.find(t=>t.index===e.camera_index):void 0,a=t??r;"
    "if(!a)return e;let o={},s=!1;return a.deviceId&&e.device_id!==a.deviceId&&"
    "(o.device_id=a.deviceId,s=!0),e.camera_index!==a.index&&(o.camera_index=a.index,s=!0),"
    "s&&(n=!0),s?{...e,...o}:e});n&&t(r)},[i]);"
)


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_source_text(text: str) -> tuple[str, bool]:
    if "matchByDeviceId" in text and "matchByIndex" in text:
        return text, False
    if SOURCE_OLD not in text:
        raise ValueError("Could not find LeLab camera device binding source block")
    return text.replace(SOURCE_OLD, SOURCE_NEW, 1), True


def patch_dist_bundle_text(text: str) -> tuple[str, bool]:
    if "a=t??r;if(!a)return e;let o={},s=!1" in text:
        return text, False
    if BUNDLE_OLD not in text:
        raise ValueError("Could not find LeLab camera device binding bundle block")
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
    camera_config_path = package_root / "frontend" / "src" / "components" / "recording" / "CameraConfiguration.tsx"
    dist = package_root / "frontend" / "dist"
    dist_assets = dist / "assets"

    changed: list[str] = []
    if camera_config_path.exists() and _patch_file(camera_config_path, patch_source_text):
        changed.append(str(camera_config_path))

    if dist_assets.exists():
        for path in sorted(dist_assets.glob("*.js")):
            try:
                if _patch_file(path, patch_dist_bundle_text):
                    changed.append(str(path))
            except ValueError:
                continue
        changed.extend(_patch_dist_index(dist))

    if changed:
        print("Patched LeLab camera device binding:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Already patched: LeLab camera device binding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
