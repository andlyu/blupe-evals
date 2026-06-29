#!/usr/bin/env python3
"""Patch installed LeLab to fall back to browser camera order for previews.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_camera_browser_order_fallback_patch.py

Mac-local LeLab needs browser `MediaDeviceInfo.deviceId` values for preview.
Those IDs are only available in the browser. When browser labels are unavailable
or don't match AVFoundation names, seeded cameras can remain stuck at
"No browser match" even though their OpenCV indices are valid. This patch keeps
the label matcher, then falls back to browser video input order by camera index.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


DIST_ASSET_SUFFIX = "-blupe-camorder1.js"
DIST_ASSET_QUERY = "blupe_camorder=1"

SOURCE_OLD = """      // Browser's MediaDeviceInfo.label starts with AVFoundation's localizedName
      // but Chrome often appends "(vendorId:productId)". Match by exact, then
      // prefix, then either-contains.
      const used = new Set<string>();
      const merged: AvailableCamera[] = backendCams.map((cam) => {
        const label = cam.name || `Camera ${cam.index}`;
        const target = norm(label);
        const candidates = browserDevices.filter(
          (d) => !used.has(d.deviceId) && d.label
        );
        const match =
          candidates.find((d) => norm(d.label) === target) ||
          candidates.find((d) => norm(d.label).startsWith(target)) ||
          candidates.find(
            (d) => norm(d.label).includes(target) || target.includes(norm(d.label))
          );
        if (match) used.add(match.deviceId);
        return {
          index: cam.index,
          name: label,
          deviceId: match?.deviceId ?? "",
          available: cam.available,
        };
      });
"""

SOURCE_NEW = """      // Browser's MediaDeviceInfo.label starts with AVFoundation's localizedName
      // but Chrome often appends "(vendorId:productId)". Match by exact, then
      // prefix, then either-contains. If labels are unavailable, fall back to
      // browser video input order, which is the best available browser-side
      // proxy for the saved OpenCV camera index.
      const used = new Set<string>();
      const merged: AvailableCamera[] = backendCams.map((cam) => {
        const label = cam.name || `Camera ${cam.index}`;
        const target = norm(label);
        const candidates = browserDevices.filter(
          (d) => !used.has(d.deviceId) && d.label
        );
        const ordinalMatch = browserDevices[cam.index];
        const match =
          candidates.find((d) => norm(d.label) === target) ||
          candidates.find((d) => norm(d.label).startsWith(target)) ||
          candidates.find(
            (d) => norm(d.label).includes(target) || target.includes(norm(d.label))
          ) ||
          (ordinalMatch && !used.has(ordinalMatch.deviceId) ? ordinalMatch : undefined);
        if (match) used.add(match.deviceId);
        return {
          index: cam.index,
          name: label,
          deviceId: match?.deviceId ?? "",
          available: cam.available,
        };
      });
"""

BUNDLE_OLD = (
    "let a=(await r.json()).cameras??[],o=new Set,s=a.map(t=>{let n=t.name||`Camera ${t.index}`,"
    "r=K_(n),i=e.filter(e=>!o.has(e.deviceId)&&e.label),a=i.find(e=>K_(e.label)===r)||"
    "i.find(e=>K_(e.label).startsWith(r))||i.find(e=>K_(e.label).includes(r)||r.includes(K_(e.label)));"
    "return a&&o.add(a.deviceId),{index:t.index,name:n,deviceId:a?.deviceId??``,available:t.available}});"
)

BUNDLE_NEW = (
    "let a=(await r.json()).cameras??[],o=new Set,s=a.map(t=>{let n=t.name||`Camera ${t.index}`,"
    "r=K_(n),i=e.filter(e=>!o.has(e.deviceId)&&e.label),l=e[t.index],a=i.find(e=>K_(e.label)===r)||"
    "i.find(e=>K_(e.label).startsWith(r))||i.find(e=>K_(e.label).includes(r)||r.includes(K_(e.label)))||"
    "(l&&!o.has(l.deviceId)?l:void 0);return a&&o.add(a.deviceId),{index:t.index,name:n,"
    "deviceId:a?.deviceId??``,available:t.available}});"
)


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_source_text(text: str) -> tuple[str, bool]:
    if "ordinalMatch" in text:
        return text, False
    if SOURCE_OLD not in text:
        raise ValueError("Could not find LeLab available camera matcher source block")
    return text.replace(SOURCE_OLD, SOURCE_NEW, 1), True


def patch_dist_bundle_text(text: str) -> tuple[str, bool]:
    if "l=e[t.index],a=i.find" in text:
        return text, False
    if BUNDLE_OLD not in text:
        raise ValueError("Could not find LeLab available camera matcher bundle block")
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
    hook_path = package_root / "frontend" / "src" / "hooks" / "useAvailableCameras.ts"
    dist = package_root / "frontend" / "dist"
    dist_assets = dist / "assets"

    changed: list[str] = []
    if hook_path.exists() and _patch_file(hook_path, patch_source_text):
        changed.append(str(hook_path))

    if dist_assets.exists():
        for path in sorted(dist_assets.glob("*.js")):
            try:
                if _patch_file(path, patch_dist_bundle_text):
                    changed.append(str(path))
            except ValueError:
                continue
        changed.extend(_patch_dist_index(dist))

    if changed:
        print("Patched LeLab camera browser order fallback:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Already patched: LeLab camera browser order fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
