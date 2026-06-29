#!/usr/bin/env python3
"""Patch installed LeLab to disable the single-tab guard modal.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_disable_single_tab_guard_patch.py

LeLab's frontend uses a BroadcastChannel guard to show a full-screen
"already open in another tab" dialog. For the BluPe Jetson workflow, the UI is
often reached through tunnels, reloads, and multiple operator tabs, so the
guard blocks more than it helps. This patch turns the guard component into a
pass-through and also patches the already-built production bundle.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


DIST_ASSET_SUFFIX = "-blupe-notab1.js"
DIST_ASSET_QUERY = "blupe_notab=1"

SOURCE_IMPORT_OLD = 'import { useCallback, useEffect, useRef, useState, ReactNode } from "react";\n'
SOURCE_IMPORT_NEW = 'import { ReactNode } from "react";\n'
SOURCE_BUTTON_IMPORT = 'import { Button } from "@/components/ui/button";\n'
SOURCE_COMPONENT_PATTERN = re.compile(
    r"const SingleTabGuard = \(\{ children \}: \{ children: ReactNode \}\) => \{.*?\n"
    r"\};\n\nexport default SingleTabGuard;",
    re.DOTALL,
)
SOURCE_COMPONENT_NEW = """const SingleTabGuard = ({ children }: { children: ReactNode }) => {
  return <>{children}</>;
};

export default SingleTabGuard;"""

BUNDLE_OVERLAY_PATTERN = re.compile(
    r"(children:\[[A-Za-z_$][A-Za-z0-9_$]*),![A-Za-z_$][A-Za-z0-9_$]*"
    r"(&&\(0,[A-Za-z_$][A-Za-z0-9_$]*\.jsx\)\(`div`,\{className:`fixed inset-0 z-\[9999\])"
)
BUNDLE_DISABLED_PATTERN = re.compile(
    r"children:\[[A-Za-z_$][A-Za-z0-9_$]*,false&&\(0,[A-Za-z_$][A-Za-z0-9_$]*\.jsx\)"
    r"\(`div`,\{className:`fixed inset-0 z-\[9999\]"
)
BUNDLE_POPUP_TEXT = "LeLab is already open in another tab"


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_source_text(text: str) -> tuple[str, bool]:
    if "return <>{children}</>;" in text and BUNDLE_POPUP_TEXT not in text:
        return text, False

    if BUNDLE_POPUP_TEXT not in text:
        raise ValueError("Could not find LeLab single-tab guard popup text")

    patched = text.replace(SOURCE_IMPORT_OLD, SOURCE_IMPORT_NEW, 1)
    patched = patched.replace(SOURCE_BUTTON_IMPORT, "", 1)
    patched, count = SOURCE_COMPONENT_PATTERN.subn(SOURCE_COMPONENT_NEW, patched, count=1)
    if count != 1:
        raise ValueError("Could not find LeLab SingleTabGuard component")
    return patched, patched != text


def patch_dist_bundle_text(text: str) -> tuple[str, bool]:
    if BUNDLE_DISABLED_PATTERN.search(text):
        return text, False

    if BUNDLE_POPUP_TEXT not in text:
        raise ValueError("Could not find LeLab single-tab guard bundle text")

    patched, count = BUNDLE_OVERLAY_PATTERN.subn(r"\1,false\2", text, count=1)
    if count != 1:
        raise ValueError("Could not find LeLab single-tab guard overlay trigger")
    return patched, True


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
    source_path = package_root / "frontend" / "src" / "components" / "SingleTabGuard.tsx"
    dist = package_root / "frontend" / "dist"
    dist_assets = dist / "assets"

    changed: list[str] = []
    if source_path.exists() and _patch_file(source_path, patch_source_text):
        changed.append(str(source_path))

    if dist_assets.exists():
        for path in sorted(dist_assets.glob("*.js")):
            try:
                if _patch_file(path, patch_dist_bundle_text):
                    changed.append(str(path))
            except ValueError:
                continue
        changed.extend(_patch_dist_index(dist))

    if changed:
        print("Patched LeLab single-tab guard:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Already patched: LeLab single-tab guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
