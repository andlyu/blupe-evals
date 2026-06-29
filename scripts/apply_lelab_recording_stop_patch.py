#!/usr/bin/env python3
"""Patch installed LeLab recording UI with an explicit stop-collection button.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_recording_stop_patch.py

LeLab already exposes `/stop-recording` and the recording page has a hidden
three-dot menu item plus Escape shortcut. This patch makes that stop action
visible during collection so operators can end a multi-episode run without
knowing the hidden menu.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


DIST_ASSET_SUFFIX = "-blupe-stop1.js"
DIST_ASSET_QUERY = "blupe_stop=1"

SOURCE_PRIMARY_BUTTON = """          <Button
            onClick={handleExitEarly}
            disabled={
              !backendStatus.available_controls.exit_early ||
              optimisticPhase !== null ||
              currentPhase === "completed"
            }
            className={`w-full text-white font-semibold py-6 text-lg disabled:opacity-50 ${phaseColor.button}`}
          >
            <PrimaryIcon className="w-5 h-5 mr-2" />
            {primaryLabel}
            {currentPhase !== "completed" && (
              <span className="ml-3 px-2 py-0.5 rounded text-xs font-mono bg-black/30 text-white/70">SPACE / →</span>
            )}
          </Button>
"""

SOURCE_VISIBLE_STOP_BUTTON = """
          {currentPhase !== "completed" && (
            <Button
              onClick={requestStopRecording}
              disabled={!backendStatus.available_controls.stop_recording}
              variant="outline"
              className="w-full mt-3 border-red-500/60 text-red-300 hover:bg-red-500/10 hover:text-red-200 disabled:opacity-50"
            >
              <Square className="w-4 h-4 mr-2" />
              Stop Collection
              <span className="ml-3 px-2 py-0.5 rounded text-xs font-mono bg-red-500/20 text-red-100/80">ESC</span>
            </Button>
          )}
"""

BUNDLE_PRIMARY_BUTTON = (
    "(0,V.jsxs)(G,{onClick:O,disabled:!s.available_controls.exit_early||d!==null||I===`completed`,"
    "className:`w-full text-white font-semibold py-6 text-lg disabled:opacity-50 ${oe.button}`,"
    "children:[(0,V.jsx)(ce,{className:`w-5 h-5 mr-2`}),se,I!==`completed`&&(0,V.jsx)(`span`,"
    "{className:`ml-3 px-2 py-0.5 rounded text-xs font-mono bg-black/30 text-white/70`,children:`SPACE / →`})]})"
)

BUNDLE_VISIBLE_STOP_BUTTON = (
    ",I!==`completed`&&(0,V.jsxs)(G,{onClick:j,disabled:!s.available_controls.stop_recording,"
    "variant:`outline`,className:`w-full mt-3 border-red-500/60 text-red-300 hover:bg-red-500/10 "
    "hover:text-red-200 disabled:opacity-50`,children:[(0,V.jsx)(ao,{className:`w-4 h-4 mr-2`}),"
    "`Stop Collection`,(0,V.jsx)(`span`,{className:`ml-3 px-2 py-0.5 rounded text-xs font-mono "
    "bg-red-500/20 text-red-100/80`,children:`ESC`})]})"
)


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_recording_source(text: str) -> tuple[str, bool]:
    if "Stop Collection" in text:
        return text, False
    if SOURCE_PRIMARY_BUTTON not in text:
        raise ValueError("Could not find LeLab recording primary control block")
    return text.replace(SOURCE_PRIMARY_BUTTON, SOURCE_PRIMARY_BUTTON + SOURCE_VISIBLE_STOP_BUTTON, 1), True


def patch_dist_bundle_text(text: str) -> tuple[str, bool]:
    if "Stop Collection" in text:
        return text, False
    if BUNDLE_PRIMARY_BUTTON not in text:
        raise ValueError("Could not find LeLab recording primary control bundle block")
    return text.replace(BUNDLE_PRIMARY_BUTTON, BUNDLE_PRIMARY_BUTTON + BUNDLE_VISIBLE_STOP_BUTTON, 1), True


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
    recording_path = package_root / "frontend" / "src" / "pages" / "Recording.tsx"
    dist = package_root / "frontend" / "dist"
    dist_assets = dist / "assets"

    changed: list[str] = []
    if recording_path.exists() and _patch_file(recording_path, patch_recording_source):
        changed.append(str(recording_path))

    if dist_assets.exists():
        for path in sorted(dist_assets.glob("*.js")):
            try:
                if _patch_file(path, patch_dist_bundle_text):
                    changed.append(str(path))
            except ValueError:
                continue
        changed.extend(_patch_dist_index(dist))

    if changed:
        print("Patched LeLab recording stop UI:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Already patched: LeLab recording stop UI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
