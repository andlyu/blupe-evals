#!/usr/bin/env python3
"""Patch installed LeLab to expose a BluPe Evals entry point.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_blupe_tab_patch.py

The patch adds a landing-page card that opens the local blupe-evals dashboard.
It updates both the frontend source and the installed production bundle when
those files are present.
"""

from __future__ import annotations

import shutil
from pathlib import Path


DEFAULT_BLUPE_EVALS_URL = "http://127.0.0.1:8099/dashboard"

SOURCE_HANDLER = (
    "  const handleBlupeEvalsClick = () => "
    f'window.location.assign(import.meta.env.VITE_BLUPE_EVALS_URL || "{DEFAULT_BLUPE_EVALS_URL}");\n'
)

SOURCE_CARD = """            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
              <h3 className="font-semibold text-lg text-left h-10 flex items-center">
                BluPe Evals
              </h3>
              <Button
                onClick={handleBlupeEvalsClick}
                className="w-full bg-cyan-500 hover:bg-cyan-600 text-white"
              >
                Open Evals
              </Button>
            </div>
"""

SOURCE_TRAINING_CARD = """            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
              <h3 className="font-semibold text-lg text-left h-10 flex items-center">
                Create a model
              </h3>
              <Button
                onClick={handleTrainingClick}
                className="w-full bg-green-500 hover:bg-green-600 text-white"
              >
                Training
              </Button>
            </div>
"""

DIST_HANDLER_NEEDLE = f"window.location.assign(`{DEFAULT_BLUPE_EVALS_URL}`)"


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_landing_source(text: str) -> tuple[str, bool]:
    changed = False

    if "handleBlupeEvalsClick" not in text:
        if "const handleEditDatasetClick" in text:
            marker = "\n  const handleEditDatasetClick"
            idx = text.find(marker)
            end = text.find("\n", idx + 1) + 1
            text = text[:end] + SOURCE_HANDLER + text[end:]
        else:
            marker = '  const handleTrainingClick = () => navigate("/training");\n'
            if marker not in text:
                raise ValueError("Could not find LeLab landing training handler")
            text = text.replace(marker, marker + SOURCE_HANDLER)
        changed = True

    for old in (
        "grid grid-cols-1 md:grid-cols-2 gap-3",
        "grid grid-cols-1 md:grid-cols-3 gap-3",
    ):
        if old in text:
            text = text.replace(old, "grid grid-cols-1 md:grid-cols-4 gap-3")
            changed = True

    if "BluPe Evals" not in text:
        if SOURCE_TRAINING_CARD not in text:
            raise ValueError("Could not find LeLab landing training card")
        text = text.replace(SOURCE_TRAINING_CARD, SOURCE_TRAINING_CARD + SOURCE_CARD)
        changed = True

    return text, changed


def patch_dist_bundle_text(text: str) -> tuple[str, bool]:
    changed = False

    if DIST_HANDLER_NEEDLE not in text:
        marker = f"window.location.assign(`{DEFAULT_BLUPE_EVALS_URL}`)"
        if marker not in text:
            handler_anchor = "F=()=>window.location.assign(`http://127.0.0.1:8092/episodes`)"
            if handler_anchor in text:
                text = text.replace(
                    handler_anchor,
                    handler_anchor + f",B=()=>window.location.assign(`{DEFAULT_BLUPE_EVALS_URL}`)",
                )
                changed = True

    if "grid grid-cols-1 md:grid-cols-3 gap-3" in text:
        text = text.replace(
            "grid grid-cols-1 md:grid-cols-3 gap-3",
            "grid grid-cols-1 md:grid-cols-4 gap-3",
        )
        changed = True
    elif "grid grid-cols-1 md:grid-cols-2 gap-3" in text:
        text = text.replace(
            "grid grid-cols-1 md:grid-cols-2 gap-3",
            "grid grid-cols-1 md:grid-cols-4 gap-3",
        )
        changed = True

    if "BluPe Evals" not in text:
        training_card = (
            "(0,V.jsxs)(`div`,{className:`bg-gray-800 rounded-lg border border-gray-700 p-3 "
            "flex flex-col gap-2`,children:[(0,V.jsx)(`h3`,{className:`font-semibold text-lg "
            "text-left h-10 flex items-center`,children:`Create a model`}),(0,V.jsx)(G,{onClick:P,"
            "className:`w-full bg-green-500 hover:bg-green-600 text-white`,children:`Training`})]})"
        )
        eval_card = (
            "(0,V.jsxs)(`div`,{className:`bg-gray-800 rounded-lg border border-gray-700 p-3 "
            "flex flex-col gap-2`,children:[(0,V.jsx)(`h3`,{className:`font-semibold text-lg "
            "text-left h-10 flex items-center`,children:`BluPe Evals`}),(0,V.jsx)(G,{onClick:B,"
            "className:`w-full bg-cyan-500 hover:bg-cyan-600 text-white`,children:`Open Evals`})]})"
        )
        if training_card in text:
            text = text.replace(training_card, training_card + "," + eval_card)
            changed = True
        else:
            raise ValueError("Could not find LeLab production training card")

    return text, changed


def _patch_file(path: Path, patcher) -> bool:
    text = path.read_text()
    patched, changed = patcher(text)
    if changed:
        _backup(path)
        path.write_text(patched)
    return changed


def main() -> int:
    try:
        import lelab
    except ImportError as exc:
        raise SystemExit("Could not import lelab. Activate the LeLab Python environment first.") from exc

    package_root = Path(lelab.__file__).resolve().parents[1]
    landing_path = package_root / "frontend" / "src" / "pages" / "Landing.tsx"
    dist_assets = package_root / "frontend" / "dist" / "assets"

    changed: list[str] = []
    if landing_path.exists() and _patch_file(landing_path, patch_landing_source):
        changed.append(str(landing_path))

    if dist_assets.exists():
        for path in sorted(dist_assets.glob("*.js")):
            try:
                if _patch_file(path, patch_dist_bundle_text):
                    changed.append(str(path))
            except ValueError:
                continue

    if changed:
        print("Patched LeLab files:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
