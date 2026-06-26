#!/usr/bin/env python3
"""Patch installed LeLab to expose the blupe dataset editor entry point.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_dataset_editor_patch.py

This is intentionally idempotent. It edits the installed LeLab frontend source
used by `lelab --dev`, adding a small Edit Dataset panel on the landing page
between Dataset and Create a model. It also patches the production bundle and
SPA route fallback when those installed files are present, so direct
`/edit-dataset` browser loads work on the Jetson.
"""

from __future__ import annotations

import shutil
from pathlib import Path


DEFAULT_EDITOR_URL = "http://127.0.0.1:8092/episodes"
OLD_HUB_EDITOR_URL = "http://127.0.0.1:8105/dataset"


EDIT_DATASET_TSX = """import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, Play, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";

const DEFAULT_EDITOR_URL = "http://127.0.0.1:8092/episodes";

const EditDataset = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const datasetInfo = location.state?.datasetInfo || {};
  const editorUrl = import.meta.env.VITE_BLUPE_DATASET_EDITOR_URL || DEFAULT_EDITOR_URL;
  const datasetRepoId = datasetInfo.dataset_repo_id || "";

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <header className="border-b border-gray-800 bg-black/95 px-4 py-3">
        <div className="mx-auto max-w-7xl flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate("/")}
              className="bg-gray-900 border-gray-700 text-white hover:bg-gray-800"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Home
            </Button>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">Edit Dataset</h1>
              <p className="text-sm text-gray-400">
                {datasetRepoId || "Select segments before upload or training"}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {datasetRepoId && (
              <Button
                variant="outline"
                onClick={() => navigate("/upload", { state: { datasetInfo } })}
                className="bg-gray-900 border-gray-700 text-white hover:bg-gray-800"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload
              </Button>
            )}
            <Button
              onClick={() =>
                navigate("/training", {
                  state: datasetRepoId ? { datasetRepoId } : undefined,
                })
              }
              className="bg-green-500 hover:bg-green-600 text-white"
            >
              <Play className="w-4 h-4 mr-2" />
              Training
            </Button>
          </div>
        </div>
      </header>
      <main className="flex-1 min-h-0">
        <iframe
          title="Blupe dataset editor"
          src={editorUrl}
          className="w-full h-[calc(100vh-82px)] border-0 bg-black"
        />
      </main>
    </div>
  );
};

export default EditDataset;
"""


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def _replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise SystemExit(f"Could not find expected block in {path}")
    _backup(path)
    path.write_text(text.replace(old, new))
    return True


def _replace_if_present(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if old not in text:
        return False
    _backup(path)
    path.write_text(text.replace(old, new))
    return True


def _patch_dist_bundle(dist: Path) -> list[str]:
    assets = dist / "assets"
    if not assets.exists():
        return []
    changed = []
    for path in assets.glob("*.js"):
        did_change = False
        if _replace_if_present(path, OLD_HUB_EDITOR_URL, DEFAULT_EDITOR_URL):
            did_change = True
        # Older patched bundles computed `${hub}/dataset?station=...` instead of
        # carrying the literal URL. Patch that compiled expression as well.
        text = path.read_text()
        compiled_hub_expr = 'r=`${bie.replace(/\\/$/,``)}/dataset?station=${encodeURIComponent(xie)}`'
        compiled_editor_expr = f'r=`{DEFAULT_EDITOR_URL}`'
        if compiled_hub_expr in text and compiled_editor_expr not in text:
            _backup(path)
            path.write_text(text.replace(compiled_hub_expr, compiled_editor_expr))
            did_change = True
        if did_change:
            changed.append(str(path))
    return changed


def _patch_server_spa_fallback(server_path: Path) -> bool:
    text = server_path.read_text()
    if "class SPAStaticFiles(StaticFiles)" in text:
        return False

    changed = False
    imports = {
        "from starlette.datastructures import Headers\n": "from starlette.datastructures import Headers\n",
        "from starlette.exceptions import HTTPException as StarletteHTTPException\n": (
            "from starlette.exceptions import HTTPException as StarletteHTTPException\n"
        ),
        "from starlette.responses import Response\n": "from starlette.responses import Response\n",
        "from starlette.types import Scope\n": "from starlette.types import Scope\n",
    }
    for import_line in imports:
        if import_line not in text:
            marker = "from fastapi.staticfiles import StaticFiles\n"
            if marker not in text:
                raise SystemExit(f"Could not find StaticFiles import in {server_path}")
            text = text.replace(marker, marker + import_line)
            changed = True

    old_mount = 'app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")'
    if old_mount not in text:
        if changed:
            _backup(server_path)
            server_path.write_text(text)
        return changed

    fallback = '''

def _accepts_html(accept: str) -> bool:
    for part in accept.split(","):
        media_type, _, params = part.strip().partition(";")
        if media_type.strip().lower() != "text/html":
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and _accepts_html(Headers(scope=scope).get("accept", "")):
                return await super().get_response("index.html", scope)
            raise
'''
    mount_marker = "\n# Serve the built frontend at /. Must be mounted last so API routes win.\n"
    if mount_marker not in text:
        raise SystemExit(f"Could not find frontend mount marker in {server_path}")
    text = text.replace(mount_marker, fallback + mount_marker)
    text = text.replace(old_mount, 'app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")')
    changed = True

    if changed:
        _backup(server_path)
        server_path.write_text(text)
    return changed


def main() -> int:
    try:
        import lelab
    except ImportError as exc:
        raise SystemExit("Could not import lelab. Activate the LeLab Python environment first.") from exc

    package_root = Path(lelab.__file__).resolve().parents[1]
    frontend = package_root / "frontend" / "src"
    dist = package_root / "frontend" / "dist"
    server_path = Path(lelab.__file__).resolve().with_name("server.py")
    edit_path = frontend / "pages" / "EditDataset.tsx"
    landing_path = frontend / "pages" / "Landing.tsx"

    changed = []

    if "Blupe dataset editor" not in edit_path.read_text():
        _backup(edit_path)
        edit_path.write_text(EDIT_DATASET_TSX)
        changed.append(str(edit_path))

    if _replace_once(
        landing_path,
        '  const handleTrainingClick = () => navigate("/training");\n',
        f'  const handleTrainingClick = () => navigate("/training");\n  const handleEditDatasetClick = () => window.location.assign("{DEFAULT_EDITOR_URL}");\n',
    ):
        changed.append(str(landing_path))

    if _replace_if_present(
        landing_path,
        '  const handleEditDatasetClick = () => navigate("/edit-dataset");',
        f'  const handleEditDatasetClick = () => window.location.assign("{DEFAULT_EDITOR_URL}");',
    ):
        changed.append(str(landing_path))

    if _replace_once(
        landing_path,
        '<div className="grid grid-cols-1 md:grid-cols-2 gap-3">',
        '<div className="grid grid-cols-1 md:grid-cols-3 gap-3">',
    ):
        changed.append(str(landing_path))

    edit_card = """            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
              <h3 className="font-semibold text-lg text-left h-10 flex items-center">
                Edit dataset
              </h3>
              <Button
                onClick={handleEditDatasetClick}
                className="w-full bg-blue-500 hover:bg-blue-600 text-white"
              >
                Edit Dataset
              </Button>
            </div>
"""
    if edit_card not in landing_path.read_text():
        _replace_once(
            landing_path,
            '            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">\n              <h3 className="font-semibold text-lg text-left h-10 flex items-center">\n                Create a model\n',
            edit_card
            + '            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">\n              <h3 className="font-semibold text-lg text-left h-10 flex items-center">\n                Create a model\n',
        )
        changed.append(str(landing_path))

    changed.extend(_patch_dist_bundle(dist))

    if _patch_server_spa_fallback(server_path):
        changed.append(str(server_path))

    if changed:
        print("Patched LeLab files:")
        for path in sorted(set(changed)):
            print(f"  {path}")
    else:
        print("Already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
