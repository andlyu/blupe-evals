#!/usr/bin/env python3
"""Patch installed LeLab launcher to support disabling browser auto-open.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_no_browser_patch.py

Then launch with:

    LELAB_OPEN_BROWSER=0 lelab

This is useful when LeLab runs on the Jetson behind an SSH tunnel and the Mac
browser tab is opened manually.
"""

from __future__ import annotations

import shutil
from pathlib import Path


HELPER_MARKER = "def _should_open_browser() -> bool:"

HELPER_BLOCK = """
def _should_open_browser() -> bool:
    value = os.environ.get("LELAB_OPEN_BROWSER", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


"""

OPEN_HELPER_OLD = """def _open_browser_when_ready():
    \"\"\"Background-thread helper: poll the port, open the browser when up.\"\"\"
"""

OPEN_HELPER_NEW = """def _open_browser_when_ready():
    \"\"\"Background-thread helper: poll the port, open the browser when up.\"\"\"
    if not _should_open_browser():
        logger.info("Browser auto-open disabled by LELAB_OPEN_BROWSER=0")
        return
"""

DEV_OPEN_OLD = """    logger.info("🌐 Opening browser...")
    webbrowser.open(f"http://localhost:{FRONTEND_DEV_PORT}/")
"""

DEV_OPEN_NEW = """    if _should_open_browser():
        logger.info("🌐 Opening browser...")
        webbrowser.open(f"http://localhost:{FRONTEND_DEV_PORT}/")
    else:
        logger.info("Browser auto-open disabled by LELAB_OPEN_BROWSER=0")
"""


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    if HELPER_MARKER not in text:
        marker = "\n\ndef _wait_for_port"
        if marker not in text:
            raise ValueError("Could not find _wait_for_port marker")
        text = text.replace(marker, "\n\n" + HELPER_BLOCK + "def _wait_for_port", 1)
        changed = True

    if OPEN_HELPER_NEW not in text:
        if OPEN_HELPER_OLD not in text:
            raise ValueError("Could not find _open_browser_when_ready block")
        text = text.replace(OPEN_HELPER_OLD, OPEN_HELPER_NEW, 1)
        changed = True

    if DEV_OPEN_NEW not in text:
        if DEV_OPEN_OLD not in text:
            raise ValueError("Could not find dev browser open block")
        text = text.replace(DEV_OPEN_OLD, DEV_OPEN_NEW, 1)
        changed = True

    return text, changed


def main() -> int:
    try:
        import lelab.scripts.lelab
    except ImportError as exc:
        raise SystemExit("Could not import lelab.scripts.lelab. Activate LeLab first.") from exc

    path = Path(lelab.scripts.lelab.__file__)
    text = path.read_text()
    patched, changed = patch_text(text)
    if not changed:
        print(f"Already patched: {path}")
        return 0

    _backup(path)
    path.write_text(patched)
    print(f"Patched LeLab no-browser launcher: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
