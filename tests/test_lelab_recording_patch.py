from pathlib import Path

import pytest

from scripts.apply_lelab_jetson_mjpg_patch import (
    MJPG_NEW_BLOCK,
    MJPG_OLD_BLOCK,
    TIMEOUT_NEW_BLOCK,
    TIMEOUT_OLD_BLOCK,
    patch_text,
)


def test_lelab_recording_patch_applies_all_recording_fixes() -> None:
    original = f"{MJPG_OLD_BLOCK}\n...\n{TIMEOUT_OLD_BLOCK}"

    patched, applied = patch_text(original, Path("/fake/lelab/record.py"))

    assert applied == [
        "jetson 3-camera MJPG default",
        "save episode on normal timeout",
    ]
    assert MJPG_NEW_BLOCK in patched
    assert TIMEOUT_NEW_BLOCK in patched
    assert "triggering re-record" not in patched
    assert 'web_events["rerecord_episode"] = True' not in patched


def test_lelab_recording_patch_is_idempotent() -> None:
    already_patched = f"{MJPG_NEW_BLOCK}\n...\n{TIMEOUT_NEW_BLOCK}"

    patched, applied = patch_text(already_patched)

    assert patched == already_patched
    assert applied == []


def test_lelab_recording_patch_fails_if_lelab_shape_changed() -> None:
    with pytest.raises(ValueError, match="save episode on normal timeout"):
        patch_text(MJPG_NEW_BLOCK)
