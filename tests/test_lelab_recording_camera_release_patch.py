from pathlib import Path

from scripts.apply_lelab_recording_camera_release_patch import (
    BUNDLE_OLD,
    SOURCE_OLD,
    patch_dist_bundle_text,
    patch_source_text,
)


def test_recording_camera_release_source_waits_three_seconds() -> None:
    patched, changed = patch_source_text(SOURCE_OLD)

    assert changed is True
    assert "setTimeout(resolve, 3000)" in patched
    assert "setTimeout(resolve, 500)" not in patched


def test_recording_camera_release_source_is_idempotent() -> None:
    patched, changed = patch_source_text(SOURCE_OLD)
    patched_again, changed_again = patch_source_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_recording_camera_release_bundle_waits_three_seconds() -> None:
    patched, changed = patch_dist_bundle_text(BUNDLE_OLD)

    assert changed is True
    assert "setTimeout(e,3000)" in patched
    assert "setTimeout(e,500)" not in patched


def test_recording_camera_release_bundle_is_idempotent() -> None:
    patched, changed = patch_dist_bundle_text(BUNDLE_OLD)
    patched_again, changed_again = patch_dist_bundle_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_recording_camera_release_dist_index_cache_bust(tmp_path) -> None:
    from scripts.apply_lelab_recording_camera_release_patch import _patch_dist_index

    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text('<script type="module" src="/assets/app.js"></script>')
    (assets / "app.js").write_text("console.log('patched');")

    changed = _patch_dist_index(dist)

    assert changed
    assert (assets / "app-blupe-camrelease1.js").exists()
    assert 'src="/assets/app-blupe-camrelease1.js?blupe_camrelease=1"' in (
        dist / "index.html"
    ).read_text()
