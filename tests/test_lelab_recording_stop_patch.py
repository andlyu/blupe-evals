from scripts.apply_lelab_recording_stop_patch import (
    BUNDLE_PRIMARY_BUTTON,
    SOURCE_PRIMARY_BUTTON,
    _patch_dist_index,
    patch_dist_bundle_text,
    patch_recording_source,
)


def test_recording_stop_patch_updates_source() -> None:
    source = SOURCE_PRIMARY_BUTTON + "completed marker"

    patched, changed = patch_recording_source(source)

    assert changed is True
    assert "Stop Collection" in patched
    assert "requestStopRecording" in patched
    assert "ESC" in patched


def test_recording_stop_patch_source_is_idempotent() -> None:
    source = SOURCE_PRIMARY_BUTTON + "completed marker"

    patched, changed = patch_recording_source(source)
    patched_again, changed_again = patch_recording_source(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_recording_stop_patch_updates_production_bundle_text() -> None:
    bundle = BUNDLE_PRIMARY_BUTTON + ",I===`completed`&&tail"

    patched, changed = patch_dist_bundle_text(bundle)

    assert changed is True
    assert "Stop Collection" in patched
    assert "onClick:j" in patched
    assert "s.available_controls.stop_recording" in patched
    assert "children:`ESC`" in patched


def test_recording_stop_patch_bundle_is_idempotent() -> None:
    bundle = BUNDLE_PRIMARY_BUTTON + ",I===`completed`&&tail"

    patched, changed = patch_dist_bundle_text(bundle)
    patched_again, changed_again = patch_dist_bundle_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_recording_stop_patch_updates_dist_index(tmp_path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    bundle = assets / "index-test.js"
    bundle.write_text("patched bundle")
    index = dist / "index.html"
    index.write_text('<script type="module" crossorigin src="/assets/index-test.js"></script>')

    changed = _patch_dist_index(dist)

    assert str(assets / "index-test-blupe-stop1.js") in changed
    assert str(index) in changed
    assert (assets / "index-test-blupe-stop1.js").read_text() == "patched bundle"
    assert 'src="/assets/index-test-blupe-stop1.js?blupe_stop=1"' in index.read_text()
