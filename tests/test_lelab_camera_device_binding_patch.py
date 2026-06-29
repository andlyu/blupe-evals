from scripts.apply_lelab_camera_device_binding_patch import (
    BUNDLE_OLD,
    SOURCE_OLD,
    _patch_dist_index,
    patch_dist_bundle_text,
    patch_source_text,
)


def test_camera_device_binding_patch_updates_source() -> None:
    patched, changed = patch_source_text(SOURCE_OLD)

    assert changed is True
    assert "matchByDeviceId" in patched
    assert "matchByIndex" in patched
    assert "updates.device_id = match.deviceId" in patched


def test_camera_device_binding_patch_source_is_idempotent() -> None:
    patched, changed = patch_source_text(SOURCE_OLD)
    patched_again, changed_again = patch_source_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_camera_device_binding_patch_updates_production_bundle_text() -> None:
    patched, changed = patch_dist_bundle_text(BUNDLE_OLD + "tail")

    assert changed is True
    assert "a=t??r;if(!a)return e;let o={},s=!1" in patched
    assert "o.device_id=a.deviceId" in patched
    assert "o.camera_index=a.index" in patched


def test_camera_device_binding_patch_bundle_is_idempotent() -> None:
    patched, changed = patch_dist_bundle_text(BUNDLE_OLD + "tail")
    patched_again, changed_again = patch_dist_bundle_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_camera_device_binding_patch_updates_dist_index(tmp_path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    bundle = assets / "index-test.js"
    bundle.write_text("patched bundle")
    index = dist / "index.html"
    index.write_text('<script type="module" crossorigin src="/assets/index-test.js?old=1"></script>')

    changed = _patch_dist_index(dist)

    assert str(assets / "index-test-blupe-camdev1.js") in changed
    assert str(index) in changed
    assert (assets / "index-test-blupe-camdev1.js").read_text() == "patched bundle"
    assert 'src="/assets/index-test-blupe-camdev1.js?blupe_camdev=1"' in index.read_text()
