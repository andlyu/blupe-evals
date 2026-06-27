from scripts.apply_lelab_remote_camera_preview_patch import (
    BUNDLE_FALLBACK_OLD,
    BUNDLE_VIDEO_OLD,
    BUNDLE_VIDEO_RETRY_NEW,
    SOURCE_FALLBACK_OLD,
    SOURCE_HOOK_OLD,
    SOURCE_IMPORT,
    SOURCE_VIDEO_OLD,
    _patch_camera_configuration,
    _patch_dist_bundle,
    _patch_dist_index,
    _patch_server,
)


def test_remote_camera_preview_patch_adds_snapshot_backend(tmp_path) -> None:
    server = tmp_path / "server.py"
    server.write_text(
        "from fastapi.responses import JSONResponse\n\n"
        '@app.get("/available-cameras")\n'
        "def get_available_cameras():\n"
        "    return {}\n"
    )

    assert _patch_server(server)
    patched = server.read_text()

    assert "JSONResponse, Response, StreamingResponse" in patched
    assert '@app.get("/camera-preview/{camera_index}.jpg")' in patched
    assert '@app.get("/camera-preview/{camera_index}.mjpg")' in patched
    assert "camera {camera_index} is temporarily unavailable" in patched
    assert not _patch_server(server)


def test_remote_camera_preview_patch_updates_recording_modal_source(tmp_path) -> None:
    source = tmp_path / "CameraConfiguration.tsx"
    source.write_text(
        SOURCE_IMPORT
        + "const CameraConfiguration = () => {\n"
        + "  const {\n"
        + "    cameras: availableCameras,\n"
        + "    isLoading: isLoadingCameras,\n"
        + "    refresh: refreshCameras,\n"
        + "  } = useAvailableCameras();\n"
        + SOURCE_HOOK_OLD
        + SOURCE_VIDEO_OLD
        + SOURCE_FALLBACK_OLD
        + "};\n"
    )

    assert _patch_camera_configuration(source)
    patched = source.read_text()

    assert 'import { useCameraStream }' not in patched
    assert "useAvailableCameras({ enabled: cameras.length === 0 })" in patched
    assert "/camera-preview/" in patched
    assert ".jpg?width=" in patched
    assert "dataset.previewRetries" in patched
    assert "window.setTimeout" in patched
    assert "Preview failed" not in patched
    assert "No browser match" not in patched
    assert not _patch_camera_configuration(source)


def test_remote_camera_preview_patch_updates_dist_bundle_and_index(tmp_path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)

    bundle = assets / "index-test.js"
    bundle.write_text(
        "}=q_(),[s,c]=(0,_.useState)(``),[l,u]=(0,_.useState)(``);"
        + BUNDLE_VIDEO_OLD
        + BUNDLE_FALLBACK_OLD
    )
    index = dist / "index.html"
    index.write_text('<script type="module" crossorigin src="/assets/index-test.js"></script>')

    changed = _patch_dist_bundle(dist)
    assert changed == [str(bundle)]
    patched_bundle = bundle.read_text()

    assert "}=q_({enabled:e.length===0})," in patched_bundle
    assert "/camera-preview/${encodeURIComponent(e.camera_index)}.jpg" in patched_bundle
    assert "dataset.previewRetries" in patched_bundle
    assert "Preview unavailable" in patched_bundle
    assert "Preview failed" not in patched_bundle

    changed = _patch_dist_index(dist)
    assert str(index) in changed
    assert (assets / "index-test-blupe-preview3.js").exists()
    assert 'src="/assets/index-test-blupe-preview3.js?blupe_preview=5"' in index.read_text()


def test_remote_camera_preview_patch_removes_stale_bundle_browser_hook(tmp_path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)

    bundle = assets / "index-test-blupe-preview3.js"
    bundle.write_text(
        "}=q_({enabled:e.length===0}),[s,c]=(0,_.useState)(``);"
        "let{videoRef:i,hasError:a}=J_(e.device_id,t);return"
        + BUNDLE_VIDEO_RETRY_NEW
        + "children:t?`Preview paused`:`Preview unavailable`"
    )

    changed = _patch_dist_bundle(dist)
    assert changed == [str(bundle)]
    patched_bundle = bundle.read_text()

    assert "dataset.previewRetries" in patched_bundle
    assert "J_(e.device_id,t)" not in patched_bundle
    assert "videoRef:i" not in patched_bundle
