#!/usr/bin/env python3
"""Patch installed LeLab to preview robot-side cameras through the backend.

Run inside the Python environment that provides `lelab`:

    python scripts/apply_lelab_remote_camera_preview_patch.py

LeLab's stock recording modal previews cameras with browser `getUserMedia`.
That works only when the browser and robot cameras are on the same machine.
For the BluPe Jetson setup, the browser is usually on the operator machine
through an SSH tunnel while the cameras are physically attached to the Jetson.
This patch adds small backend preview endpoints and points the recording modal
camera cards at one-shot `.jpg` snapshots. A single snapshot is enough to verify
camera naming without keeping the device open continuously.
"""

from __future__ import annotations

import shutil
import re
from pathlib import Path


DIST_ASSET_SUFFIX = "-blupe-preview3.js"
DIST_ASSET_QUERY = "blupe_preview=5"

SERVER_IMPORT_OLD = "from fastapi.responses import JSONResponse\n"
SERVER_IMPORT_STREAMING = "from fastapi.responses import JSONResponse, StreamingResponse\n"
SERVER_IMPORT_NEW = "from fastapi.responses import JSONResponse, Response, StreamingResponse\n"

SERVER_ENDPOINT_MARKER = '@app.get("/available-cameras")'
SERVER_SNAPSHOT_OPEN_OLD = """    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=404, detail=f"camera {camera_index} is not available")
"""
SERVER_SNAPSHOT_OPEN_NEW = """    cap = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)
        if cap.isOpened():
            break
        cap.release()
        cap = None
        time.sleep(0.1)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise HTTPException(status_code=503, detail=f"camera {camera_index} is temporarily unavailable")
"""
SERVER_ENDPOINT_BLOCK = r'''
def _camera_preview_jpeg(camera_index: int, width: int = 640, height: int = 360, fps: int = 15) -> bytes:
    """Capture one JPEG frame from a robot-side OpenCV camera."""
    import cv2

    safe_fps = max(1, min(int(fps or 15), 30))
    cap = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)
        if cap.isOpened():
            break
        cap.release()
        cap = None
        time.sleep(0.1)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise HTTPException(status_code=503, detail=f"camera {camera_index} is temporarily unavailable")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width or 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height or 360))
    cap.set(cv2.CAP_PROP_FPS, safe_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    try:
        frame = None
        for _ in range(8):
            ok, candidate = cap.read()
            if ok:
                frame = candidate
                break
            time.sleep(0.03)
        if frame is None:
            raise HTTPException(status_code=503, detail=f"camera {camera_index} did not return a frame")
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 75],
        )
        if not encoded_ok:
            raise HTTPException(status_code=500, detail=f"camera {camera_index} frame encode failed")
        return encoded.tobytes()
    finally:
        cap.release()


@app.get("/camera-preview/{camera_index}.jpg")
def get_camera_preview_snapshot(camera_index: int, width: int = 640, height: int = 360, fps: int = 15):
    return Response(
        content=_camera_preview_jpeg(camera_index, width=width, height=height, fps=fps),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


def _camera_preview_frames(camera_index: int, width: int = 640, height: int = 360, fps: int = 15):
    """Yield MJPEG frames from a robot-side OpenCV camera.

    The generator owns the camera only while a browser preview is connected.
    When the recording modal pauses previews before recording, the HTTP
    connection closes and the `finally` block releases the device.
    """
    import cv2

    safe_fps = max(1, min(int(fps or 15), 30))
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=404, detail=f"camera {camera_index} is not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width or 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height or 360))
    cap.set(cv2.CAP_PROP_FPS, safe_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    delay_s = 1.0 / safe_fps

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(delay_s)
                continue
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 75],
            )
            if not encoded_ok:
                time.sleep(delay_s)
                continue
            payload = encoded.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                + payload
                + b"\r\n"
            )
            time.sleep(delay_s)
    finally:
        cap.release()


@app.get("/camera-preview/{camera_index}.mjpg")
def get_camera_preview(camera_index: int, width: int = 640, height: int = 360, fps: int = 15):
    return StreamingResponse(
        _camera_preview_frames(camera_index, width=width, height=height, fps=fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


'''

SOURCE_IMPORT = 'import { useCameraStream } from "@/hooks/useCameraStream";\n'
SOURCE_HOOK_OLD = """  const { videoRef, hasError: streamError } = useCameraStream(
    camera.device_id,
    paused
  );
  const showVideo = !paused && camera.device_id && !streamError;
"""
SOURCE_HOOK_NEW = """  const showVideo = !paused && camera.camera_index !== undefined;
"""
SOURCE_AVAILABLE_OLD = """  } = useAvailableCameras();
"""
SOURCE_AVAILABLE_NEW = """  } = useAvailableCameras({ enabled: cameras.length === 0 });
"""
SOURCE_VIDEO_OLD = """        {showVideo ? (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
"""
SOURCE_VIDEO_NEW = """        {showVideo ? (
          <img
            src={`/camera-preview/${encodeURIComponent(String(camera.camera_index))}.jpg?width=${camera.width || 640}&height=${camera.height || 360}&fps=${camera.fps || 15}&t=${Date.now()}`}
            alt={`${camera.name} preview`}
            className="w-full h-full object-cover"
          />
"""
SOURCE_VIDEO_RETRY_NEW = """        {showVideo ? (
          <img
            src={`/camera-preview/${encodeURIComponent(String(camera.camera_index))}.jpg?width=${camera.width || 640}&height=${camera.height || 360}&fps=${camera.fps || 15}&t=${Date.now()}`}
            alt={`${camera.name} preview`}
            className="w-full h-full object-cover"
            onError={(event) => {
              const image = event.currentTarget;
              const retries = Number(image.dataset.previewRetries || "0");
              if (retries >= 60) return;
              image.dataset.previewRetries = String(retries + 1);
              window.setTimeout(() => {
                image.src = `/camera-preview/${encodeURIComponent(String(camera.camera_index))}.jpg?width=${camera.width || 640}&height=${camera.height || 360}&fps=${camera.fps || 15}&t=${Date.now()}`;
              }, 500);
            }}
            onLoad={(event) => {
              event.currentTarget.dataset.previewRetries = "0";
            }}
          />
"""
SOURCE_VIDEO_MJPEG = """        {showVideo ? (
          <img
            src={`/camera-preview/${encodeURIComponent(String(camera.camera_index))}.mjpg?width=${camera.width || 640}&height=${camera.height || 360}&fps=${camera.fps || 15}`}
            alt={`${camera.name} preview`}
            className="w-full h-full object-cover"
          />
"""
SOURCE_FALLBACK_OLD = """              {paused
                ? "Preview paused"
                : camera.device_id
                ? "Preview failed"
                : "No browser match"}
"""
SOURCE_FALLBACK_NEW = """              {paused ? "Preview paused" : "Preview unavailable"}
"""

BUNDLE_HOOK_OLD = "let{videoRef:i,hasError:a}=J_(e.device_id,t);return"
BUNDLE_HOOK_PATTERN = re.compile(
    r"let\{videoRef:[A-Za-z_$][A-Za-z0-9_$]*,hasError:[A-Za-z_$][A-Za-z0-9_$]*\}="
    r"[A-Za-z_$][A-Za-z0-9_$]*\(e\.device_id,t\);return"
)
BUNDLE_HOOK_NEW = "return"
BUNDLE_AVAILABLE_PATTERN = re.compile(
    r"\}=([A-Za-z_$][A-Za-z0-9_$]*)\(\),(\[[A-Za-z_$][A-Za-z0-9_$]*,[A-Za-z_$][A-Za-z0-9_$]*\]=\(0,_.useState\)\(``\),\[[A-Za-z_$][A-Za-z0-9_$]*,[A-Za-z_$][A-Za-z0-9_$]*\]=\(0,_.useState\)\(``\))"
)
BUNDLE_VIDEO_OLD = (
    "children:!t&&e.device_id&&!a?(0,V.jsx)(`video`,"
    "{ref:i,autoPlay:!0,muted:!0,playsInline:!0,className:`w-full h-full object-cover`}):"
)
BUNDLE_VIDEO_NEW = (
    "children:!t&&e.camera_index!==void 0?(0,V.jsx)(`img`,"
    "{src:`/camera-preview/${encodeURIComponent(e.camera_index)}.jpg?width=${e.width||640}&height=${e.height||360}&fps=${e.fps||15}&t=${Date.now()}`,"
    "alt:`${e.name} preview`,className:`w-full h-full object-cover`}):"
)
BUNDLE_VIDEO_RETRY_NEW = (
    "children:!t&&e.camera_index!==void 0?(0,V.jsx)(`img`,"
    "{src:`/camera-preview/${encodeURIComponent(e.camera_index)}.jpg?width=${e.width||640}&height=${e.height||360}&fps=${e.fps||15}&t=${Date.now()}`,"
    "alt:`${e.name} preview`,className:`w-full h-full object-cover`,onError:o=>{let s=o.currentTarget,c=Number(s.dataset.previewRetries||`0`);"
    "c<60&&(s.dataset.previewRetries=String(c+1),window.setTimeout(()=>{s.src=`/camera-preview/${encodeURIComponent(e.camera_index)}.jpg?width=${e.width||640}&height=${e.height||360}&fps=${e.fps||15}&t=${Date.now()}`},500))},"
    "onLoad:o=>{o.currentTarget.dataset.previewRetries=`0`}}):"
)
BUNDLE_VIDEO_MJPEG = (
    "children:!t&&e.camera_index!==void 0?(0,V.jsx)(`img`,"
    "{src:`/camera-preview/${encodeURIComponent(e.camera_index)}.mjpg?width=${e.width||640}&height=${e.height||360}&fps=${e.fps||15}`,"
    "alt:`${e.name} preview`,className:`w-full h-full object-cover`}):"
)
BUNDLE_FALLBACK_OLD = "children:t?`Preview paused`:e.device_id?`Preview failed`:`No browser match`"
BUNDLE_FALLBACK_NEW = "children:t?`Preview paused`:`Preview unavailable`"


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".blupe-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def _replace_required(text: str, old: str, new: str, path: Path) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise SystemExit(f"Could not find expected block in {path}")
    return text.replace(old, new), True


def _patch_server(server_path: Path) -> bool:
    text = server_path.read_text()
    changed = False

    if SERVER_IMPORT_NEW not in text:
        if SERVER_IMPORT_STREAMING in text:
            text = text.replace(SERVER_IMPORT_STREAMING, SERVER_IMPORT_NEW)
        elif SERVER_IMPORT_OLD in text:
            text = text.replace(SERVER_IMPORT_OLD, SERVER_IMPORT_NEW)
        else:
            raise SystemExit(f"Could not find FastAPI response import in {server_path}")
        changed = True

    if "/camera-preview/{camera_index}.mjpg" not in text:
        if SERVER_ENDPOINT_MARKER not in text:
            raise SystemExit(f"Could not find available-cameras marker in {server_path}")
        text = text.replace(SERVER_ENDPOINT_MARKER, SERVER_ENDPOINT_BLOCK + SERVER_ENDPOINT_MARKER)
        changed = True
    elif "/camera-preview/{camera_index}.jpg" not in text:
        marker = '@app.get("/camera-preview/{camera_index}.mjpg")'
        if marker not in text:
            raise SystemExit(f"Could not find MJPEG preview marker in {server_path}")
        snapshot_block = SERVER_ENDPOINT_BLOCK.split("def _camera_preview_frames", 1)[0]
        text = text.replace(marker, snapshot_block + marker)
        changed = True

    if SERVER_SNAPSHOT_OPEN_OLD in text and SERVER_SNAPSHOT_OPEN_NEW not in text:
        text = text.replace(SERVER_SNAPSHOT_OPEN_OLD, SERVER_SNAPSHOT_OPEN_NEW, 1)
        changed = True

    if changed:
        _backup(server_path)
        server_path.write_text(text)
    return changed


def _patch_camera_configuration(source_path: Path) -> bool:
    text = source_path.read_text()
    changed = False

    if SOURCE_IMPORT in text:
        text = text.replace(SOURCE_IMPORT, "")
        changed = True
    if SOURCE_AVAILABLE_NEW not in text:
        text, did_change = _replace_required(text, SOURCE_AVAILABLE_OLD, SOURCE_AVAILABLE_NEW, source_path)
        changed = changed or did_change
    text, did_change = _replace_required(text, SOURCE_HOOK_OLD, SOURCE_HOOK_NEW, source_path)
    changed = changed or did_change
    if SOURCE_VIDEO_RETRY_NEW not in text:
        if SOURCE_VIDEO_NEW in text:
            text = text.replace(SOURCE_VIDEO_NEW, SOURCE_VIDEO_RETRY_NEW)
            changed = True
        elif SOURCE_VIDEO_MJPEG in text:
            text = text.replace(SOURCE_VIDEO_MJPEG, SOURCE_VIDEO_RETRY_NEW)
            changed = True
        else:
            text, did_change = _replace_required(text, SOURCE_VIDEO_OLD, SOURCE_VIDEO_RETRY_NEW, source_path)
            changed = changed or did_change
    text, did_change = _replace_required(text, SOURCE_FALLBACK_OLD, SOURCE_FALLBACK_NEW, source_path)
    changed = changed or did_change

    if changed:
        _backup(source_path)
        source_path.write_text(text)
    return changed


def _patch_dist_bundle(dist: Path) -> list[str]:
    assets = dist / "assets"
    if not assets.exists():
        return []

    changed_paths: list[str] = []
    for path in assets.glob("*.js"):
        text = path.read_text()
        has_snapshot_preview = "/camera-preview/${encodeURIComponent(e.camera_index)}.jpg" in text
        has_retry_preview = "dataset.previewRetries" in text
        has_scan_gate = "{enabled:e.length===0}" in text
        has_browser_hook = BUNDLE_HOOK_OLD in text or BUNDLE_HOOK_PATTERN.search(text) is not None
        if has_snapshot_preview and has_retry_preview and has_scan_gate and not has_browser_hook:
            continue
        if BUNDLE_VIDEO_OLD not in text and BUNDLE_VIDEO_MJPEG not in text and BUNDLE_VIDEO_NEW not in text:
            changed = False
            if not has_scan_gate:
                text, count = BUNDLE_AVAILABLE_PATTERN.subn(r"}=\1({enabled:e.length===0}),\2", text, count=1)
                if count:
                    changed = True
            if BUNDLE_HOOK_OLD in text:
                text = text.replace(BUNDLE_HOOK_OLD, BUNDLE_HOOK_NEW, 1)
                changed = True
            else:
                text, hook_count = BUNDLE_HOOK_PATTERN.subn(BUNDLE_HOOK_NEW, text, count=1)
                changed = changed or hook_count > 0
            if changed:
                _backup(path)
                path.write_text(text)
                changed_paths.append(str(path))
            continue
        did_available = False
        if not has_scan_gate:
            text, count = BUNDLE_AVAILABLE_PATTERN.subn(r"}=\1({enabled:e.length===0}),\2", text, count=1)
            did_available = count > 0
        did_hook = False
        if BUNDLE_HOOK_OLD in text:
            text, did_hook = _replace_required(text, BUNDLE_HOOK_OLD, BUNDLE_HOOK_NEW, path)
        else:
            text, hook_count = BUNDLE_HOOK_PATTERN.subn(BUNDLE_HOOK_NEW, text, count=1)
            did_hook = hook_count > 0
        if BUNDLE_VIDEO_NEW in text:
            text = text.replace(BUNDLE_VIDEO_NEW, BUNDLE_VIDEO_RETRY_NEW)
            did_video = True
        elif BUNDLE_VIDEO_MJPEG in text:
            text = text.replace(BUNDLE_VIDEO_MJPEG, BUNDLE_VIDEO_RETRY_NEW)
            did_video = True
        else:
            text, did_video = _replace_required(text, BUNDLE_VIDEO_OLD, BUNDLE_VIDEO_RETRY_NEW, path)
        text, did_fallback = _replace_required(text, BUNDLE_FALLBACK_OLD, BUNDLE_FALLBACK_NEW, path)
        if did_available or did_hook or did_video or did_fallback:
            _backup(path)
            path.write_text(text)
            changed_paths.append(str(path))
    return changed_paths


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
        base_name = re.sub(r"-blupe-preview\d+\.js$", ".js", source_name)
        target_name = base_name.removesuffix(".js") + DIST_ASSET_SUFFIX
        target_path = dist / "assets" / target_name

    changed: list[str] = []
    if not target_path.exists() or target_path.read_text() != source_path.read_text():
        shutil.copy2(source_path, target_path)
        changed.append(str(target_path))

    old = match.group(0)
    new = f'src="/assets/{target_name}?{DIST_ASSET_QUERY}"'
    if old == new:
        return changed

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
    server_path = Path(lelab.__file__).resolve().with_name("server.py")
    source_path = package_root / "frontend" / "src" / "components" / "recording" / "CameraConfiguration.tsx"
    dist = package_root / "frontend" / "dist"

    changed: list[str] = []
    if _patch_server(server_path):
        changed.append(str(server_path))
    if source_path.exists() and _patch_camera_configuration(source_path):
        changed.append(str(source_path))
    changed.extend(_patch_dist_bundle(dist))
    changed.extend(_patch_dist_index(dist))

    if changed:
        print("Patched LeLab remote camera preview:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Already patched: LeLab remote camera preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
