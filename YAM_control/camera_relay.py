"""Robot-site camera relay: serve local webcams as multipart MJPEG over HTTP.

The M2 camera path (PLAN "Remote topology"): cameras stay at the robot site; the eval on the
operator's machine pulls these streams (CameraGrabber opens the URLs), composites the HUD,
and re-encodes to the headset. This keeps the Quest talking only to the operator node.

Run on the Orin (cv2 lives in the conda `xr` env, NOT the i2rt venv):
    ~/miniforge3/envs/xr/bin/python ~/blupe-evals/YAM_control/camera_relay.py --devices 0 2

Consume:  http://<orin>:8089/0  and  http://<orin>:8089/2   (one stream per device).
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"--frame"


class Cam:
    """Capture thread holding the latest JPEG for one device (drop frames, never queue)."""

    def __init__(self, dev, width, height, fps, quality):
        self.dev = dev
        self.jpeg = None
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok = self.cap.isOpened()
        self.quality = quality
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()
        print(f"[relay] /dev/video{dev}: {'open' if self.ok else 'FAILED'}", flush=True)

    def _run(self):
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()


CAMS = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):           # quiet
        pass

    def do_GET(self):
        key = self.path.strip("/")
        cam = CAMS.get(key)
        if cam is None or not cam.ok:
            self.send_error(404, f"no camera {key!r}; have {sorted(CAMS)}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        print(f"[relay] client {self.client_address} -> /{key}", flush=True)
        try:
            while True:
                with cam.lock:
                    jpeg = cam.jpeg
                if jpeg is not None:
                    self.wfile.write(BOUNDARY + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
                time.sleep(1.0 / 30)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[relay] client {self.client_address} left /{key}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()

    for d in args.devices:
        CAMS[str(d)] = Cam(d, args.width, args.height, args.fps, args.quality)
    if not any(c.ok for c in CAMS.values()):
        raise SystemExit("[relay] no cameras opened")
    print(f"[relay] serving {sorted(k for k, c in CAMS.items() if c.ok)} on :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
