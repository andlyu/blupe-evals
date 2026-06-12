"""Robot-site camera relay: serve local webcams as multipart MJPEG over HTTP.

The M2 camera path (PLAN "Remote topology"): cameras stay at the robot site; the eval on the
operator's machine pulls these streams (CameraGrabber opens the URLs), composites the HUD,
and re-encodes to the headset. This keeps the Quest talking only to the operator node.

Run on the Orin (cv2 lives in the conda `xr` env, NOT the i2rt venv):
    ~/miniforge3/envs/xr/bin/python ~/blupe-evals/YAM_control/camera_relay.py --devices 0 2

Consume:  http://<orin>:8089/0  and  http://<orin>:8089/2   (one stream per device).
"""

import argparse
import select
import socket
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
        n = 0
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            n += 1
            # Staleness diagnostic: burn capture wall-time + frame counter into the pixels, so
            # EVERY consumer (browser, eval, headset) displays how old the frame it shows is.
            stamp = f"{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 10)} #{n}"
            cv2.rectangle(bgr, (0, 0), (250, 28), (0, 0, 0), -1)
            cv2.putText(bgr, stamp, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
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
        # Small send buffer + drop-on-backpressure: a slow consumer (Wi-Fi dip, cloud-relay
        # congestion) must get FEWER frames, not OLDER frames. Blocking writes with a big
        # default buffer queue seconds of video in TCP and the stream never catches up.
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        print(f"[relay] client {self.client_address} -> /{key}", flush=True)
        last = None
        try:
            while True:
                with cam.lock:
                    jpeg = cam.jpeg
                if jpeg is None or jpeg is last:       # only NEW frames (fps trap: re-sending
                    time.sleep(0.005)                  # dupes wastes the path's bandwidth)
                    continue
                _, writable, _ = select.select([], [self.connection], [], 0)
                if not writable:                       # client stalled -> drop, stay current
                    time.sleep(1.0 / 30)
                    continue
                last = jpeg
                self.wfile.write(BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg + b"\r\n")
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
