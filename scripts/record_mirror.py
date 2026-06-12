"""Record the headset mirror (preview_server MJPEG) to an mp4 — the "what I see in VR"
half of a demo video. Run it on the Mac while teleoping; Ctrl-C stops and finalizes.

    .venv/bin/python scripts/record_mirror.py                       # -> runs/demo/headset_<ts>.mp4
    .venv/bin/python scripts/record_mirror.py --out /tmp/demo.mp4

Reads http://127.0.0.1:8810/stream (the EXACT canvas the Quest shows, HUD included), so
it works during TELEOP — independent of the per-POLICY trial recorder. A connected reader
also makes the eval render the canvas, so this records even with no browser tab open.
"""

import os
import time
from fractions import Fraction

import tyro

import av
import cv2


def main(url: str = "http://127.0.0.1:8810/stream", out: str = "", fps: int = 30):
    if not out:
        os.makedirs("runs/demo", exist_ok=True)
        out = time.strftime("runs/demo/headset_%Y%m%d_%H%M%S.mp4")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"[record] cannot open {url} — is the eval running?")
    container = stream = None
    n, t0 = 0, time.monotonic()
    print(f"[record] {url} -> {out}  (Ctrl-C to stop)", flush=True)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("[record] stream ended", flush=True)
                break
            if container is None:                      # size known after the first frame
                h, w = bgr.shape[:2]
                container = av.open(out, "w")
                stream = container.add_stream("h264", rate=fps)
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.options = {"preset": "fast", "crf": "20"}
                # wall-clock pts: the mirror's frame rate VARIES (render gating, network);
                # a fixed-rate encode would time-warp the demo. 1 ms timebase = true speed.
                stream.codec_context.time_base = Fraction(1, 1000)
                t0 = time.monotonic()
            frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            frame.pts = int((time.monotonic() - t0) * 1000)
            for pkt in stream.encode(frame):
                container.mux(pkt)
            n += 1
            if n % (fps * 10) == 0:
                print(f"[record] {n} frames, {time.monotonic() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if container is not None:
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            print(f"[record] saved {out} ({n} frames, {time.monotonic() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
