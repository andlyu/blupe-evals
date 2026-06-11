"""Stream the robot's USB webcam to the Quest's Remote Vision (XRoboToolkit).

The operator sees the REAL arm + scene. Same wire format as sim_video_sender.py (the Quest
LISTENs; we connect and send per encoded frame: [4-byte big-endian length][H.264 Annex-B]).
Network is LAN for now (remote / VPN comes later).

On the Quest: Remote Vision -> set camera-source IP = this host -> LISTEN.  Then:
    python scripts/orin/camera_sender.py --quest-ip 192.168.0.30
"""

import socket
import struct
import time
from fractions import Fraction

import av
import cv2
import numpy as np
import tyro


def _new_encoder(width, height, fps):
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = width, height, "yuv420p"
    enc.time_base = Fraction(1, fps)
    enc.options = {"preset": "ultrafast", "tune": "zerolatency", "g": "15",
                   "profile": "baseline"}   # baseline = most MediaCodec-compatible
    return enc


def _open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG -> 30fps over USB
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main(quest_ip: str, port: int = 12345, device: int = 0,
         width: int = 960, height: int = 540, fps: int = 30):
    cap = _open_camera(device, width, height, fps)
    if not cap.isOpened():
        print(f"cannot open camera /dev/video{device}", flush=True)
        return
    cw, ch = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {device} opened (capturing {cw}x{ch}, streaming {width}x{height})", flush=True)

    def connect():
        print(f"waiting for Quest Remote Vision (LISTEN) at {quest_ip}:{port} ...", flush=True)
        for _ in range(600):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            try:
                s.connect((quest_ip, port))
                s.settimeout(5.0)   # send timeout: a stalled Quest -> reconnect, not blocked
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except OSError:
                s.close()
                time.sleep(0.5)
        return None

    i = 0
    period = 1.0 / fps
    while True:
        sock = connect()
        if sock is None:
            print("never connected — is LISTEN on?", flush=True)
            return
        print("connected — streaming camera", flush=True)
        enc = _new_encoder(width, height, fps)   # fresh => leads with SPS/PPS + IDR
        sent = 0
        try:
            while True:
                t0 = time.monotonic()
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                if bgr.shape[1] != width or bgr.shape[0] != height:
                    bgr = cv2.resize(bgr, (width, height))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
                frame.pts = i
                for pkt in enc.encode(frame):
                    b = bytes(pkt)
                    sock.sendall(struct.pack(">I", len(b)) + b)
                    sent += 1
                i += 1
                if i % fps == 0:
                    print(f"sent {i} frames", flush=True)
                dt = period - (time.monotonic() - t0)
                if dt > 0:
                    time.sleep(dt)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            try:
                sock.close()
            except OSError:
                pass
            print(f"stream dropped after {sent} pkts ({e}); reconnecting...", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    tyro.cli(main)
