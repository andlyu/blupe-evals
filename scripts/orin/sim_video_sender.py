"""Stream a rendered view of the MuJoCo sim to the Quest's Remote Vision (XRoboToolkit).

Wire format (verified vs the Quest decoder MediaDecoderTextureViewTCP.java + CameraDataReceiver):
the Quest LISTENs (ServerSocket); the sender connects and sends, per encoded frame:
    [4-byte big-endian length][H.264 Annex-B bytes]
No config handshake. The Quest re-accept()s in a loop, so we reconnect on any drop and start a
fresh encoder each time (SPS/PPS + IDR leads every connection so the decoder can sync).

Standalone tester: renders the YAM at home with a slow orbit. On the Quest: Remote Vision ->
set camera-source IP = this host -> LISTEN.  Then:
    MUJOCO_GL=glfw DISPLAY=:0 python scripts/orin/sim_video_sender.py --quest-ip 192.168.0.30
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")  # EGL is flaky on this Jetson; GLFW+DISPLAY works

import socket
import struct
import time
from fractions import Fraction

import av
import mujoco
import numpy as np
import tyro


def _new_encoder(width, height, fps):
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = width, height, "yuv420p"
    enc.time_base = Fraction(1, fps)
    enc.options = {"preset": "ultrafast", "tune": "zerolatency", "g": "15",
                   "profile": "baseline"}  # baseline = most MediaCodec-compatible
    return enc


def main(quest_ip: str, port: int = 12345, width: int = 960, height: int = 540, fps: int = 30):
    m = mujoco.MjModel.from_xml_path("assets/yam/scene.xml")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0.0, 0.3]
    cam.distance = 1.3
    cam.elevation = -20.0

    def connect():
        print(f"waiting for Quest Remote Vision (LISTEN) at {quest_ip}:{port} ...", flush=True)
        for _ in range(600):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            try:
                s.connect((quest_ip, port))
                s.settimeout(None)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except OSError:
                s.close()
                time.sleep(0.5)
        return None

    i = 0
    fast_fails = 0
    period = 1.0 / fps
    while True:
        sock = connect()
        if sock is None:
            print("never connected — is LISTEN on?", flush=True)
            return
        print("connected — streaming", flush=True)
        enc = _new_encoder(width, height, fps)  # fresh => leads with SPS/PPS + IDR
        sent_this_conn = 0
        try:
            while True:
                t0 = time.monotonic()
                cam.azimuth = (90 + i * 0.5) % 360
                renderer.update_scene(d, cam)
                rgb = renderer.render()
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = i
                for pkt in enc.encode(frame):
                    b = bytes(pkt)
                    sock.sendall(struct.pack(">I", len(b)) + b)
                    sent_this_conn += 1
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
            if sent_this_conn == 0:
                fast_fails += 1
                if fast_fails >= 6:
                    print(f"Quest keeps closing immediately ({e}) — likely a source/format "
                          f"mismatch in Remote Vision; stopping.", flush=True)
                    return
            else:
                fast_fails = 0
            print(f"stream dropped after {sent_this_conn} pkts ({e}); reconnecting...", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    tyro.cli(main)
