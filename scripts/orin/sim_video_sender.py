"""Stream a rendered view of the MuJoCo sim to the Quest's Remote Vision (XRoboToolkit).

Wire format (from XRoboToolkit-Orin-Video-Sender): the sender CONNECTS to the headset
(which is in Remote Vision "Listen" mode) and sends, per encoded frame:
    [4-byte big-endian length][H.264 Annex-B bytes]

This standalone tester renders the YAM at its home pose with a slowly orbiting camera, to
validate the pipeline end-to-end before we wire it into the live teleop loop.

  python scripts/orin/sim_video_sender.py --quest-ip 192.168.0.XXX [--port 12345]

Needs (xr env): mujoco (offscreen via EGL), av (PyAV / libx264).
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen GL on the Orin

import socket
import struct
import time
from fractions import Fraction

import av
import mujoco
import numpy as np
import tyro


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

    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = width, height, "yuv420p"
    enc.time_base = Fraction(1, fps)
    enc.options = {"preset": "ultrafast", "tune": "zerolatency", "g": "15"}

    print(f"connecting to Quest Remote Vision at {quest_ip}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((quest_ip, port))  # Quest is the listener
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("connected — streaming (Ctrl-C to stop)")

    def send(pkt_bytes):
        sock.sendall(struct.pack(">I", len(pkt_bytes)) + pkt_bytes)

    i = 0
    period = 1.0 / fps
    try:
        while True:
            t0 = time.monotonic()
            cam.azimuth = (90 + i * 0.5) % 360  # slow orbit so it's visibly live
            renderer.update_scene(d, cam)
            rgb = renderer.render()  # HxWx3 uint8
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            for pkt in enc.encode(frame):
                send(bytes(pkt))
            i += 1
            if i % fps == 0:
                print(f"sent {i} frames")
            dt = period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        for pkt in enc.encode(None):  # flush
            send(bytes(pkt))
        sock.close()
        print("stopped")


if __name__ == "__main__":
    tyro.cli(main)
