"""Fake Quest for headless stereo-video testing: plays the headset's side of the ZEDMINI flow.

Drives stereo_sender.StereoVisionServer (standalone, or inside mac_quest_bridge.py --video stereo)
exactly like the Quest app does: listen on a local video port, dial the control port, send
OPEN_CAMERA, then count the framed H.264 packets that come back. Exit 0 = video flowed.

    XR_INPUT=stub .venv/bin/python scripts/mac_quest_bridge.py --quest-ip 127.0.0.1 --cameras none &
    .venv/bin/python scripts/fake_quest_stereo.py        # [fake-quest] PASS: 30/30 packets ...
"""

import socket
import sys
import time

import tyro

from stereo_sender import build_open_camera, read_framed


def main(host: str = "127.0.0.1", control_port: int = 13579, video_port: int = 23456,
         width: int = 2560, height: int = 720, fps: int = 60, frames: int = 30,
         timeout: float = 15.0):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, video_port))                  # the Quest's decoder port: it listens, sender dials
    srv.listen(1)
    srv.settimeout(timeout)

    deadline = time.monotonic() + timeout          # retry: the eval may still be loading MuJoCo
    while True:
        try:
            ctrl = socket.create_connection((host, control_port), timeout=2.0)
            break
        except OSError:
            if time.monotonic() > deadline:
                print("[fake-quest] FAIL: control port never opened", flush=True)
                sys.exit(1)
            time.sleep(0.5)
    ctrl.sendall(build_open_camera(width, height, fps, 4_000_000, video_port, "ZEDMINI", host))
    print(f"[fake-quest] OPEN_CAMERA sent to :{control_port} -> waiting for video on :{video_port}",
          flush=True)

    conn, addr = srv.accept()
    conn.settimeout(timeout)
    print(f"[fake-quest] sender connected from {addr[0]}", flush=True)
    got, total_bytes, annexb = 0, 0, True
    t0 = time.monotonic()
    while got < frames and time.monotonic() - t0 < timeout:
        pkt = read_framed(conn)
        if pkt is None:
            break
        got += 1
        total_bytes += len(pkt)
        if got == 1 and not (pkt.startswith(b"\x00\x00\x00\x01") or pkt.startswith(b"\x00\x00\x01")):
            annexb = False
    for s in (conn, ctrl, srv):
        try:
            s.close()
        except OSError:
            pass
    ok = got >= frames and annexb
    print(f"[fake-quest] {'PASS' if ok else 'FAIL'}: {got}/{frames} packets "
          f"({total_bytes} bytes), Annex-B start code: {annexb}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    tyro.cli(main)
