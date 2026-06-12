"""Stereo camera -> Quest stereo-3D view (XRoboToolkit "ZEDMINI" Remote Vision flow).

3D sight for teleop: send the headset a side-by-side stereo frame and the Quest app
renders one half per eye (press B on the right controller to toggle flat <-> 3D).

This flow REVERSES the mono Remote Vision direction (camera_sender.py pushes to the
Quest's LISTEN port; here the Quest dials US):
  1. We LISTEN on :13579 (the control port).
  2. Quest: Camera panel -> video source ZEDMINI -> Listen -> enter THIS host's IP.
     The app connects and sends OPEN_CAMERA carrying CameraRequestData: its own IP,
     the video port (12345), and the preset's frame size (2560x720@60).
  3. We connect back to quest_ip:12345 and stream H.264 side-by-side frames with the
     same wire format as the mono flow: [4-byte big-endian length][Annex-B].
  4. CLOSE_CAMERA (or either socket dropping) stops the stream; we keep listening.

Protocol verified against both ends' source: XRoboToolkit-Orin-Video-Sender
(main_zed_tcp.cpp, the working wire peer) and XRoboToolkit-Unity-Client-Quest
(CameraRequestSerializer.cs / NetworkDataProtocolSerializer.cs). Wire details in
docs/refs/xrobotoolkit/stereo-vision.md.

Camera input (robot side):
  --left 0 --right 2     two separate UVC cameras mounted as a stereo pair
  --device 0             ONE dual-lens stereo camera whose frames are already
                         side-by-side (most "3D USB" cams); --swap if eyes are flipped
  (none)                 synthetic disparity test pattern -- proves the whole path,
                         including real 3D pop, with no camera plugged in

Run:      python scripts/stereo_sender.py --left 0 --right 2
Verify:   python scripts/stereo_sender.py --left 0 --right 2 --snapshot /tmp/sbs.png
"""

import select
import socket
import struct
import threading
import time
from fractions import Fraction
from typing import Optional

import numpy as np
import tyro

try:
    import cv2
except ImportError:
    cv2 = None

CONTROL_PORT = 13579               # the Quest app dials this (hardcoded in TcpManager.cs)


class LatencyStats:
    """Per-stage latency tracker for the teleop data flow. note(stage, seconds) from any
    thread; report() drains and formats "stage=avg/max ms" so each window stands alone.
    Cross-machine rule: only ever note() durations measured on ONE clock (e.g. ack RTT =
    send->echo on the Mac) — Mac/Orin monotonic clocks are not comparable."""

    def __init__(self):
        self._lock = threading.Lock()
        self._acc = {}                 # stage -> [sum_s, n, max_s]

    def note(self, stage, dt):
        with self._lock:
            a = self._acc.setdefault(stage, [0.0, 0, 0.0])
            a[0] += dt
            a[1] += 1
            if dt > a[2]:
                a[2] = dt

    def count(self, stage):
        """Tally an event (e.g. dropped frame) with no duration."""
        with self._lock:
            a = self._acc.setdefault(stage, [0.0, 0, 0.0])
            a[1] += 1

    def report(self):
        with self._lock:
            acc, self._acc = self._acc, {}
        parts = []
        for k in sorted(acc):
            s, n, mx = acc[k]
            if s == 0.0 and mx == 0.0:
                parts.append(f"{k}={n}")                       # pure counter
            else:
                parts.append(f"{k}={s / n * 1e3:.1f}/{mx * 1e3:.1f}ms")
        return " ".join(parts)


# ---------------------------------------------------------------- wire protocol

def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_framed(sock):
    """One control message: [4-byte big-endian length][body]. None on EOF."""
    head = _recv_exact(sock, 4)
    if head is None:
        return None
    (length,) = struct.unpack(">I", head)
    if length == 0 or length > 1 << 20:
        return None                                   # nonsense length -> treat as EOF
    return _recv_exact(sock, length)


def parse_protocol(body):
    """NetworkDataProtocol: [int32 LE cmdLen][cmd utf-8][int32 LE dataLen][data]."""
    (cmd_len,) = struct.unpack_from("<i", body, 0)
    cmd = body[4:4 + cmd_len].decode("utf-8")
    (data_len,) = struct.unpack_from("<i", body, 4 + cmd_len)
    data = body[8 + cmd_len:8 + cmd_len + data_len]
    return cmd, data


def parse_camera_request(data):
    """CameraRequestData: magic 0xCAFE, version 1, 7x int32 LE, 2 length-prefixed strings."""
    if len(data) < 3 or data[0] != 0xCA or data[1] != 0xFE:
        raise ValueError("bad magic bytes")
    if data[2] != 1:
        raise ValueError(f"unsupported protocol version {data[2]}")
    w, h, fps, bitrate, mv_hevc, render_mode, port = struct.unpack_from("<7i", data, 3)
    off = 3 + 28
    cam_len = data[off]; camera = data[off + 1:off + 1 + cam_len].decode("utf-8")
    off += 1 + cam_len
    ip_len = data[off]; ip = data[off + 1:off + 1 + ip_len].decode("utf-8")
    return {"width": w, "height": h, "fps": fps, "bitrate": bitrate,
            "mv_hevc": mv_hevc, "render_mode": render_mode, "port": port,
            "camera": camera, "ip": ip}


def build_open_camera(width, height, fps, bitrate, port, camera, ip):
    """The Quest's OPEN_CAMERA message, for tests (mirror of the two C# serializers)."""
    cam_b, ip_b = camera.encode(), ip.encode()
    req = bytes([0xCA, 0xFE, 1]) + struct.pack("<7i", width, height, fps, bitrate,
                                               0, 2, port)
    req += bytes([len(cam_b)]) + cam_b + bytes([len(ip_b)]) + ip_b
    cmd = b"OPEN_CAMERA"
    body = struct.pack("<i", len(cmd)) + cmd + struct.pack("<i", len(req)) + req
    return struct.pack(">I", len(body)) + body


# ---------------------------------------------------------------- stereo capture

class StereoGrabber:
    """Threaded capture -> one side-by-side RGB frame (left eye | right eye).

    Two modes: a dual-lens camera that already emits SBS frames (device=N), or two
    separate cameras composed here (left=N, right=M). swap flips the eyes -- if the
    3D looks "inside out" (depth inverted), the eyes are crossed: swap them.
    """

    def __init__(self, device=None, left=None, right=None, swap=False,
                 eye_width=1280, eye_height=720, fps=30):
        self.eye_width, self.eye_height = eye_width, eye_height
        self.swap = swap
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._single = device is not None
        self.caps = []
        if cv2 is None:
            print("[stereo] cv2 unavailable -> no camera capture", flush=True)
            return
        devices = [device] if self._single else [left, right]
        if any(d is None for d in devices):
            return
        for dev in devices:
            cap = cv2.VideoCapture(dev)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 30fps over USB
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, eye_width * (2 if self._single else 1))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, eye_height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # latest frame, low latency
            if cap.isOpened():
                self.caps.append(cap)
                print(f"[stereo] /dev/video{dev} open", flush=True)
            else:
                print(f"[stereo] /dev/video{dev} FAILED to open", flush=True)
                cap.release()
        need = 1 if self._single else 2
        if len(self.caps) != need:
            for cap in self.caps:
                cap.release()
            self.caps = []

    @property
    def ok(self):
        return bool(self.caps)

    def start(self):
        if self.caps:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        ew, eh = self.eye_width, self.eye_height
        eyes = [np.zeros((eh, ew, 3), np.uint8) for _ in range(2)]
        while not self._stop.is_set():
            if self._single:
                ok, bgr = self.caps[0].read()
                if not ok:
                    continue
                if bgr.shape[1] != 2 * ew or bgr.shape[0] != eh:
                    bgr = cv2.resize(bgr, (2 * ew, eh))
                eyes[0], eyes[1] = bgr[:, :ew], bgr[:, ew:]
            else:
                for i, cap in enumerate(self.caps):
                    ok, bgr = cap.read()
                    if ok:
                        eyes[i] = cv2.resize(bgr, (ew, eh))
            sbs = np.hstack((eyes[1], eyes[0]) if self.swap else (eyes[0], eyes[1]))
            rgb = cv2.cvtColor(sbs, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest = np.ascontiguousarray(rgb)

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for cap in self.caps:
            cap.release()


def test_pattern(width, height, t):
    """Synthetic SBS frame with real disparity: the square floats in front of the
    grid in 3D mode. Proves the whole Quest path with no camera attached."""
    ew = width // 2
    eye = np.zeros((height, ew, 3), np.uint8)
    eye[:] = (24, 24, 32)
    for x in range(0, ew, 80):                        # grid at screen depth
        eye[:, x:x + 2] = (70, 70, 80)
    for y in range(0, height, 80):
        eye[y:y + 2, :] = (70, 70, 80)
    left, right = eye.copy(), eye.copy()
    cx = int(ew / 2 + (ew / 4) * np.sin(t * 0.8))     # slow horizontal sweep
    cy, half, disp = height // 2, 60, 24              # disp px crossed = pops forward
    left[cy - half:cy + half, cx - half + disp // 2:cx + half + disp // 2] = (60, 220, 90)
    right[cy - half:cy + half, cx - half - disp // 2:cx + half - disp // 2] = (60, 220, 90)
    return np.ascontiguousarray(np.hstack([left, right]))


# ---------------------------------------------------------------- server + stream

class StereoVisionServer:
    """LISTEN for the Quest's OPEN_CAMERA, then stream submitted SBS frames back.

    submit() the latest side-by-side RGB frame from any loop; the streaming thread
    H.264-encodes it at the size/fps/bitrate the Quest requested and sends
    [4-byte big-endian length][Annex-B] -- the proven mono wire format.
    """

    def __init__(self, listen_port: int = CONTROL_PORT):
        self.listen_port = listen_port
        self.status = "waiting"        # waiting | streaming
        self.request = None            # dict once OPEN_CAMERA arrives
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._streaming = threading.Event()
        self._serve_thread = threading.Thread(target=self._serve, daemon=True)
        self._stream_thread = None
        self._ctrl = None              # current control conn; newest connection wins
        self._seq = 0                  # bumps per submit; stream sends each frame ONCE
        self._submit_t = 0.0           # monotonic time of the latest submit (queue-age probe)
        self.lat = LatencyStats()      # video-path stage timings, reported from _stream

    def start(self):
        self._serve_thread.start()

    def submit(self, rgb):
        with self._lock:
            self._latest = rgb
            self._seq += 1
            self._submit_t = time.monotonic()

    def stop(self):
        self._stop.set()
        self._streaming.clear()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
        self._serve_thread.join(timeout=2.0)

    # -- control: the Quest dials us and asks for the camera
    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", self.listen_port))
        except OSError as e:
            print(f"[stereo] FATAL: cannot bind :{self.listen_port} ({e}) — "
                  f"another eval/stereo_sender running?", flush=True)
            return
        srv.listen(2)
        srv.settimeout(1.0)
        print(f"[stereo] control: listening on :{self.listen_port} "
              f"(Quest: ZEDMINI source -> Listen -> this host's IP)", flush=True)
        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            # Newest connection WINS. A network hop leaves the previous control session
            # half-dead but ESTABLISHED forever (no FIN ever arrives) — blocking on it
            # would deadlock every future panel-open (seen live: green screen at a cafe).
            old, self._ctrl = self._ctrl, conn
            if old is not None:
                print("[stereo] control: new connection preempts the old one", flush=True)
                try:
                    old.close()                       # unblocks the old session's read
                except OSError:
                    pass
            threading.Thread(target=self._control_session, args=(conn, addr[0]),
                             daemon=True).start()
        srv.close()

    def _control_session(self, conn, peer):
        print(f"[stereo] control: Quest connected from {peer}", flush=True)
        try:
            while not self._stop.is_set():
                body = read_framed(conn)
                if body is None:
                    break
                try:
                    cmd, data = parse_protocol(body)
                except (struct.error, UnicodeDecodeError, IndexError) as e:
                    print(f"[stereo] control: unparseable message ({e})", flush=True)
                    continue
                if cmd == "OPEN_CAMERA":
                    try:
                        req = parse_camera_request(data)
                    except (ValueError, struct.error, IndexError) as e:
                        print(f"[stereo] bad OPEN_CAMERA: {e}", flush=True)
                        continue
                    print(f"[stereo] OPEN_CAMERA: {req['width']}x{req['height']}"
                          f"@{req['fps']} {req['bitrate']}bps type={req['camera']}"
                          f" -> send to {req['ip']}:{req['port']}", flush=True)
                    self.request = req
                    self._restart_stream()
                elif cmd == "CLOSE_CAMERA":
                    print("[stereo] CLOSE_CAMERA", flush=True)
                    self._streaming.clear()
                else:
                    print(f"[stereo] control: unknown command {cmd!r}", flush=True)
        except OSError:
            pass                                      # closed under us (preempted) or peer reset
        finally:
            try:
                conn.close()
            except OSError:
                pass
            if self._ctrl is conn:                    # only the CURRENT session stops the video
                self._streaming.clear()
                self.status = "waiting"
            print(f"[stereo] control: {peer} disconnected", flush=True)

    def _restart_stream(self):
        self._streaming.clear()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
        self._streaming.set()
        self._stream_thread = threading.Thread(target=self._stream, daemon=True)
        self._stream_thread.start()

    # -- video: we dial the Quest's decoder and push frames
    def _stream(self):
        import av
        req = self.request
        w, h, fps = req["width"], req["height"], max(req["fps"], 1)
        period = 1.0 / fps
        pts = 0
        while self._streaming.is_set() and not self._stop.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            # Small send buffer: if the Quest dozes (app freezes, TCP keeps ACKing) we must
            # DROP frames, not queue minutes of backlog the decoder then plays time-shifted.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
            try:
                sock.connect((req["ip"], req["port"]))
            except OSError:
                sock.close()
                time.sleep(0.5)                        # decoder not up yet -> retry
                continue
            sock.settimeout(5.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            enc = av.CodecContext.create("libx264", "w")  # fresh -> leads SPS/PPS + IDR
            enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
            enc.time_base = Fraction(1, fps)
            enc.bit_rate = req["bitrate"] or 4_000_000
            enc.options = {"preset": "ultrafast", "tune": "zerolatency",
                           "g": "15", "profile": "baseline"}
            self.status = "streaming"
            print(f"[stereo] video: streaming {w}x{h}@{fps} to "
                  f"{req['ip']}:{req['port']}", flush=True)
            try:
                sent_seq = -1
                last_lat = time.monotonic()
                while self._streaming.is_set() and not self._stop.is_set():
                    with self._lock:
                        rgb, seq, t_sub = self._latest, self._seq, self._submit_t
                    if rgb is None or seq == sent_seq:  # nothing NEW -> send nothing.
                        time.sleep(0.002)               # re-sending duplicates doubles the
                        continue                        # decode load -> Quest-side queue -> lag
                    if seq - sent_seq > 1 and sent_seq >= 0:   # frames superseded before encode
                        for _ in range(seq - sent_seq - 1):
                            self.lat.count("vid_skip")
                    sent_seq = seq
                    _, writable, _ = select.select([], [sock], [], 0)
                    if not writable:                   # receiver stalled -> drop this frame
                        self.lat.count("vid_drop")
                        time.sleep(period)
                        continue
                    t1 = time.monotonic()
                    self.lat.note("vid_queue", t1 - t_sub)     # submit -> encode start
                    if rgb.shape[1] != w or rgb.shape[0] != h:
                        rgb = cv2.resize(rgb, (w, h)) if cv2 is not None else rgb
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame.pts = pts
                    pts += 1
                    pkts = [bytes(p) for p in enc.encode(frame)]
                    t2 = time.monotonic()
                    self.lat.note("vid_encode", t2 - t1)
                    for b in pkts:
                        sock.sendall(struct.pack(">I", len(b)) + b)
                    t3 = time.monotonic()
                    self.lat.note("vid_send", t3 - t2)
                    if t3 - last_lat >= 5.0:
                        last_lat = t3
                        rep = self.lat.report()
                        if rep:
                            print(f"[lat] stereo {rep}", flush=True)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"[stereo] video: dropped ({e})", flush=True)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
        self.status = "waiting"
        print("[stereo] video: stream stopped", flush=True)


# ---------------------------------------------------------------- standalone CLI

def main(listen_port: int = CONTROL_PORT,
         device: Optional[int] = None, left: Optional[int] = None,
         right: Optional[int] = None, swap: bool = False,
         eye_width: int = 1280, eye_height: int = 720, fps: int = 30,
         snapshot: Optional[str] = None):
    grabber = StereoGrabber(device=device, left=left, right=right, swap=swap,
                            eye_width=eye_width, eye_height=eye_height, fps=fps)
    if snapshot:
        if not grabber.ok:
            print("[stereo] no camera for --snapshot", flush=True)
            return
        grabber.start()
        deadline = time.monotonic() + 5.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = grabber.latest()
            time.sleep(0.05)
        grabber.stop()
        if frame is None:
            print("[stereo] no frame within 5s", flush=True)
            return
        cv2.imwrite(snapshot, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[stereo] wrote {snapshot} ({frame.shape[1]}x{frame.shape[0]} SBS)", flush=True)
        return

    if grabber.ok:
        grabber.start()
        print("[stereo] source: camera(s)", flush=True)
    else:
        print("[stereo] source: synthetic disparity pattern (no camera args)", flush=True)
    server = StereoVisionServer(listen_port)
    server.start()
    t0 = time.monotonic()
    try:
        while True:
            frame = grabber.latest() if grabber.ok else \
                test_pattern(2 * eye_width, eye_height, time.monotonic() - t0)
            if frame is not None:
                server.submit(frame)
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        grabber.stop()


if __name__ == "__main__":
    tyro.cli(main)
