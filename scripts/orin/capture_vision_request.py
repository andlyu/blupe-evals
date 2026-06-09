"""Capture the Quest's StartReceivePcCamera request so we send exactly what it asks for.

On LISTEN (with camera-source IP = this host) the Quest connects to the source's control port
and sends a request. Wire (from XRoboToolkit-Orin-Video-Sender main_zed_tcp.cpp):
    [4-byte BIG-endian total length][NetworkDataProtocol]
    NetworkDataProtocol = [cmdLen i32 LE][cmd bytes][dataLen i32 LE][data bytes]
    data = CameraRequestData = CA FE, version(1B), 7x int32 LE
           (width,height,fps,bitrate,enableMvHevc,renderMode,port), then 2 compact strings
           (1-byte len + bytes): camera, ip

  python scripts/orin/capture_vision_request.py [--port 13579]
"""

import socket
import struct
import tyro


def main(port: int = 13579):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    srv.settimeout(150)
    print(f"listening on 0.0.0.0:{port} — press LISTEN on the Quest now ...", flush=True)
    try:
        conn, addr = srv.accept()
    except socket.timeout:
        print("no connection in 150s — wrong port, or request goes via the PC Service.", flush=True)
        return
    print(f"connection from {addr}", flush=True)
    conn.settimeout(3.0)
    data = b""
    try:
        while len(data) < 65536:
            b = conn.recv(4096)
            if not b:
                break
            data += b
    except socket.timeout:
        pass
    conn.close()
    srv.close()
    print(f"raw {len(data)} bytes: {data[:160].hex()}", flush=True)

    try:
        total = struct.unpack(">I", data[:4])[0]
        proto = data[4:4 + total]
        off = 0
        clen = struct.unpack("<i", proto[off:off + 4])[0]; off += 4
        cmd = proto[off:off + clen].decode(errors="replace"); off += clen
        dlen = struct.unpack("<i", proto[off:off + 4])[0]; off += 4
        cr = proto[off:off + dlen]
        print(f"command: {cmd!r}")
        if len(cr) >= 31 and cr[0] == 0xCA and cr[1] == 0xFE:
            names = ["width", "height", "fps", "bitrate", "enableMvHevc", "renderMode", "port"]
            vals = struct.unpack("<7i", cr[3:31])
            print("CameraRequest:", dict(zip(names, vals)))
            o = 31
            for nm in ("camera", "ip"):
                if o < len(cr):
                    n = cr[o]; o += 1
                    print(f"  {nm}: {cr[o:o+n].decode(errors='replace')}"); o += n
        else:
            print("no CA FE magic — payload hex:", cr[:80].hex())
    except Exception as e:
        print("parse error:", e)


if __name__ == "__main__":
    tyro.cli(main)
