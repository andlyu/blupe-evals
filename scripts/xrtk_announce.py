"""Fake XRoboToolkit PC-Service discovery announce — kills the in-VR IP typing.

The Quest app ALREADY auto-discovers the PC service: it listens on UDP :29888 and pops a
one-click IP-select dialog (UIUdpReceiver.cs). The real service broadcasts its IP every 5 s
(tcpserverworker.cpp::onBraodCastTCPServer) — but ours runs inside Docker on macOS, so the
broadcast (a) stays on the container subnet and (b) would advertise the container IP anyway.
This script replays the same announce packet natively on the Mac, advertising the Mac's LAN
IP (whose published :63901 reaches the container). Headset flow becomes: launch app -> tap
the popped IP button. No keyboard.

Wire format (Manage_global.h, #pragma pack(1); multi-byte fields little-endian):
  [0]    0xCF   head            [1]     0x7E  cmd (UDP_PACKET_CMD_TCPIP)
  [2:6]  uint32 LE ip length    [6:6+N] ascii ip string
  [..+8] uint64 LE epoch secs (ignored by client)        [last] 0xA5 tail (not validated)
Details + client-side parse rules: docs/refs/xrobotoolkit/discovery-announce.md.

Run (broadcast + belt-and-braces unicast to the Quest, which binds 0.0.0.0:29888):
    python scripts/xrtk_announce.py --unicast 192.168.0.30
"""

import socket
import struct
import time
from typing import Optional

import tyro

PORT = 29888          # BROADCAST_UDP_PORT (Manage_global.h:15)


def build_packet(ip: str) -> bytes:
    payload = ip.encode("ascii")
    return (bytes([0xCF, 0x7E]) + struct.pack("<I", len(payload)) + payload
            + struct.pack("<Q", int(time.time())) + bytes([0xA5]))


def lan_ip(target: str = "8.8.8.8") -> str:
    """The Mac's source IP for reaching target (no traffic actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 53))
        return s.getsockname()[0]
    finally:
        s.close()


def iface_broadcast(ip: str):
    """The REAL broadcast address of the interface owning ip (ifconfig), or None.
    The /24 guess (x.y.z.255) is wrong on wider subnets — Starbucks was a /22."""
    import re
    import subprocess
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    m = re.search(rf"inet {re.escape(ip)} netmask \S+ broadcast (\S+)", out)
    return m.group(1) if m else None


def main(ip: Optional[str] = None, unicast: list[str] = [],
         interval: float = 5.0, port: int = PORT):
    """ip: address to advertise (default: this host's LAN IP). unicast: also send straight
    to these hosts (e.g. the Quest) in case the AP/Android filters subnet broadcasts."""
    ip = ip or lan_ip(unicast[0] if unicast else "8.8.8.8")
    guess24 = ".".join(ip.split(".")[:3] + ["255"])   # /24, same as the real service
    dests = list(dict.fromkeys(                       # dedup, keep order
        [b for b in (iface_broadcast(ip), guess24, "255.255.255.255") if b] + list(unicast)))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"[announce] {ip} -> {', '.join(f'{d}:{port}' for d in dests)} every {interval:.0f}s",
          flush=True)
    while True:
        pkt = build_packet(ip)
        for dest in dests:
            try:
                sock.sendto(pkt, (dest, port))
            except OSError as e:
                print(f"[announce] send to {dest} failed: {e}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    p = build_packet("192.168.0.190")               # self-check vs client Unpack() rules
    assert p[0] == 0xCF and p[1] == 0x7E and p[-1] == 0xA5
    assert struct.unpack("<I", p[2:6])[0] == 13 and p[6:19] == b"192.168.0.190"
    tyro.cli(main)
