"""Integration test: is the YAM robot being served?

Connects to yam_real_serve.py and reads its no-jump handshake (`start_joints`). That line is only
sent after `get_yam_robot` succeeds, so receiving it proves BOTH: the serve is up AND the real arm
is connected. With --hold it also streams the current pose back for ~2 s (a no-op round-trip) to
exercise the command path.

    python scripts/orin/check_serve.py                 # quick health check
    python scripts/orin/check_serve.py --hold          # also verify the command path

Exit code 0 = robot is being served, 1 = not.
"""

import argparse
import json
import socket
import sys
import time


def check(host: str, port: int, hold: bool) -> int:
    try:
        s = socket.create_connection((host, port), timeout=4.0)
    except OSError as e:
        print(f"✗ NOT served — cannot connect to {host}:{port} ({e}).")
        print("  Start it:  ~/i2rt/.venv/bin/python scripts/orin/yam_real_serve.py --channel can0")
        return 1

    s.settimeout(5.0)
    f = s.makefile("rwb")
    try:
        line = f.readline()
    except OSError as e:
        print(f"✗ serve is up but sent no handshake within 5 s ({e}) — arm may not be connected.")
        return 1
    if not line:
        print("✗ serve closed the connection without a handshake — arm not connected?")
        return 1
    try:
        start = json.loads(line.decode()).get("start_joints")
    except (ValueError, UnicodeDecodeError) as e:
        print(f"✗ bad handshake {line!r} ({e})")
        return 1
    if start is None:
        print(f"✗ handshake had no start_joints: {line!r}")
        return 1

    print(f"✓ ROBOT IS BEING SERVED — start_joints = {[round(x, 4) for x in start]}")

    if hold:
        msg = (json.dumps({"q": start}) + "\n").encode()
        try:
            for _ in range(100):                 # ~2 s at 50 Hz: command current pose (no motion)
                f.write(msg)
                f.flush()
                time.sleep(0.02)
        except OSError as e:
            print(f"✗ command path failed mid-stream ({e})")
            return 1
        print("✓ command round-trip OK — held current pose for 2 s (no motion expected)")

    s.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--hold", action="store_true", help="also stream current pose 2 s (no-op round-trip)")
    args = ap.parse_args()
    return check(args.host, args.port, args.hold)


if __name__ == "__main__":
    sys.exit(main())
