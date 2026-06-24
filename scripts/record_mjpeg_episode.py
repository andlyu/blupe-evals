from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_period = 1.0 / args.fps if args.fps > 0 else 0.0
    end_t = time.monotonic() + args.duration_s
    next_save_t = 0.0
    saved = 0
    buf = bytearray()

    with urllib.request.urlopen(args.url, timeout=5) as resp:
        while time.monotonic() < end_t:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                start = buf.find(b"\xff\xd8")
                if start < 0:
                    if len(buf) > 1024 * 1024:
                        del buf[:-2]
                    break
                end = buf.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if start > 0:
                        del buf[:start]
                    break
                frame = bytes(buf[start : end + 2])
                del buf[: end + 2]
                now = time.monotonic()
                if now >= next_save_t:
                    (out_dir / f"frame_{saved:05d}.jpg").write_bytes(frame)
                    saved += 1
                    next_save_t = now + min_period
                if time.monotonic() >= end_t:
                    break

    print(f"saved={saved} out_dir={out_dir}")


if __name__ == "__main__":
    main()
