#!/usr/bin/env python3
"""Probe OpenCV camera access for Mac-local LeLab recording."""

from __future__ import annotations

import argparse
import json
import platform


def backend_id(name: str):
    import cv2

    aliases = {
        "ANY": cv2.CAP_ANY,
        "AVFOUNDATION": cv2.CAP_AVFOUNDATION,
        "V4L2": cv2.CAP_V4L2,
    }
    return aliases[name.upper()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, action="append", default=None)
    parser.add_argument("--backend", default="AVFOUNDATION" if platform.system() == "Darwin" else "ANY")
    args = parser.parse_args(argv)
    if args.camera is None:
        args.camera = [0, 1, 2]
    return args


def main() -> int:
    args = parse_args()

    import cv2

    backend = backend_id(args.backend)
    results = []
    ok_all = True
    for index in args.camera:
        cap = cv2.VideoCapture(index, backend)
        opened = cap.isOpened()
        ok, frame = cap.read() if opened else (False, None)
        cap.release()
        ok_all = ok_all and opened and ok
        results.append(
            {
                "index": index,
                "backend": args.backend.upper(),
                "opened": opened,
                "read": ok,
                "shape": None if frame is None else list(frame.shape),
            }
        )

    print(json.dumps({"ok": ok_all, "cameras": results}, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
