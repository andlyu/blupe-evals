#!/usr/bin/env python3
"""Periodically pull MolmoAct2 checkpoints from a gstack job with rsync."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _rsync(source: str, dest: Path, delete: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-az", "--partial", "--info=stats1,progress2"]
    if delete:
        cmd.append("--delete")
    cmd.extend([source.rstrip("/") + "/", str(dest) + "/"])
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Remote checkpoint directory, e.g. user@host:/runs/job/checkpoints")
    parser.add_argument("--dest", required=True, help="Local checkpoint mirror directory.")
    parser.add_argument("--interval-s", type=float, default=0.0, help="Repeat interval. Omit/0 for one-shot.")
    parser.add_argument("--delete", action="store_true", help="Mirror deletions from source.")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    while True:
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"[{started}] syncing {args.source} -> {dest}", flush=True)
        _rsync(args.source, dest, delete=args.delete)
        if args.interval_s <= 0:
            return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
