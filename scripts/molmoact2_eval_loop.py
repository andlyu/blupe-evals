#!/usr/bin/env python3
"""Run an eval command once per new checkpoint and append JSONL metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _checkpoint_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _load_seen(metrics_path: Path) -> set[str]:
    seen = set()
    if not metrics_path.exists():
        return seen
    with metrics_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            checkpoint = row.get("checkpoint")
            if checkpoint:
                seen.add(str(checkpoint))
    return seen


def _append_metric(metrics_path: Path, row: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _run_eval(command: str, checkpoint: Path, variant: str, timeout_s: float) -> dict[str, object]:
    env = os.environ.copy()
    env["MOLMOACT2_CHECKPOINT"] = str(checkpoint)
    env["MOLMOACT2_VARIANT"] = variant
    started = time.time()
    proc = subprocess.run(
        command,
        shell=True,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s if timeout_s > 0 else None,
    )
    elapsed_s = round(time.time() - started, 3)
    return {
        "checkpoint": str(checkpoint),
        "variant": variant,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_s": elapsed_s,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--metrics", required=True, help="JSONL metrics output.")
    parser.add_argument("--command", required=True, help="Eval command. Receives MOLMOACT2_CHECKPOINT and MOLMOACT2_VARIANT.")
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=0.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = Path(args.checkpoints_root).expanduser()
    metrics_path = Path(args.metrics).expanduser()
    while True:
        seen = _load_seen(metrics_path)
        pending = [path for path in _checkpoint_dirs(root) if str(path) not in seen]
        for checkpoint in pending:
            row = _run_eval(args.command, checkpoint, args.variant, args.timeout_s)
            _append_metric(metrics_path, row)
            print(json.dumps(row, indent=2), flush=True)
        if args.once:
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
