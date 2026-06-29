#!/usr/bin/env python3
"""Register the Blupe SO101 robot in a local Mac LeLab cache.

This copies the calibrated SO101 leader/follower files from the Jetson and
writes the local LeLab robot record with Mac USB serial ports.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_HOST = "192.168.0.185"
DEFAULT_USER = "andrew"
DEFAULT_KEY = "~/.ssh/id_ed25519_jetson_nopass"
DEFAULT_REMOTE_ROOT = "/home/andrew/.cache/huggingface/lerobot"
DEFAULT_LOCAL_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"

LEADER_REL = "calibration/teleoperators/so_leader/blupe_leader.json"
FOLLOWER_REL = "calibration/robots/so_follower/blupe_follower.json"

LEADER_SERIAL = "58FA102515"
FOLLOWER_SERIAL = "58FD016976"

DEFAULT_CAMERAS = [
    ("front", 0),
    ("side", 1),
    ("wrist", 2),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def remote_spec(args: argparse.Namespace, rel_path: str) -> str:
    return f"{args.jetson_user}@{args.jetson_host}:{args.remote_root.rstrip('/')}/{rel_path}"


def ssh_command(args: argparse.Namespace) -> str:
    return f"ssh -i {Path(args.ssh_key).expanduser()} -o IdentitiesOnly=yes -o BatchMode=yes"


def copy_from_jetson(args: argparse.Namespace, rel_path: str, local_root: Path) -> Path:
    dest = local_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-az", "-e", ssh_command(args), remote_spec(args, rel_path), str(dest)])
    return dest


def remote_hash(args: argparse.Namespace, rel_path: str) -> str:
    cmd = [
        "ssh",
        "-i",
        str(Path(args.ssh_key).expanduser()),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        f"{args.jetson_user}@{args.jetson_host}",
        f"sha256sum {args.remote_root.rstrip('/')}/{rel_path}",
    ]
    output = run(cmd).stdout.strip()
    return output.split()[0]


def mac_tty_for(serial: str, fallback: str) -> str:
    matches = sorted(glob.glob(f"/dev/tty.usbmodem*{serial}*"))
    if matches:
        return matches[0]
    matches = sorted(glob.glob(f"/dev/cu.usbmodem*{serial}*"))
    if matches:
        return matches[0]
    return fallback


def parse_cameras(values: list[str]) -> list[dict]:
    cameras = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Camera must be NAME=INDEX, got {value!r}")
        name, index = value.split("=", 1)
        cameras.append(
            {
                "id": name,
                "name": name,
                "type": "opencv",
                "camera_index": int(index),
                "device_id": "",
                "width": 640,
                "height": 360,
                "fps": 30,
                "fourcc": "MJPG",
            }
        )
    return cameras


def write_robot_record(args: argparse.Namespace, local_root: Path) -> Path:
    leader_port = args.leader_port or mac_tty_for(LEADER_SERIAL, f"/dev/tty.usbmodem{LEADER_SERIAL}")
    follower_port = args.follower_port or mac_tty_for(FOLLOWER_SERIAL, f"/dev/tty.usbmodem{FOLLOWER_SERIAL}")
    cameras = parse_cameras(args.camera)
    record = {
        "name": args.robot_name,
        "leader_port": leader_port,
        "follower_port": follower_port,
        "leader_config": Path(LEADER_REL).name,
        "follower_config": Path(FOLLOWER_REL).name,
        "cameras": cameras,
    }
    robots_dir = local_root / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)
    path = robots_dir / f"{args.robot_name}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jetson-host", default=DEFAULT_HOST)
    parser.add_argument("--jetson-user", default=DEFAULT_USER)
    parser.add_argument("--ssh-key", default=DEFAULT_KEY)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--robot-name", default="blupe_so101")
    parser.add_argument("--leader-port")
    parser.add_argument("--follower-port")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera mapping as NAME=INDEX. May be repeated.",
    )
    parser.add_argument("--skip-copy", action="store_true", help="Use existing local calibration files.")
    args = parser.parse_args()

    if args.camera is None:
        args.camera = [f"{name}={index}" for name, index in DEFAULT_CAMERAS]

    local_root = Path(args.local_root).expanduser()
    copied = []
    for rel_path in (LEADER_REL, FOLLOWER_REL):
        local_path = local_root / rel_path
        if not args.skip_copy:
            local_path = copy_from_jetson(args, rel_path, local_root)
        if not local_path.exists():
            print(f"missing local calibration: {local_path}", file=sys.stderr)
            return 1
        copied.append((rel_path, local_path, sha256(local_path)))

    if not args.skip_copy:
        for rel_path, local_path, local_digest in copied:
            jetson_digest = remote_hash(args, rel_path)
            if local_digest != jetson_digest:
                print(f"hash mismatch for {rel_path}: local={local_digest} jetson={jetson_digest}", file=sys.stderr)
                return 1
            print(f"matched {rel_path}: {local_digest}")

    record_path = write_robot_record(args, local_root)
    print(f"wrote {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
