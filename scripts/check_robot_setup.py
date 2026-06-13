#!/usr/bin/env python3
"""Robot-side setup checker — run this ON the robot computer; every line is PASS or a fix.

    python3 scripts/check_robot_setup.py --relay 35.185.232.107:8443 --robot yam-2 --token <tok>

STDLIB ONLY (like the agent): works on a bare python3, no venv needed.

Checks, in dependency order:
  1. serve      your serve answers on :5599 with a valid {"start_joints": [...]} handshake
  2. cameras    your camera endpoint serves MJPEG frames on :8089 (each --cam index)
  3. relay      the cloud relay is reachable (one OUTBOUND tcp connect — the only thing
                your site ever needs network-wise)
  4. token      your arm token is accepted (verified via a harmless operator-role hello;
                does NOT disturb a running agent)

Exit 0 = everything green: run the agent one-liner and your arm card flips online.
No commands are sent to the arm; check 1 only reads the handshake (zero motion).
"""

import argparse
import json
import socket
import sys
import time

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"


def result(name, ok, detail, fix=None):
    print(f"  [{OK if ok else BAD}] {name}: {detail}")
    if not ok and fix:
        print(f"         fix: {fix}")
    return ok


def check_serve(host, port):
    try:
        s = socket.create_connection((host, port), timeout=3)
    except OSError as e:
        return result("serve", False, f"nothing listening on {host}:{port} ({e})",
                      "start your serve (e.g. yam_real_serve.py); it must own the motors")
    try:
        s.settimeout(3)
        line = s.makefile().readline()
        s.close()
        msg = json.loads(line)
        q = msg.get("start_joints")
        assert isinstance(q, list) and q and all(isinstance(x, (int, float)) for x in q)
        return result("serve", True, f"handshake OK, {len(q)} joints at "
                      f"{[round(float(x), 2) for x in q]}")
    except Exception as e:
        return result("serve", False, f"connected but no valid start_joints handshake ({e})",
                      "the serve must send {\"start_joints\": [...]} immediately on connect "
                      "(full spec: docs/serve-protocol.md)")


def check_camera(host, port, idx):
    try:
        s = socket.create_connection((host, port), timeout=3)
        s.sendall(f"GET /{idx} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        s.settimeout(4)
        buf = b""
        t0 = time.time()
        while time.time() - t0 < 4:
            buf += s.recv(65536)
            if b"\xff\xd8" in buf and b"\xff\xd9" in buf[buf.find(b"\xff\xd8"):]:
                s.close()
                return result(f"camera /{idx}", True, "MJPEG frames flowing")
        s.close()
        return result(f"camera /{idx}", False, "connected but no JPEG within 4 s",
                      "is the camera device present? check `ls /dev/video*` and the "
                      "--devices flag on camera_relay.py")
    except OSError as e:
        return result(f"camera /{idx}", False, f"nothing on {host}:{port} ({e})",
                      "start it: python3 YAM_control/camera_relay.py --devices 0 2")


def check_relay(relay):
    host, port = relay.rsplit(":", 1)
    try:
        s = socket.create_connection((host, int(port)), timeout=5)
        s.close()
        return result("relay", True, f"outbound {relay} reachable")
    except OSError as e:
        return result("relay", False, f"cannot reach {relay} ({e})",
                      "this is the ONLY network requirement: outbound tcp to the relay. "
                      "check internet/proxy/egress firewall")


def check_token(relay, robot, token):
    host, port = relay.rsplit(":", 1)
    try:
        s = socket.create_connection((host, int(port)), timeout=5)
        # operator-role hello: validates the token WITHOUT registering as the robot
        # (registering would briefly kick a live agent; this never does).
        s.sendall((json.dumps({"role": "operator", "robot": robot,
                               "token": token, "port": 5599}) + "\n").encode())
        s.settimeout(8)
        line = s.makefile().readline()
        s.close()
        msg = json.loads(line) if line.strip() else {}
        if msg.get("err") == "auth":
            return result("token", False, f"relay rejected the token for '{robot}'",
                          "use the exact token from your onboarding one-liner "
                          "(fleet admin can re-issue via Add arm)")
        if msg.get("ok"):
            return result("token", True, f"accepted; agent for '{robot}' is ONLINE "
                          "(an operator channel opened and was closed)")
        return result("token", True, f"accepted ({msg.get('err', 'agent not running yet')} "
                      "— expected before you start the agent)")
    except Exception as e:
        return result("token", False, f"could not verify ({e})",
                      "re-run; if persistent, check the relay address")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--relay", default="35.185.232.107:8443")
    ap.add_argument("--robot", help="your arm id (from onboarding)")
    ap.add_argument("--token", help="your arm token (from onboarding)")
    ap.add_argument("--serve-host", default="127.0.0.1")
    ap.add_argument("--serve-port", type=int, default=5599)
    ap.add_argument("--cam-port", type=int, default=8089)
    ap.add_argument("--cams", type=int, nargs="*", default=[0, 2])
    a = ap.parse_args()

    print("blupe robot-side setup check\n")
    oks = [check_serve(a.serve_host, a.serve_port)]
    oks += [check_camera(a.serve_host, a.cam_port, i) for i in a.cams]
    oks.append(check_relay(a.relay))
    if a.robot and a.token:
        oks.append(check_token(a.relay, a.robot, a.token))
    else:
        print("  [    ] token: skipped (pass --robot and --token to verify your credentials)")

    print()
    if all(oks):
        print("ALL GREEN — start the agent (your onboarding one-liner) and watch your arm "
              "card flip online in the fleet UI.")
        return 0
    print("Fix the FAIL lines above, top to bottom (each depends on the previous), then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
