"""Serve: the real YAM (i2rt) follows a streamed joint command (the eval's CONNECT).

Sim and real are fed the SAME joints; this is the real-arm consumer. Runs in the i2rt env on
the robot host (separate from the eval's conda `xr`):

    PYTHONPATH=$HOME/i2rt $HOME/yam-venv/bin/python scripts/orin/yam_real_serve.py --channel can0

Wire (newline-delimited JSON, matches blupe-eval-console/link.py):
  server->client once:  {"start_joints": [6]}          (no-jump seed)
  client->server tick:  {"q": [6], "g": 0..1, "t": s}  (arm joints + gripper; g 0=closed 1=open;
                                                        t = OPTIONAL sender monotonic timestamp)
  server->client ack:   {"ack": t}                      (echo of t AFTER the command is applied —
                                                        the sender computes RTT on ITS OWN clock;
                                                        only sent when t is present)
  client->server quit:  {"shutdown": true}              (-> torque OFF)

Gripper: i2rt's gripper is NORMALIZED [0,1] (0=closed, 1=open) — never raw motor radians. We follow
TRI raiden's pattern: each tick walk the gripper a small BOUNDED step (GRIPPER_LEAD) toward the
desired g. A blocked jaw stalls, so the command can't lead the actual pos by more than GRIPPER_LEAD
-> bounded force -> no grind, no crush.

Safety: every command is velocity-clamped (MAX_VEL rad/s, robot-side backstop); a dropped
connection holds the last pose; shutdown / Ctrl-C cuts motor torque (full limp).
"""

import argparse
import json
import socket
import time

import numpy as np

MAX_VEL = 0.6          # rad/s hard cap on the real arm (independent of the eval's own cap)
N = 6                  # arm joints (gripper is index N)
GRIPPER_LEAD = 0.9 * 6 / 71   # max [0,1] the gripper target may lead its actual pos (raiden ~0.076)


def disable_motorchain(robot):
    """Full limp on an i2rt MotorChainRobot. ORDER MATTERS: the gravity-comp control thread keeps
    re-commanding the motors, so you must STOP IT FIRST — otherwise motor_off is instantly undone and
    the arm stays stiff (this was the 'arm won't limp' bug). Then motor_off per id, then close."""
    # 1) Stop the control thread first (so it can't re-energize after we cut torque).
    try:
        robot._stop_event.set()
        robot._server_thread.join(timeout=2.0)
    except Exception as e:
        print(f"  stop thread: {e}", flush=True)
    try:
        robot.motor_chain.running = False
    except Exception:
        pass
    # 2) Cut torque per motor (ids 1..len = 6 arm + gripper).
    chain = robot.motor_chain
    try:
        n = len(chain)
    except Exception:
        n = 7
    try:
        mi = chain.motor_interface
        for mid in range(1, n + 1):
            try:
                mi.motor_off(mid)
            except Exception as e:
                print(f"  motor_off {mid} err: {e}", flush=True)
    except Exception as e:
        print(f"  disable err: {e}", flush=True)
    # 3) Close the CAN interface last.
    try:
        chain.close()
    except Exception:
        pass
    print("[off] all motors off — arm is limp.", flush=True)


class _FakeRobot:
    """--fake: no hardware. Stores commands so the stream can be verified end-to-end."""
    def __init__(self):
        self._q = np.zeros(7)
        class _MC:
            running = True
            class _MI:
                def motor_off(self, mid):
                    pass
            motor_interface = _MI()
        self.motor_chain = _MC()

    def get_joint_pos(self):
        return self._q.copy()

    def command_joint_pos(self, cmd):
        self._q = np.asarray(cmd, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--fake", action="store_true", help="no hardware: echo received joints only")
    args = ap.parse_args()

    if args.fake:
        print("[serve] FAKE mode — no hardware, echoing the stream", flush=True)
        robot = _FakeRobot()
    else:
        from i2rt.robots.get_robot import get_yam_robot
        print(f"[serve] connecting real YAM on {args.channel} ...", flush=True)
        robot = get_yam_robot(args.channel)        # i2rt's limits set the gripper's [0,1] normalization

    info = robot.get_robot_info() if hasattr(robot, "get_robot_info") else {}
    gidx = info.get("gripper_index", N)
    print(f"[serve] gripper idx={gidx} (normalized 0=closed 1=open; lead {GRIPPER_LEAD:.3f}/tick)", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"[serve] listening on :{args.port} — arm follows the stream (<= {MAX_VEL} rad/s). "
          f"Ctrl-C = torque off.", flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[serve] client {addr} connected — mirror live", flush=True)
            f = conn.makefile("rwb")
            last = np.asarray(robot.get_joint_pos(), dtype=float)[:N].copy()   # seed at current
            f.write((json.dumps({"start_joints": [float(x) for x in last]}) + "\n").encode())
            f.flush()
            last_t = None
            nrx = 0
            # latency probes (all on the Orin clock): inter-command gap = network+sender
            # jitter as seen here; apply = i2rt command_joint_pos duration.
            gap_sum, gap_max, app_sum, app_max, nlat = 0.0, 0.0, 0.0, 0.0, 0
            last_lat = time.monotonic()
            try:
                for line in f:
                    if not line.strip():
                        continue
                    msg = json.loads(line.decode())
                    if msg.get("shutdown"):
                        print("[serve] shutdown -> torque off", flush=True)
                        disable_motorchain(robot)
                        return
                    q = msg.get("q")
                    if q is None:
                        continue
                    now = time.monotonic()
                    if last_t is not None:
                        gap = now - last_t
                        gap_sum += gap
                        gap_max = max(gap_max, gap)
                        nlat += 1
                    dt = 0.02 if last_t is None else min(max(now - last_t, 1e-4), 0.1)
                    last_t = now
                    step = MAX_VEL * dt
                    last = last + np.clip(np.asarray(q[:N], dtype=float) - last, -step, step)
                    cmd = np.asarray(robot.get_joint_pos(), dtype=float).copy()
                    cmd[:N] = last
                    g = msg.get("g")
                    if g is not None and gidx is not None and len(cmd) > gidx:
                        desired = min(max(float(g), 0.0), 1.0)         # 0=closed .. 1=open (normalized)
                        actual = float(cmd[gidx])                       # current gripper, normalized [0,1]
                        if desired > actual:                           # raiden bounded-lead walk: no grind/crush
                            cmd[gidx] = min(desired, actual + GRIPPER_LEAD)
                        else:
                            cmd[gidx] = max(desired, actual - GRIPPER_LEAD)
                    robot.command_joint_pos(cmd)
                    t_applied = time.monotonic()
                    app = t_applied - now
                    app_sum += app
                    app_max = max(app_max, app)
                    t_cmd = msg.get("t")
                    if t_cmd is not None:              # ack AFTER apply -> sender-side RTT probe
                        f.write((json.dumps({"ack": t_cmd}) + "\n").encode())
                        f.flush()
                    nrx += 1
                    if nrx % 25 == 0:
                        print(f"[serve] {nrx} cmds; applied joints {np.round(last, 3)}", flush=True)
                    if nlat and t_applied - last_lat >= 5.0:
                        print(f"[lat] serve gap={gap_sum / nlat * 1e3:.1f}/{gap_max * 1e3:.1f}ms "
                              f"apply={app_sum / nlat * 1e3:.1f}/{app_max * 1e3:.1f}ms "
                              f"({nlat / (t_applied - last_lat):.0f}cmd/s)", flush=True)
                        gap_sum, gap_max, app_sum, app_max, nlat = 0.0, 0.0, 0.0, 0.0, 0
                        last_lat = t_applied
            except Exception as e:
                print(f"[serve] stream ended: {e}", flush=True)
            finally:
                conn.close()
                print("[serve] client disconnected — holding last pose", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        disable_motorchain(robot)
        srv.close()


if __name__ == "__main__":
    main()
