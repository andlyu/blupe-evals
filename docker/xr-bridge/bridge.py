"""XR input bridge — the container side of docs/XR-INPUT-BRIDGE.md (protocol v1).

Polls the real xrobotoolkit_sdk at RATE_HZ and broadcasts newline-JSON ticks to every
connected client on 0.0.0.0:8765. Design points:
- The socket server starts FIRST and serves null ticks until xrt.init() succeeds (a hung
  or crashing service can't take the protocol down with it — the #1-risk failure mode).
- Every getter is wrapped: an SDK error -> null for that field, never a dead bridge.
- Slow clients are dropped (latest-tick semantics; consumers reconnect).
"""

import json
import socket
import threading
import time

RATE_HZ = 60
PORT = 8765

_xrt = None                      # set by the background init thread once the SDK is up
_clients = set()                 # sockets
_clients_lock = threading.Lock()


def _init_sdk_forever():
    global _xrt
    while _xrt is None:
        try:
            import xrobotoolkit_sdk as xrt
            xrt.init()
            _xrt = xrt
            print("[bridge] xrobotoolkit_sdk init OK", flush=True)
        except Exception as e:
            print(f"[bridge] sdk init failed ({e}); retrying in 2s", flush=True)
            time.sleep(2.0)


def _safe(fn, cast=None):
    try:
        v = fn()
        return cast(v) if cast and v is not None else v
    except Exception:
        return None


def _tick():
    x = _xrt
    t = time.monotonic()
    if x is None:
        return {"t": t, "right": None, "left": None, "head": None}
    return {
        "t": t,
        "right": {
            "pose": _safe(x.get_right_controller_pose, list),
            "grip": _safe(x.get_right_grip, float),
            "trigger": _safe(x.get_right_trigger, float),
            "axis": _safe(x.get_right_axis, list),
            "axis_click": _safe(x.get_right_axis_click, bool),
            "A": _safe(x.get_A_button, bool),
            "B": _safe(x.get_B_button, bool),
            "menu": _safe(x.get_right_menu_button, bool),
        },
        "left": {
            "pose": _safe(x.get_left_controller_pose, list),
            "grip": _safe(x.get_left_grip, float),
            "trigger": _safe(x.get_left_trigger, float),
            "axis": _safe(x.get_left_axis, list),
            "axis_click": _safe(x.get_left_axis_click, bool),
            "X": _safe(x.get_X_button, bool),
            "Y": _safe(x.get_Y_button, bool),
            "menu": _safe(x.get_left_menu_button, bool),
        },
        "head": {"pose": _safe(x.get_headset_pose, list)},
    }


def _accept_loop(srv):
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            conn.sendall((json.dumps({"hello": {"v": 1, "rate_hz": RATE_HZ}}) + "\n").encode())
        except OSError:
            conn.close()
            continue
        with _clients_lock:
            _clients.add(conn)
        print(f"[bridge] client {addr} connected ({len(_clients)} total)", flush=True)


def main():
    threading.Thread(target=_init_sdk_forever, daemon=True).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(4)
    threading.Thread(target=_accept_loop, args=(srv,), daemon=True).start()
    print(f"[bridge] serving ticks on :{PORT} at {RATE_HZ} Hz "
          f"(null until the SDK is up)", flush=True)

    period = 1.0 / RATE_HZ
    n = 0
    while True:
        t0 = time.monotonic()
        line = (json.dumps(_tick()) + "\n").encode()
        dead = []
        with _clients_lock:
            conns = list(_clients)
        for c in conns:
            try:
                c.sendall(line)            # TCP_NODELAY; a stalled client raises/blocks -> drop
            except OSError:
                dead.append(c)
        for c in dead:
            with _clients_lock:
                _clients.discard(c)
            c.close()
            print(f"[bridge] dropped client ({len(_clients)} left)", flush=True)
        n += 1
        if n % (RATE_HZ * 10) == 0:
            sdk = "up" if _xrt is not None else "DOWN"
            print(f"[bridge] alive: sdk={sdk} clients={len(_clients)}", flush=True)
        dt = period - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    main()
