"""Drop-in xrobotoolkit_sdk with selectable backends — the input seam (PLAN Part 4).

This file lives in scripts/, which is sys.path[0] when running the eval scripts, so it
SHADOWS the real compiled xrobotoolkit_sdk everywhere (including the Orin). Backend is
chosen with the XR_INPUT env var:

  XR_INPUT=sdk     delegate every call to the REAL module found later on sys.path
                   (the Orin path; default when the real module exists)
  XR_INPUT=bridge  read XR state from the container bridge per docs/XR-INPUT-BRIDGE.md
                   (the Mac path; XR_BRIDGE_HOST/XR_BRIDGE_PORT, default 127.0.0.1:8765)
  XR_INPUT=stub    deterministic scripted session for headless tests (no Quest, no
                   container): press A (TELEOP), clutch + circle the EE, nav the menu to
                   CONNECT, clutch + circle again with a trigger toggle, then B (GO_HOME).

The API surface mirrors exactly what xrobotoolkit_teleop/common/xr_client.py consumes.
Poses are [x, y, z, qx, qy, qz, qw] (identity = IDENT). Consumers tolerate "no data":
grip/trigger 0.0, buttons False, identity poses (see the staleness rule in the spec).
"""

import importlib.machinery
import json
import math
import os
import socket
import sys
import threading
import time

IDENT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# backend: sdk (delegate to the real compiled module, skipping this file's dir)
# --------------------------------------------------------------------------- #
def _load_real_sdk():
    paths = [p for p in sys.path if os.path.abspath(p or ".") != _THIS_DIR]
    spec = importlib.machinery.PathFinder.find_spec("xrobotoolkit_sdk", paths)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# backend: bridge (TCP client of the container bridge — docs/XR-INPUT-BRIDGE.md)
# --------------------------------------------------------------------------- #
class _Bridge:
    STALE_S = 0.5            # spec: no tick for >0.5s -> "input lost" (reads as clutch released)

    def __init__(self, host, port):
        self.host, self.port = host, port
        self._tick = None            # latest decoded tick (dict)
        self._rx_t = 0.0             # local monotonic receive time (bridge t is not comparable)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.host, self.port), timeout=3.0)
                s.settimeout(3.0)
                f = s.makefile("rb")
                hello = json.loads(f.readline().decode())
                print(f"[xr-bridge] connected {self.host}:{self.port} hello={hello.get('hello')}",
                      flush=True)
                for line in f:
                    if self._stop.is_set():
                        break
                    if not line.strip():
                        continue
                    tick = json.loads(line.decode())
                    with self._lock:
                        self._tick = tick
                        self._rx_t = time.monotonic()
            except (OSError, ValueError) as e:
                if not self._stop.is_set():
                    print(f"[xr-bridge] {e}; retrying", flush=True)
                    self._stop.wait(1.0)

    def close(self):
        self._stop.set()

    def _side(self, name):
        """Latest tick's controller dict, or {} when missing/stale (=> null semantics)."""
        with self._lock:
            tick, rx = self._tick, self._rx_t
        if tick is None or (time.monotonic() - rx) > self.STALE_S:
            return {}
        return tick.get(name) or {}

    def pose(self, name):
        p = self._side(name).get("pose")
        return list(p) if p else list(IDENT)

    def analog(self, side, key):
        v = self._side(side).get(key)
        return 0.0 if v is None else float(v)

    def button(self, side, key):
        return bool(self._side(side).get(key) or False)

    def axis(self, side):
        a = self._side(side).get("axis")
        return [float(a[0]), float(a[1])] if a else [0.0, 0.0]

    def time_ns(self):
        with self._lock:
            tick = self._tick
        t = (tick or {}).get("t")
        return int((t if t is not None else time.monotonic()) * 1e9)

    def input_age_s(self):
        """Seconds since the latest tick ARRIVED (local clock — bridge t isn't comparable)."""
        with self._lock:
            rx = self._rx_t
        return time.monotonic() - rx if rx else 0.0


# --------------------------------------------------------------------------- #
# backend: stub (deterministic scripted session for headless tests)
# --------------------------------------------------------------------------- #
class _Stub:
    """Timeline (seconds since init): exercises every input path the eval uses.
      1.0-1.3   A pressed            -> state TELEOP
      2.0-8.0   grip=1 + slow EE circle (5 cm, 4 s period)    -> IK drives the sim
      9.0-10.4  three right-tilts of the stick (re-armed via deadzone) -> highlight CONNECT
      10.8-11.1 stick click          -> CONNECT toggles on (RobotLink dials the serve)
      12.0-18.0 grip=1 + circle again -> joints stream to the serve
      13.0-13.3 right trigger pressed -> gripper toggles (g flips in the stream)
      19.0-19.3 B pressed            -> GO_HOME eases back
    """

    def __init__(self):
        self.t0 = None                # set on FIRST poll, not init(): model loading takes seconds

    def _t(self):
        if self.t0 is None:
            self.t0 = time.monotonic()
        return time.monotonic() - self.t0

    @staticmethod
    def _within(t, a, b):
        return a <= t < b

    def pose(self, name):
        t = self._t()
        if name != "right":
            return list(IDENT)
        x, y, z = 0.0, 0.0, 0.0
        for a, b in ((2.0, 8.0), (12.0, 18.0)):
            if self._within(t, a, b):
                ph = 2.0 * math.pi * (t - a) / 4.0
                x, z = 0.05 * math.cos(ph) - 0.05, 0.05 * math.sin(ph)
        return [x, y, z, 0.0, 0.0, 0.0, 1.0]

    def grip(self, side):
        t = self._t()
        on = side == "right" and (self._within(t, 2.0, 8.0) or self._within(t, 12.0, 18.0))
        return 1.0 if on else 0.0

    def trigger(self, side):
        return 1.0 if (side == "right" and self._within(self._t(), 13.0, 13.3)) else 0.0

    def button(self, key):
        t = self._t()
        if key == "A":
            return self._within(t, 1.0, 1.3)
        if key == "B":
            return self._within(t, 19.0, 19.3)
        if key == "right_axis_click":
            return self._within(t, 10.8, 11.1)
        return False

    def axis(self, side):
        if side != "right":
            return [0.0, 0.0]
        t = self._t()
        for k in range(3):                       # three tilt pulses, deadzone gaps re-arm the nav
            if self._within(t, 9.0 + 0.5 * k, 9.0 + 0.5 * k + 0.2):
                return [1.0, 0.0]
        return [0.0, 0.0]


# --------------------------------------------------------------------------- #
# module API (the exact surface common/xr_client.py imports)
# --------------------------------------------------------------------------- #
_MODE = os.environ.get("XR_INPUT", "").strip().lower()
_impl = None        # _Bridge | _Stub | the real module


def init():
    global _impl, _MODE
    if _impl is not None:
        return
    if _MODE in ("", "sdk"):
        real = _load_real_sdk()
        if real is not None:
            _impl = real
            real.init()
            print("[xr-input] backend: real xrobotoolkit_sdk", flush=True)
            return
        if _MODE == "sdk":
            raise ImportError("XR_INPUT=sdk but no real xrobotoolkit_sdk on sys.path")
        raise ImportError(
            "no real xrobotoolkit_sdk found (this box has no PC Service). "
            "Set XR_INPUT=bridge (container bridge) or XR_INPUT=stub (headless test).")
    if _MODE == "bridge":
        _impl = _Bridge(os.environ.get("XR_BRIDGE_HOST", "127.0.0.1"),
                        int(os.environ.get("XR_BRIDGE_PORT", "8765")))
        print(f"[xr-input] backend: bridge {_impl.host}:{_impl.port}", flush=True)
    elif _MODE == "stub":
        _impl = _Stub()
        print("[xr-input] backend: stub (scripted session)", flush=True)
    else:
        raise ValueError(f"XR_INPUT={_MODE!r}: expected sdk | bridge | stub")


def close():
    global _impl
    if _impl is None:
        return
    if isinstance(_impl, _Bridge):
        _impl.close()
    elif not isinstance(_impl, _Stub):
        _impl.close()                 # real SDK
    _impl = None


def _real():
    if _impl is None:
        init()
    return _impl


def _is_real():
    return not isinstance(_impl, (_Bridge, _Stub))


# -- poses ------------------------------------------------------------------ #
def get_left_controller_pose():
    r = _real()
    return r.get_left_controller_pose() if _is_real() else r.pose("left")


def get_right_controller_pose():
    r = _real()
    return r.get_right_controller_pose() if _is_real() else r.pose("right")


def get_headset_pose():
    r = _real()
    return r.get_headset_pose() if _is_real() else r.pose("head")


# -- analogs ------------------------------------------------------------------ #
def get_left_grip():
    r = _real()
    if _is_real():
        return r.get_left_grip()
    return r.analog("left", "grip") if isinstance(r, _Bridge) else r.grip("left")


def get_right_grip():
    r = _real()
    if _is_real():
        return r.get_right_grip()
    return r.analog("right", "grip") if isinstance(r, _Bridge) else r.grip("right")


def get_left_trigger():
    r = _real()
    if _is_real():
        return r.get_left_trigger()
    return r.analog("left", "trigger") if isinstance(r, _Bridge) else r.trigger("left")


def get_right_trigger():
    r = _real()
    if _is_real():
        return r.get_right_trigger()
    return r.analog("right", "trigger") if isinstance(r, _Bridge) else r.trigger("right")


# -- buttons / axes ----------------------------------------------------------- #
def _btn(side, key, stub_key):
    r = _real()
    if isinstance(r, _Bridge):
        return r.button(side, key)
    return r.button(stub_key)


def get_A_button():
    return _real().get_A_button() if _is_real() else _btn("right", "A", "A")


def get_B_button():
    return _real().get_B_button() if _is_real() else _btn("right", "B", "B")


def get_X_button():
    return _real().get_X_button() if _is_real() else _btn("left", "X", "X")


def get_Y_button():
    return _real().get_Y_button() if _is_real() else _btn("left", "Y", "Y")


def get_left_menu_button():
    return _real().get_left_menu_button() if _is_real() else _btn("left", "menu", "left_menu")


def get_right_menu_button():
    return _real().get_right_menu_button() if _is_real() else _btn("right", "menu", "right_menu")


def get_left_axis_click():
    return _real().get_left_axis_click() if _is_real() else _btn("left", "axis_click", "left_axis_click")


def get_right_axis_click():
    return _real().get_right_axis_click() if _is_real() else _btn("right", "axis_click", "right_axis_click")


def get_left_axis():
    r = _real()
    return r.get_left_axis() if _is_real() else r.axis("left")


def get_right_axis():
    r = _real()
    return r.get_right_axis() if _is_real() else r.axis("right")


# -- misc --------------------------------------------------------------------- #
def get_input_age_s():
    """Latency probe (this shim only, not in the real SDK): seconds since the newest XR
    tick arrived. 0.0 for sdk/stub backends (no transport hop to measure)."""
    r = _real()
    return r.input_age_s() if isinstance(r, _Bridge) else 0.0


def get_time_stamp_ns():
    r = _real()
    if _is_real():
        return r.get_time_stamp_ns()
    return r.time_ns() if isinstance(r, _Bridge) else int(time.monotonic() * 1e9)


# -- hand / motion-tracker / body: not used by our teleop; report "inactive" --- #
def get_left_hand_is_active():
    return _real().get_left_hand_is_active() if _is_real() else False


def get_right_hand_is_active():
    return _real().get_right_hand_is_active() if _is_real() else False


def get_left_hand_tracking_state():
    return _real().get_left_hand_tracking_state() if _is_real() else []


def get_right_hand_tracking_state():
    return _real().get_right_hand_tracking_state() if _is_real() else []


def num_motion_data_available():
    return _real().num_motion_data_available() if _is_real() else 0


def get_motion_tracker_pose():
    return _real().get_motion_tracker_pose() if _is_real() else []


def get_motion_tracker_velocity():
    return _real().get_motion_tracker_velocity() if _is_real() else []


def get_motion_tracker_acceleration():
    return _real().get_motion_tracker_acceleration() if _is_real() else []


def get_motion_tracker_serial_numbers():
    return _real().get_motion_tracker_serial_numbers() if _is_real() else []


def is_body_data_available():
    return _real().is_body_data_available() if _is_real() else False


def get_body_joints_pose():
    return _real().get_body_joints_pose() if _is_real() else []


def get_body_joints_velocity():
    return _real().get_body_joints_velocity() if _is_real() else []


def get_body_joints_acceleration():
    return _real().get_body_joints_acceleration() if _is_real() else []
