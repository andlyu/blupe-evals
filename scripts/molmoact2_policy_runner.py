#!/usr/bin/env python3
"""HTTP policy runner shim for MolmoAct2.

This process owns the MolmoAct2 dependency. The SO101 station process should
talk to this runner over HTTP instead of importing MolmoAct2 directly.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

DEFAULT_TIMEOUT_S = 75.0


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _decode_image(payload: dict[str, Any]) -> np.ndarray:
    encoding = str(payload.get("encoding") or "jpeg_base64")
    if encoding != "jpeg_base64":
        raise ValueError(f"unsupported image encoding: {encoding}")
    data = str(payload.get("data") or "")
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode JPEG image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class MolmoAct2Runner:
    def __init__(self, molmo_url: str, timeout_s: float):
        try:
            from molmoact2 import adapter
            from molmoact2.client import MolmoActClient, Observation
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "molmoact2 is not installed in this environment. Run this runner in the "
                "MolmoAct2 environment, or point the station at a policy server that implements /act."
            ) from exc

        self.molmo_url = molmo_url
        self.timeout_s = float(timeout_s)
        self.observation_cls = Observation
        self.client = MolmoActClient(molmo_url, conv=adapter.LEROBOT_V21_COMPAT_DEG, timeout_s=self.timeout_s)

    def health(self) -> dict[str, Any]:
        try:
            health = self.client.health()
        except Exception as exc:
            return {"ok": False, "policy": "molmoact2", "molmo_url": self.molmo_url, "error": str(exc)}
        return {"ok": True, "policy": "molmoact2", "molmo_url": self.molmo_url, "backend": health}

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        images_payload = payload.get("images")
        if not isinstance(images_payload, dict) or not images_payload:
            raise ValueError("payload.images must be a non-empty object")
        camera_order = payload.get("camera_order")
        if not isinstance(camera_order, list) or not camera_order:
            camera_order = list(images_payload.keys())

        images = []
        for name in camera_order:
            item = images_payload.get(str(name))
            if not isinstance(item, dict):
                raise ValueError(f"missing image for camera {name!r}")
            images.append(_decode_image(item))

        state = np.asarray(payload.get("state"), dtype=np.float32).reshape(-1)
        instruction = str(payload.get("instruction") or "")
        start = time.monotonic()
        actions = self.client.act(self.observation_cls(images=images, state_rad=state, instruction=instruction))
        elapsed_s = time.monotonic() - start
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape(1, -1)
        return {
            "policy": "molmoact2",
            "action_units": payload.get("action_units") or "degrees",
            "latency_s": round(elapsed_s, 6),
            "actions": actions_arr.tolist(),
        }


def make_handler(runner: MolmoAct2Runner):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                _json_response(self, 200, runner.health())
            else:
                _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                _json_response(self, 400, {"error": "invalid json"})
                return

            try:
                if parsed.path == "/act":
                    _json_response(self, 200, runner.act(payload))
                else:
                    _json_response(self, 404, {"error": "not found"})
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8302)
    parser.add_argument("--molmo-url", default="http://127.0.0.1:8202")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    runner = MolmoAct2Runner(args.molmo_url, timeout_s=args.timeout_s)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner))
    print(
        f"MolmoAct2 policy runner listening on http://{args.host}:{args.port} "
        f"(backend {args.molmo_url})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
