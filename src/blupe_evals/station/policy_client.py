"""Generic HTTP client for station policy runners."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
import requests


def encode_rgb_jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("failed to encode camera frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class HttpPolicyClient:
    """Thin process boundary for policy runners.

    The station owns robot IO, recording, and safety. The policy process owns model
    inference and exposes /act plus optional /health over HTTP.
    """

    def __init__(self, base_url: str, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def health(self) -> dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3.0)
            resp.raise_for_status()
            payload = resp.json()
            return payload if isinstance(payload, dict) else {"ok": True, "response": payload}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": self.base_url}

    def act(
        self,
        *,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
        joints: list[str],
        robot_type: str = "so101_follower",
    ) -> np.ndarray:
        image_payload = {
            name: {
                "encoding": "jpeg_base64",
                "data": encode_rgb_jpeg_b64(rgb),
            }
            for name, rgb in images.items()
        }
        payload = {
            "schema_version": 1,
            "robot_type": robot_type,
            "joints": joints,
            "instruction": instruction,
            "state": [float(x) for x in np.asarray(state, dtype=np.float32).reshape(-1)],
            "state_units": "degrees",
            "images": image_payload,
            "camera_order": list(images.keys()),
            "action_units": "degrees",
        }
        resp = requests.post(f"{self.base_url}/act", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("policy /act response must be a JSON object")
        raw_actions = (
            data.get("actions")
            or data.get("action_chunk")
            or data.get("actions_deg")
            or data.get("action")
        )
        if raw_actions is None:
            raise RuntimeError("policy /act response missing actions/action_chunk/action")
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] != len(joints):
            raise RuntimeError(f"policy returned action shape {actions.shape}, expected Nx{len(joints)}")
        return actions
