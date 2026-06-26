#!/usr/bin/env python3
"""HTTP policy runner for LeRobot MolmoAct2 checkpoints.

The SO101 web process owns robot IO and safety. This process owns MolmoAct2
dependencies and model inference, then exposes the small `/act` contract used by
`src/blupe_evals/station/policy_client.py`.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8202
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_IMAGE_KEYS = ("observation.images.front", "observation.images.wrist")
DEFAULT_ACTION_DIM = 6


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_json_list(value: str | None, *, default: tuple[str, ...] = ()) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    raw = str(value).strip()
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON list")
        return [str(item) for item in parsed]
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def _camera_name_from_image_key(image_key: str) -> str:
    return image_key.rsplit(".", 1)[-1]


def _as_numpy_actions(actions: Any) -> np.ndarray:
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise RuntimeError(f"expected batch size 1 action chunk, got {arr.shape}")
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise RuntimeError(f"policy returned unsupported action shape {arr.shape}")
    return arr


class LeRobotMolmoAct2Runner:
    def __init__(
        self,
        *,
        policy_path: str | None,
        checkpoint_path: str | None,
        norm_tag: str | None,
        image_keys: list[str],
        state_key: str,
        device: str,
        model_dtype: str,
        inference_action_mode: str,
        use_amp: bool,
        enable_cuda_graph: bool,
        num_flow_timesteps: int,
        num_actions: int,
        action_dim: int,
    ):
        if bool(policy_path) == bool(checkpoint_path):
            raise ValueError("provide exactly one of --policy-path or --checkpoint-path")

        self.policy_path = str(Path(policy_path).expanduser()) if policy_path else None
        self.checkpoint_path = checkpoint_path
        self.norm_tag = norm_tag
        self.image_keys = list(image_keys)
        self.state_key = state_key
        self.device = device
        self.model_dtype = model_dtype
        self.inference_action_mode = inference_action_mode
        self.use_amp = bool(use_amp)
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self.num_flow_timesteps = int(num_flow_timesteps)
        self.num_actions = int(num_actions)
        self.action_dim = int(action_dim)
        self.preprocessor = None
        self.postprocessor = None
        self.policy = None
        self.policy_type = "molmoact2"
        self.loaded_from = self.policy_path or self.checkpoint_path or ""

        self._load_policy()

    def _load_policy(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise SystemExit("torch is required in the MolmoAct2 policy environment") from exc

        self.torch = torch

        if self.policy_path:
            self._load_lerobot_policy(self.policy_path)
        else:
            assert self.checkpoint_path is not None
            self._load_original_hf_checkpoint(self.checkpoint_path)

        self.policy.to(self.device)
        self.policy.eval()

    def _load_lerobot_policy(self, policy_path: str) -> None:
        try:
            try:
                import lerobot.policies.molmoact2.configuration_molmoact2  # noqa: F401
            except Exception:
                pass
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        except Exception as exc:
            raise SystemExit(
                "LeRobot is not importable. Run this in a MolmoAct2-enabled LeRobot environment."
            ) from exc

        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.device = self.device
        if hasattr(cfg, "model_dtype"):
            cfg.model_dtype = self.model_dtype
        if hasattr(cfg, "inference_action_mode"):
            cfg.inference_action_mode = self.inference_action_mode
        if hasattr(cfg, "use_amp"):
            cfg.use_amp = self.use_amp
        if hasattr(cfg, "enable_inference_cuda_graph"):
            cfg.enable_inference_cuda_graph = self.enable_cuda_graph
        if hasattr(cfg, "num_flow_timesteps"):
            cfg.num_flow_timesteps = self.num_flow_timesteps

        policy_cls = get_policy_class(cfg.type)
        self.policy_type = str(cfg.type)
        self.policy = policy_cls.from_pretrained(policy_path, config=cfg)

        device_override = {"device": self.device}
        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=policy_path,
                preprocessor_overrides={"device_processor": device_override},
                postprocessor_overrides={"device_processor": device_override},
            )
        except Exception as exc:
            print(f"[molmoact2] warning: could not load saved processors: {exc}", file=sys.stderr, flush=True)

    def _load_original_hf_checkpoint(self, checkpoint_path: str) -> None:
        try:
            from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
            from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        except Exception as exc:
            raise SystemExit(
                "This LeRobot environment does not include MolmoAct2. Install LeRobot from source with "
                "the molmoact2 extra before starting this runner."
            ) from exc

        cfg_kwargs: dict[str, Any] = {
            "checkpoint_path": checkpoint_path,
            "device": self.device,
            "model_dtype": self.model_dtype,
            "inference_action_mode": self.inference_action_mode,
            "use_amp": self.use_amp,
            "enable_inference_cuda_graph": self.enable_cuda_graph,
            "num_flow_timesteps": self.num_flow_timesteps,
            "image_keys": self.image_keys,
            "chunk_size": self.num_actions,
            "n_action_steps": self.num_actions,
        }
        if self.norm_tag:
            cfg_kwargs["norm_tag"] = self.norm_tag
        cfg = MolmoAct2Config(**cfg_kwargs)
        self.policy = MolmoAct2Policy(cfg)
        self.policy_type = "molmoact2"

    def health(self) -> dict[str, Any]:
        cuda_available = None
        if hasattr(self, "torch"):
            cuda_available = bool(self.torch.cuda.is_available())
        return {
            "ok": self.policy is not None,
            "policy": self.policy_type,
            "loaded_from": self.loaded_from,
            "norm_tag": self.norm_tag,
            "device": self.device,
            "cuda_available": cuda_available,
            "model_dtype": self.model_dtype,
            "inference_action_mode": self.inference_action_mode,
            "image_keys": self.image_keys,
            "state_key": self.state_key,
            "num_actions": self.num_actions,
            "action_dim": self.action_dim,
            "has_preprocessor": self.preprocessor is not None,
            "has_postprocessor": self.postprocessor is not None,
        }

    def _target_image_shape(self, image_key: str) -> tuple[int, int, int] | None:
        config = getattr(self.policy, "config", None)
        features = getattr(config, "image_features", None) or getattr(config, "input_features", None) or {}
        feature = features.get(image_key) if isinstance(features, dict) else None
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if not shape:
            return None
        dims = tuple(int(x) for x in shape)
        return dims if len(dims) == 3 else None

    def _image_tensor(self, image_key: str, rgb: np.ndarray) -> Any:
        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        tensor = tensor.permute(2, 0, 1).to(dtype=self.torch.float32).div_(255.0)
        target_shape = self._target_image_shape(image_key)
        if target_shape is not None:
            _, target_h, target_w = target_shape
            if tensor.shape[-2:] != (target_h, target_w):
                tensor = self.torch.nn.functional.interpolate(
                    tensor.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        return tensor.unsqueeze(0)

    def _prepare_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        images_payload = payload.get("images")
        if not isinstance(images_payload, dict) or not images_payload:
            raise ValueError("payload.images must be a non-empty object")

        camera_order = payload.get("camera_order")
        if not isinstance(camera_order, list) or not camera_order:
            camera_order = list(images_payload.keys())
        camera_order = [str(name) for name in camera_order]

        batch: dict[str, Any] = {
            self.state_key: self.torch.as_tensor(
                np.asarray(payload.get("state"), dtype=np.float32).reshape(1, -1),
                device=self.device,
            ),
            "task": [str(payload.get("instruction") or "")],
        }

        for idx, image_key in enumerate(self.image_keys):
            camera_name = _camera_name_from_image_key(image_key)
            if camera_name not in images_payload and idx < len(camera_order):
                camera_name = camera_order[idx]
            item = images_payload.get(camera_name)
            if not isinstance(item, dict):
                available = sorted(str(name) for name in images_payload)
                raise ValueError(f"missing image for {image_key!r}; expected camera {camera_name!r}, got {available}")
            batch[image_key] = self._image_tensor(image_key, _decode_image(item))

        return batch

    def _predict_action_chunk(self, batch: dict[str, Any]) -> Any:
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)

        if hasattr(self.policy, "reset"):
            self.policy.reset()

        with self.torch.inference_mode():
            if hasattr(self.policy, "predict_action_chunk"):
                actions = self.policy.predict_action_chunk(batch)
            else:
                selected = [self.policy.select_action(batch) for _ in range(max(1, self.num_actions))]
                actions = self.torch.stack(selected, dim=1)

        if actions.ndim != 3:
            actions = actions.unsqueeze(0)
        actions = actions[:, : self.num_actions, :]

        if self.postprocessor is not None:
            processed = []
            for idx in range(actions.shape[1]):
                processed.append(self.postprocessor(actions[:, idx, :]))
            actions = self.torch.stack(processed, dim=1)

        return actions

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        batch = self._prepare_batch(payload)
        actions = _as_numpy_actions(self._predict_action_chunk(batch))
        if actions.shape[1] != self.action_dim:
            raise RuntimeError(f"policy returned action dim {actions.shape[1]}, expected {self.action_dim}")
        elapsed_s = time.monotonic() - start
        return {
            "policy": self.policy_type,
            "loaded_from": self.loaded_from,
            "action_units": payload.get("action_units") or "degrees",
            "latency_s": round(elapsed_s, 6),
            "actions": actions.tolist(),
        }


def make_handler(runner: LeRobotMolmoAct2Runner):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--policy-path", default="", help="LeRobot-saved policy directory or Hub repo.")
    parser.add_argument("--checkpoint-path", default="", help="Original MolmoAct2 HF checkpoint path/repo.")
    parser.add_argument("--norm-tag", default="", help="Required for original MolmoAct2 HF checkpoints.")
    parser.add_argument("--image-keys", default=json.dumps(list(DEFAULT_IMAGE_KEYS)))
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--inference-action-mode", default="continuous", choices=("continuous", "discrete"))
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-flow-timesteps", type=int, default=8)
    parser.add_argument("--num-actions", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    args = parser.parse_args()

    runner = LeRobotMolmoAct2Runner(
        policy_path=args.policy_path or None,
        checkpoint_path=args.checkpoint_path or None,
        norm_tag=args.norm_tag or None,
        image_keys=_parse_json_list(args.image_keys, default=DEFAULT_IMAGE_KEYS),
        state_key=args.state_key,
        device=args.device,
        model_dtype=args.model_dtype,
        inference_action_mode=args.inference_action_mode,
        use_amp=args.use_amp,
        enable_cuda_graph=args.enable_cuda_graph,
        num_flow_timesteps=args.num_flow_timesteps,
        num_actions=args.num_actions,
        action_dim=args.action_dim,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner))
    print(
        f"MolmoAct2 LeRobot policy runner listening on http://{args.host}:{args.port} "
        f"loaded_from={runner.loaded_from}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
