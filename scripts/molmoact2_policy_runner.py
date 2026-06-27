#!/usr/bin/env python3
"""HTTP policy runner for LeRobot MolmoAct2 checkpoints.

The SO101 web process owns robot IO and safety. This process owns MolmoAct2
dependencies and model inference, then exposes the small `/act` contract used by
`src/blupe_evals/station/policy_client.py`.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import contextlib
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


def _filter_kwargs_for_callable(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _is_local_training_checkpoint(checkpoint_path: str | None) -> bool:
    if not checkpoint_path:
        return False
    checkpoint_dir = Path(checkpoint_path).expanduser()
    return checkpoint_dir.exists() and (checkpoint_dir / "config.yaml").exists()


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
        self.processor = None
        self.action_tokenizer = None
        self.policy = None
        self.policy_type = "molmoact2"
        self.runner_api = "lerobot_policy"
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
            if _is_local_training_checkpoint(self.checkpoint_path):
                self._load_local_training_checkpoint(self.checkpoint_path)
            else:
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

    def _load_local_training_checkpoint(self, checkpoint_path: str) -> None:
        try:
            import lerobot.policies.molmoact2.configuration_molmoact2  # noqa: F401
            from lerobot.configs.types import FeatureType, PolicyFeature
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
            from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
            from lerobot.utils.constants import ACTION
        except Exception as exc:
            raise SystemExit(
                "LeRobot MolmoAct2 is not importable. Run this in the MolmoAct2 experiments environment."
            ) from exc

        if not self.norm_tag:
            raise SystemExit("--norm-tag is required for local MolmoAct2 training checkpoints.")

        cfg = MolmoAct2Config(
            checkpoint_path=str(Path(checkpoint_path).expanduser()),
            device=self.device,
            num_steps=self.num_flow_timesteps,
            inference_action_mode=self.inference_action_mode,
            norm_tag=self.norm_tag,
            enable_inference_cuda_graph=self.enable_cuda_graph,
        )
        cfg.input_features = {
            self.state_key: PolicyFeature(type=FeatureType.STATE, shape=(self.action_dim,)),
            **{
                image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640))
                for image_key in self.image_keys
            },
        }
        cfg.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
        }
        self.policy = MolmoAct2Policy(cfg)
        self.preprocessor, self.postprocessor = make_pre_post_processors(cfg)
        self.policy_type = "molmoact2"
        self.runner_api = "lerobot_local_training_checkpoint"

    def _load_original_hf_checkpoint(self, checkpoint_path: str) -> None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise SystemExit(
                "Transformers with MolmoAct2 trust_remote_code support is required for base HF checkpoints."
            ) from exc

        dtype_by_name = {
            "float32": self.torch.float32,
            "bfloat16": self.torch.bfloat16,
            "float16": self.torch.float16,
        }
        dtype = dtype_by_name[self.model_dtype]
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        try:
            self.policy = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                dtype=dtype,
            )
        except TypeError:
            self.policy = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        if self.inference_action_mode == "discrete":
            self.action_tokenizer = AutoProcessor.from_pretrained(
                "allenai/MolmoAct2-FAST-Tokenizer",
                trust_remote_code=True,
            )
        self.policy_type = "molmoact2"
        self.runner_api = "hf_predict_action"

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
            "runner_api": self.runner_api,
            "image_keys": self.image_keys,
            "state_key": self.state_key,
            "num_actions": self.num_actions,
            "action_dim": self.action_dim,
            "num_steps": self.num_flow_timesteps,
            "enable_cuda_graph": self.enable_cuda_graph,
            "normalize_language": True,
            "enable_depth_reasoning": False,
            "has_processor": self.processor is not None,
            "has_preprocessor": self.preprocessor is not None,
            "has_postprocessor": self.postprocessor is not None,
        }

    def _ordered_hf_inputs(self, payload: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray, str]:
        images_payload = payload.get("images")
        if not isinstance(images_payload, dict) or not images_payload:
            raise ValueError("payload.images must be a non-empty object")

        camera_order = payload.get("camera_order")
        if not isinstance(camera_order, list) or not camera_order:
            camera_order = list(images_payload.keys())
        camera_order = [str(name) for name in camera_order]

        images: list[np.ndarray] = []
        for idx, image_key in enumerate(self.image_keys):
            camera_name = _camera_name_from_image_key(image_key)
            if camera_name not in images_payload and idx < len(camera_order):
                camera_name = camera_order[idx]
            item = images_payload.get(camera_name)
            if not isinstance(item, dict):
                available = sorted(str(name) for name in images_payload)
                raise ValueError(f"missing image for {image_key!r}; expected camera {camera_name!r}, got {available}")
            images.append(_decode_image(item))

        state = np.asarray(payload.get("state"), dtype=np.float32).reshape(-1)
        if state.shape != (self.action_dim,):
            raise ValueError(f"payload.state must have shape ({self.action_dim},), got {state.shape}")
        task = str(payload.get("instruction") or "")
        return images, state, task

    def _hf_autocast(self):
        if self.model_dtype == "float32" or not str(self.device).startswith("cuda"):
            return contextlib.nullcontext()
        dtype = self.torch.bfloat16 if self.model_dtype == "bfloat16" else self.torch.float16
        return self.torch.autocast("cuda", dtype=dtype)

    def _predict_hf_actions(self, payload: dict[str, Any]) -> Any:
        if self.processor is None:
            raise RuntimeError("HF processor is not loaded")
        images, state, task = self._ordered_hf_inputs(payload)
        predict_kwargs: dict[str, Any] = {
            "processor": self.processor,
            "images": images,
            "task": task,
            "state": state,
            "norm_tag": self.norm_tag,
            "inference_action_mode": self.inference_action_mode,
            "enable_depth_reasoning": False,
            "num_steps": self.num_flow_timesteps,
            "normalize_language": True,
            "enable_cuda_graph": self.enable_cuda_graph,
        }
        if self.action_tokenizer is not None:
            predict_kwargs["action_tokenizer"] = self.action_tokenizer
        with self.torch.inference_mode(), self._hf_autocast():
            out = self.policy.predict_action(**predict_kwargs)
        return out.actions

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

        already_unnormalized = False
        with self.torch.inference_mode():
            if hasattr(self.policy, "predict_action_chunk"):
                actions = self.policy.predict_action_chunk(batch)
            else:
                selected = [self.policy.select_action(batch) for _ in range(max(1, self.num_actions))]
                actions = self.torch.stack(selected, dim=1)

        if actions.ndim != 3:
            actions = actions.unsqueeze(0)
        actions = actions[:, : self.num_actions, :]

        if self.runner_api == "lerobot_local_training_checkpoint":
            actions, already_unnormalized = self._unnormalize_local_training_actions(actions)

        if self.postprocessor is not None and not already_unnormalized:
            processed = []
            for idx in range(actions.shape[1]):
                processed.append(self.postprocessor(actions[:, idx, :]))
            actions = self.torch.stack(processed, dim=1)

        return actions

    def _unnormalize_local_training_actions(self, actions: Any) -> tuple[Any, bool]:
        handles = getattr(self.policy, "_handles", None)
        robot_processor = getattr(handles, "robot_processor", None)
        if robot_processor is None or not hasattr(robot_processor, "unnormalize_action"):
            return actions, False

        norm_tag = self.norm_tag or str(getattr(handles, "norm_tag", "") or "").strip()
        if not norm_tag:
            return actions, False

        try:
            actions = robot_processor.unnormalize_action(actions, repo_id=norm_tag)
        except Exception as exc:
            raise RuntimeError(f"failed to unnormalize MolmoAct2 actions for norm_tag={norm_tag!r}") from exc
        return actions, True

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        if self.runner_api == "hf_predict_action":
            actions = _as_numpy_actions(self._predict_hf_actions(payload))
        else:
            batch = self._prepare_batch(payload)
            actions = _as_numpy_actions(self._predict_action_chunk(batch))
        actions = actions[: self.num_actions]
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
    parser.add_argument("--model-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--inference-action-mode", default="continuous", choices=("continuous", "discrete"))
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-flow-timesteps", type=int, default=10)
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
