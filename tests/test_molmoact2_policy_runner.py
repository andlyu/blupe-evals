import contextlib
from types import SimpleNamespace

import numpy as np

import scripts.molmoact2_policy_runner as runner_module
from scripts.molmoact2_policy_runner import LeRobotMolmoAct2Runner, _filter_kwargs_for_callable


def test_original_checkpoint_kwargs_match_legacy_molmoact2_config_signature():
    class LegacyMolmoAct2Config:
        def __init__(
            self,
            checkpoint_path: str,
            device: str | None = None,
            inference_action_mode: str = "continuous",
            use_amp: bool = False,
            enable_inference_cuda_graph: bool = True,
            num_steps: int | None = None,
            norm_tag: str = "",
        ):
            pass

    kwargs = {
        "checkpoint_path": "allenai/MolmoAct2-SO100_101",
        "device": "cuda",
        "model_dtype": "bfloat16",
        "inference_action_mode": "continuous",
        "use_amp": True,
        "enable_inference_cuda_graph": False,
        "num_flow_timesteps": 8,
        "num_steps": 8,
        "image_keys": ["observation.images.front", "observation.images.wrist"],
        "chunk_size": 30,
        "n_action_steps": 30,
        "norm_tag": "",
    }

    filtered = _filter_kwargs_for_callable(LegacyMolmoAct2Config, kwargs)

    assert filtered == {
        "checkpoint_path": "allenai/MolmoAct2-SO100_101",
        "device": "cuda",
        "inference_action_mode": "continuous",
        "use_amp": True,
        "enable_inference_cuda_graph": False,
        "num_steps": 8,
        "norm_tag": "",
    }
    LegacyMolmoAct2Config(**filtered)


def test_hf_checkpoint_path_calls_model_card_predict_action(monkeypatch):
    class FakeTorch:
        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    class FakePolicy:
        def __init__(self):
            self.kwargs = None

        def predict_action(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(actions=np.zeros((30, 6), dtype=np.float32))

    monkeypatch.setattr(
        runner_module,
        "_decode_image",
        lambda payload: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    policy = FakePolicy()
    runner = LeRobotMolmoAct2Runner.__new__(LeRobotMolmoAct2Runner)
    runner.processor = object()
    runner.action_tokenizer = None
    runner.policy = policy
    runner.torch = FakeTorch()
    runner.model_dtype = "float32"
    runner.device = "cuda"
    runner.norm_tag = "so100_so101_molmoact2"
    runner.inference_action_mode = "continuous"
    runner.num_flow_timesteps = 10
    runner.enable_cuda_graph = True
    runner.image_keys = ["observation.images.front", "observation.images.wrist"]
    runner.action_dim = 6

    runner._predict_hf_actions(
        {
            "images": {"front": {}, "wrist": {}},
            "state": [0, -90, 70, 0, -45, 0],
            "instruction": "Move to blue ball.",
        }
    )

    assert policy.kwargs is not None
    assert policy.kwargs["processor"] is runner.processor
    assert len(policy.kwargs["images"]) == 2
    np.testing.assert_array_equal(
        policy.kwargs["state"],
        np.asarray([0, -90, 70, 0, -45, 0], dtype=np.float32),
    )
    assert policy.kwargs["task"] == "Move to blue ball."
    assert policy.kwargs["norm_tag"] == "so100_so101_molmoact2"
    assert policy.kwargs["inference_action_mode"] == "continuous"
    assert policy.kwargs["enable_depth_reasoning"] is False
    assert policy.kwargs["num_steps"] == 10
    assert policy.kwargs["normalize_language"] is True
    assert policy.kwargs["enable_cuda_graph"] is True
    assert "action_tokenizer" not in policy.kwargs
