from scripts.molmoact2_policy_runner import _filter_kwargs_for_callable


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

