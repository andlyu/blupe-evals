import os
from pathlib import Path

from scripts.run_molmoact2_experiments_lora import _set_default_env


def test_set_default_env_adds_single_process_torch_distributed_defaults(monkeypatch):
    for key in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "PYTHONPATH",
    ]:
        monkeypatch.delenv(key, raising=False)

    _set_default_env(Path("/workspace/molmoact2/experiments"), "/workspace/lerobot_data")

    assert {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
    }.items() <= dict(os.environ).items()
