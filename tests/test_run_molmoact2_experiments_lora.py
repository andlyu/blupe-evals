import os
import sys
import types
from pathlib import Path

from scripts.run_molmoact2_experiments_lora import _data_mix_rows, _log_data_mix_bar_charts, _set_default_env


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


def _data_mix_test_metrics() -> dict[str, float]:
    return {
        "data_mix/custom_sample_pct": 50.0,
        "data_mix/general_sample_pct": 50.0,
        "data_mix/custom_episode_pct": 23.0769,
        "data_mix/general_episode_pct": 76.9231,
        "data_mix/custom_frame_pct": 8.8199,
        "data_mix/general_frame_pct": 91.1801,
        "data_mix/custom_sample_weight": 0.5,
        "data_mix/general_sample_weight": 0.5,
        "data_mix/custom_episodes": 6.0,
        "data_mix/general_episodes": 20.0,
        "data_mix/custom_frames": 1130.0,
        "data_mix/general_frames": 11682.0,
    }


def test_data_mix_rows_are_wandb_table_ready():
    assert _data_mix_rows(_data_mix_test_metrics()) == [
        {
            "source": "custom",
            "sample_pct": 50.0,
            "episode_pct": 23.0769,
            "frame_pct": 8.8199,
            "sample_weight": 0.5,
            "episodes": 6.0,
            "frames": 1130.0,
        },
        {
            "source": "general",
            "sample_pct": 50.0,
            "episode_pct": 76.9231,
            "frame_pct": 91.1801,
            "sample_weight": 0.5,
            "episodes": 20.0,
            "frames": 11682.0,
        },
    ]


def test_log_data_mix_bar_charts_sends_wandb_payload(monkeypatch):
    logged_payloads = []

    class FakeTable:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    class FakePlot:
        @staticmethod
        def bar(table, x, y, title):
            return {"table": table, "x": x, "y": y, "title": title}

    fake_wandb = types.SimpleNamespace(
        run=object(),
        Table=FakeTable,
        plot=FakePlot(),
        log=logged_payloads.append,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    assert _log_data_mix_bar_charts(_data_mix_test_metrics()) is True

    payload = logged_payloads[0]
    assert payload["data_mix/table"].columns == [
        "source",
        "sample_pct",
        "episode_pct",
        "frame_pct",
        "sample_weight",
        "episodes",
        "frames",
    ]
    assert payload["data_mix/table"].data[0] == ["custom", 50.0, 23.0769, 8.8199, 0.5, 6.0, 1130.0]
    assert payload["data_mix/sample_pct_bar"]["y"] == "sample_pct"
    assert payload["data_mix/episode_pct_bar"]["y"] == "episode_pct"
    assert payload["data_mix/frame_pct_bar"]["y"] == "frame_pct"


def test_log_data_mix_bar_charts_waits_for_wandb_run(monkeypatch):
    fake_wandb = types.SimpleNamespace(run=None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    assert _log_data_mix_bar_charts(_data_mix_test_metrics()) is False
