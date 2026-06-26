import os
import sys
import types
from pathlib import Path

import scripts.run_molmoact2_experiments_lora as run_lora
from scripts.run_molmoact2_experiments_lora import (
    _data_mix_rows,
    _log_data_mix_bar_charts,
    _patch_runtime_split_metrics,
    _runtime_split_metrics,
    _set_default_env,
)


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

    def log(payload, **kwargs):
        logged_payloads.append((payload, kwargs))

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
        log=log,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    assert _log_data_mix_bar_charts(_data_mix_test_metrics()) is True

    payload, kwargs = logged_payloads[0]
    assert kwargs == {"step": 0, "commit": False}
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


def test_runtime_split_metrics_tracks_training_validation_and_overhead():
    metrics = _runtime_split_metrics(
        elapsed_seconds=100.0,
        train_step_seconds=70.0,
        validation_seconds=20.0,
    )

    assert metrics["runtime/training_seconds"] == 80.0
    assert metrics["runtime/validation_seconds"] == 20.0
    assert metrics["runtime/train_step_seconds"] == 70.0
    assert metrics["runtime/overhead_seconds"] == 10.0
    assert metrics["runtime/training_pct"] == 80.0
    assert metrics["runtime/validation_pct"] == 20.0
    assert metrics["runtime/train_step_pct"] == 70.0
    assert metrics["runtime/overhead_pct"] == 10.0


def test_patch_runtime_split_metrics_adds_metrics_to_train_and_eval(monkeypatch):
    class FakeTrainer:
        def fit(self):
            return "fit-result"

        def train_step(self):
            return {"train/loss": 1.0}

        def loss_eval(self):
            return {"val/loss": 2.0}

    olmo_mod = types.ModuleType("olmo")
    train_mod = types.ModuleType("olmo.train")
    trainer_mod = types.ModuleType("olmo.train.trainer")
    trainer_mod.Trainer = FakeTrainer
    train_mod.trainer = trainer_mod
    olmo_mod.train = train_mod
    monkeypatch.setitem(sys.modules, "olmo", olmo_mod)
    monkeypatch.setitem(sys.modules, "olmo.train", train_mod)
    monkeypatch.setitem(sys.modules, "olmo.train.trainer", trainer_mod)

    perf_counter_values = iter([100.0, 101.0, 111.0, 112.0, 120.0, 125.0, 130.0])
    monkeypatch.setattr(run_lora.time, "perf_counter", lambda: next(perf_counter_values))

    _patch_runtime_split_metrics()

    trainer = FakeTrainer()
    assert trainer.fit() == "fit-result"

    train_metrics = trainer.train_step()
    assert train_metrics["train/loss"] == 1.0
    assert train_metrics["runtime/train_step_seconds"] == 10.0
    assert train_metrics["runtime/validation_seconds"] == 0.0
    assert train_metrics["runtime/elapsed_seconds"] == 12.0

    eval_metrics = trainer.loss_eval()
    assert eval_metrics["val/loss"] == 2.0
    assert eval_metrics["runtime/train_step_seconds"] == 10.0
    assert eval_metrics["runtime/validation_seconds"] == 5.0
    assert eval_metrics["runtime/training_seconds"] == 25.0
    assert eval_metrics["runtime/validation_pct"] == 100.0 * 5.0 / 30.0
