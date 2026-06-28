import importlib.util
from pathlib import Path

import numpy as np


def _load_so101_web_intervene():
    path = Path(__file__).resolve().parents[1] / "scripts" / "so101_web_intervene.py"
    spec = importlib.util.spec_from_file_location("so101_web_intervene", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CountingSam3Tracker:
    def __init__(self):
        module = _load_so101_web_intervene()

        class Tracker(module.LiveCupSuccessTracker):
            def __init__(self):
                self.sam3_calls = 0
                super().__init__()

            def _calculate_sam3_cup_mask(self, rgb, default_area):
                self.sam3_calls += 1
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[100:180, 240:320] = True
                self.cup_mask_score = 1.0
                self._set_sam3_mask_status("accepted", "test")
                return mask, "sam3:test cup"

        self.tracker = Tracker()


def _frame() -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[130:155, 270:295] = [0, 0, 255]
    return image


def test_success_tracker_uses_one_sam3_seed_across_frames() -> None:
    tracker = CountingSam3Tracker().tracker

    for _ in range(4):
        tracker.update(_frame())

    assert tracker.sam3_calls == 1
    assert tracker.status()["container_mask_source"] == "sam3:test cup"


def test_success_tracker_reset_without_recalculate_keeps_sam3_seed() -> None:
    tracker = CountingSam3Tracker().tracker
    tracker.update(_frame())

    tracker.reset(recalculate_container=False)
    tracker.update(_frame())

    assert tracker.sam3_calls == 1


def test_success_tracker_recalculate_container_runs_one_new_sam3_seed() -> None:
    tracker = CountingSam3Tracker().tracker
    tracker.update(_frame())

    tracker.reset(recalculate_container=True, container_reason="new_policy_run")
    for _ in range(3):
        tracker.update(_frame())

    assert tracker.sam3_calls == 2
