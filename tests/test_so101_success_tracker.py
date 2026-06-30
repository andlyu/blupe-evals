import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _load_so101_web_intervene():
    path = Path(__file__).resolve().parents[1] / "scripts" / "so101_web_intervene.py"
    spec = importlib.util.spec_from_file_location("so101_web_intervene", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CountingSam3Tracker:
    def __init__(self):
        module = _load_so101_web_intervene()

        class Tracker(module.LiveCupSuccessTracker):
            def __init__(self):
                self.sam3_calls = 0
                self.ball_sam3_calls = 0
                super().__init__()

            def _calculate_sam3_cup_mask(self, rgb, default_area):
                self.sam3_calls += 1
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[100:180, 240:320] = True
                self.cup_mask_score = 1.0
                self._set_sam3_mask_status("accepted", "test")
                return mask, "sam3:test cup"

            def _calculate_sam3_ball_mask(self, rgb):
                self.ball_sam3_calls += 1
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[130:155, 270:295] = True
                self.ball_mask_score = 0.9
                self._set_ball_sam3_status("accepted", "test")
                return mask, "sam3:test ball"

        self.tracker = Tracker()


def _frame() -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[130:155, 270:295] = [0, 0, 255]
    return image


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _mask_payload(module, mask: np.ndarray, score: float = 0.9) -> dict:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    box = module.LiveCupSuccessTracker._mask_box(mask > 0)
    return {
        "top_mask": {
            "score": score,
            "area_px": int((mask > 0).sum()),
            "box_xyxy": box,
            "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    }


def _install_fake_sam3(monkeypatch, module, cup_mask_ref: dict[str, np.ndarray]) -> None:
    ball_mask = np.zeros((360, 640), dtype=np.uint8)
    ball_mask[130:155, 270:295] = 255

    def fake_post(url, json, timeout):
        del url, timeout
        prompt = (json.get("prompts") or [""])[0]
        mask = ball_mask if "ball" in prompt else cup_mask_ref["mask"]

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return _mask_payload(module, mask, score=0.9)

        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)


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


def test_success_tracker_auto_iou_failure_keeps_previous_cup_mask(monkeypatch) -> None:
    module = _load_so101_web_intervene()
    first_cup = np.zeros((360, 640), dtype=np.uint8)
    first_cup[100:180, 240:320] = 255
    moved_cup = np.zeros((360, 640), dtype=np.uint8)
    moved_cup[210:290, 420:500] = 255
    cup_mask_ref = {"mask": first_cup}
    _install_fake_sam3(monkeypatch, module, cup_mask_ref)

    tracker = module.LiveCupSuccessTracker()
    tracker.ball_sam2_url = ""
    tracker.update(_frame())
    assert tracker.status()["container_mask_sam3_status"] == "accepted"
    assert tracker.status()["container_mask_box_xyxy"] == [240, 100, 319, 179]

    cup_mask_ref["mask"] = moved_cup
    tracker.reset(recalculate_container=True, container_reason="policy_start")
    tracker.update(_frame())

    status = tracker.status()
    assert status["container_mask_sam3_status"] == "episode_iou_fallback_previous"
    assert status["container_mask_source"].startswith("sam3:previous_fallback")
    assert status["container_mask_box_xyxy"] == [240, 100, 319, 179]
    assert status["container_mask_area"] == int((first_cup > 0).sum())


def test_success_tracker_manual_rerun_ignores_previous_cup_iou(monkeypatch) -> None:
    module = _load_so101_web_intervene()
    first_cup = np.zeros((360, 640), dtype=np.uint8)
    first_cup[100:180, 240:320] = 255
    moved_cup = np.zeros((360, 640), dtype=np.uint8)
    moved_cup[210:290, 420:500] = 255
    cup_mask_ref = {"mask": first_cup}
    _install_fake_sam3(monkeypatch, module, cup_mask_ref)

    tracker = module.LiveCupSuccessTracker()
    tracker.ball_sam2_url = ""
    tracker.update(_frame())

    cup_mask_ref["mask"] = moved_cup
    tracker.reset(recalculate_container=True, container_reason="manual_sam3_rerun")
    tracker.update(_frame())

    status = tracker.status()
    assert status["container_mask_sam3_status"] == "accepted"
    assert status["container_mask_source"].startswith("sam3:")
    assert "previous_fallback" not in status["container_mask_source"]
    assert status["container_mask_box_xyxy"] == [420, 210, 499, 289]
    assert status["container_mask_area"] == int((moved_cup > 0).sum())


def test_success_tracker_defaults_to_blue_rubber_ball_prompt_and_lower_threshold() -> None:
    module = _load_so101_web_intervene()

    assert module.SUCCESS_CONTAINER_SAM3_MIN_SCORE == 0.25
    assert module.SUCCESS_BALL_SAM3_PROMPT == "blue rubber ball"
    assert module.SUCCESS_BALL_SAM3_MIN_SCORE == 0.25
    assert module.SUCCESS_BALL_SAM3_EVERY_N_FRAMES == 0
    assert module.SUCCESS_BALL_SAM2_EVERY_N_FRAMES == 2
    assert module.SUCCESS_BALL_SAM2_BOX_PAD_PX == 2
    assert module.SUCCESS_BALL_SAM2_MAX_AREA_MULT == 1.0
    assert module.SUCCESS_CUP_MIN_MASK_AREA == 500


def test_success_tracker_uses_sam3_ball_masks_when_sam2_disabled() -> None:
    tracker = CountingSam3Tracker().tracker
    tracker.ball_sam2_url = ""

    for _ in range(3):
        tracker.update(_frame())

    assert tracker.ball_sam3_calls == 3
    assert tracker.status()["ball_mask_source"] == "sam3:test ball"
    assert tracker.status()["ball_mask_sam2_status"] == "disabled"


def test_success_tracker_uses_sam2_after_sam3_ball_seed(monkeypatch) -> None:
    module = _load_so101_web_intervene()

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            self.sam3_ball_calls = 0
            self.sam2_calls = 0
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"

        def _calculate_sam3_cup_mask(self, rgb, default_area):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[100:180, 240:320] = True
            self.cup_mask_score = 1.0
            self._set_sam3_mask_status("accepted", "test")
            return mask, "sam3:test cup"

        def _calculate_sam3_ball_mask(self, rgb):
            self.sam3_ball_calls += 1
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[130:155, 270:295] = True
            self.ball_mask_score = 0.9
            self._set_ball_sam3_status("accepted", "test")
            return mask, "sam3:blue rubber ball score=0.90"

    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[132:157, 272:297] = 255
        ok, encoded = cv2.imencode(".png", mask)
        assert ok

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "mode": "sam2_video",
                    "top_mask": {
                        "tracked": True,
                        "source": "sam2_video_track",
                        "frame_idx": 1,
                        "score": 0.77,
                        "area": int((mask > 0).sum()),
                        "box_xyxy": [272, 132, 296, 156],
                        "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                }

        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)
    tracker = Tracker()

    tracker.update(_frame())
    tracker.update(_frame())
    tracker.update(_frame())

    assert _wait_until(lambda: len(calls) == 1)
    assert _wait_until(lambda: tracker.status()["ball_mask_source"].startswith("sam2:video"))
    status = tracker.status()
    assert tracker.sam3_ball_calls == 1
    assert len(calls) == 1
    assert calls[0]["url"] == "http://sam2.test/api/track_image"
    assert "mask_png_b64" in calls[0]["json"]
    assert calls[0]["json"]["session_id"] == f"ball-{tracker.ball_mask_generation}"
    assert calls[0]["json"]["reset_session"] is True
    assert calls[0]["json"]["box_pad_px"] == 2
    assert calls[0]["json"]["max_area"] == 625
    assert status["ball_mask_source"].startswith("sam2:video")
    assert status["ball_mask_sam2_status"] == "accepted"
    assert status["ball_mask_sam2_raw_score"] == 0.77
    assert status["ball_mask_sam3_area_mean"] == 625.0
    assert status["ball_mask_sam2_max_area"] == 625


def test_success_tracker_rejects_sam2_area_larger_than_sam3_reference(monkeypatch) -> None:
    module = _load_so101_web_intervene()

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"

        def _calculate_sam3_cup_mask(self, rgb, default_area):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[100:180, 240:320] = True
            self.cup_mask_score = 1.0
            self._set_sam3_mask_status("accepted", "test")
            return mask, "sam3:test cup"

        def _calculate_sam3_ball_mask(self, rgb):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[130:155, 270:295] = True
            self.ball_mask_score = 0.9
            self._set_ball_sam3_status("accepted", "test")
            return mask, "sam3:blue rubber ball score=0.90"

    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[100:200, 240:340] = 255
        ok, encoded = cv2.imencode(".png", mask)
        assert ok

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "mode": "sam2_image",
                    "top_mask": {
                        "tracked": True,
                        "source": "sam2_image_track",
                        "score": 0.88,
                        "area": int((mask > 0).sum()),
                        "box_xyxy": [240, 100, 339, 199],
                        "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                }

        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)
    tracker = Tracker()

    tracker.update(_frame())
    tracker.update(_frame())

    assert _wait_until(lambda: len(calls) == 1)
    assert _wait_until(lambda: not tracker.status()["ball_mask_sam2_refresh_running"])
    status = tracker.status()
    assert calls[0]["json"]["max_area"] == 625
    assert status["ball_mask_source"].startswith("sam3:")
    assert status["ball_area"] == 625
    assert status["ball_mask_sam2_status"] == "area_gt_sam3_reference"
    assert status["ball_mask_sam2_raw_area"] == 10000
    assert status["ball_mask_sam2_max_area"] == 625


def test_success_tracker_sam2_refresh_does_not_block_live_overlay() -> None:
    module = _load_so101_web_intervene()

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"

        def _calculate_sam3_cup_mask(self, rgb, default_area):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[100:180, 240:320] = True
            self.cup_mask_score = 1.0
            self._set_sam3_mask_status("accepted", "test")
            return mask, "sam3:test cup"

        def _calculate_sam3_ball_mask(self, rgb):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[130:155, 270:295] = True
            self.ball_mask_score = 0.9
            self._set_ball_sam3_status("accepted", "test")
            return mask, "sam3:blue rubber ball score=0.90"

        def _calculate_sam2_ball_mask(self, rgb, seed_mask=None, seed_source=None):
            time.sleep(0.2)
            self._set_ball_sam2_status("accepted", "test")
            return seed_mask.copy(), "sam2:video test"

    tracker = Tracker()
    tracker.update(_frame())

    started = time.monotonic()
    _, overlay = tracker.update(_frame())
    elapsed = time.monotonic() - started

    assert elapsed < 0.08
    assert overlay.shape == _frame().shape
    assert tracker.status()["ball_mask_sam2_refresh_running"] is True
    assert _wait_until(lambda: tracker.status()["ball_mask_source"].startswith("sam2:video"), timeout_s=1.0)


def test_success_tracker_sam2_seed_mask_only_sent_on_session_reset(monkeypatch) -> None:
    module = _load_so101_web_intervene()

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"

        def _calculate_sam3_cup_mask(self, rgb, default_area):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[100:180, 240:320] = True
            self.cup_mask_score = 1.0
            self._set_sam3_mask_status("accepted", "test")
            return mask, "sam3:test cup"

        def _calculate_sam3_ball_mask(self, rgb):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[130:155, 270:295] = True
            self.ball_mask_score = 0.9
            self._set_ball_sam3_status("accepted", "test")
            return mask, "sam3:blue rubber ball score=0.90"

    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[132:157, 272:297] = 255
        ok, encoded = cv2.imencode(".png", mask)
        assert ok

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "mode": "sam2_video",
                    "top_mask": {
                        "tracked": True,
                        "source": "sam2_video_track",
                        "frame_idx": len(calls),
                        "score": 0.77,
                        "area": int((mask > 0).sum()),
                        "box_xyxy": [272, 132, 296, 156],
                        "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                }

        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)
    tracker = Tracker()
    tracker.update(_frame())
    tracker.update(_frame())
    assert _wait_until(lambda: len(calls) == 1)
    assert _wait_until(lambda: tracker.status()["ball_mask_source"].startswith("sam2:video"))

    tracker.update(_frame())
    tracker.update(_frame())
    assert _wait_until(lambda: len(calls) == 2)

    assert calls[0]["json"]["reset_session"] is True
    assert "mask_png_b64" in calls[0]["json"]
    assert calls[1]["json"]["reset_session"] is False
    assert "mask_png_b64" not in calls[1]["json"]


def test_success_tracker_periodically_refreshes_ball_with_sam3_when_sam2_enabled() -> None:
    module = _load_so101_web_intervene()

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            self.sam3_ball_calls = 0
            self.sam2_calls = 0
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"
            self.ball_mask_sam3_every_n_frames = 2

        def _calculate_sam3_cup_mask(self, rgb, default_area):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[100:180, 240:320] = True
            self.cup_mask_score = 1.0
            self._set_sam3_mask_status("accepted", "test")
            return mask, "sam3:test cup"

        def _calculate_sam3_ball_mask(self, rgb, request_frame_idx=None):
            self.sam3_ball_calls += 1
            self.ball_mask_sam3_last_request_frame = self.frame_idx if request_frame_idx is None else int(request_frame_idx)
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            offset = 10 * (self.sam3_ball_calls - 1)
            mask[130:155, 270 + offset : 295 + offset] = True
            self.ball_mask_score = 0.9
            self._set_ball_sam3_status("accepted", "test")
            return mask, f"sam3:blue rubber ball call={self.sam3_ball_calls}"

        def _calculate_sam2_ball_mask(self, rgb, seed_mask=None, seed_source=None):
            self.sam2_calls += 1
            self._set_ball_sam2_status("accepted", "test")
            return seed_mask.copy(), "sam2:video test"

    tracker = Tracker()

    for _ in range(5):
        tracker.update(_frame())
        _wait_until(
            lambda: (
                not tracker.status()["ball_mask_sam2_refresh_running"]
                and not tracker.status()["ball_mask_sam3_refresh_running"]
            ),
            timeout_s=0.5,
        )

    status = tracker.status()
    assert tracker.sam3_ball_calls == 3
    assert tracker.sam2_calls == 2
    assert status["ball_mask_source"].startswith(("sam2:", "sam3:"))
    assert status["ball_mask_sam3_every_n_frames"] == 2
    assert status["ball_mask_sam3_last_request_frame"] == 5
    assert status["ball_mask_box_xyxy"] == [290, 130, 314, 154]


def test_success_tracker_tracks_sam3_request_timings(monkeypatch) -> None:
    module = _load_so101_web_intervene()
    calls = []

    def fake_post(url, json, timeout):
        del timeout
        calls.append({"url": url, "json": json})
        time.sleep(0.02)
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[130:155, 270:295] = 255
        ok, encoded = cv2.imencode(".png", mask)
        assert ok

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "top_mask": {
                        "score": 0.92,
                        "area_px": int(mask.sum()),
                        "box_xyxy": [270, 130, 294, 154],
                        "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                }

        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"
            self.ball_mask_sam3_every_n_frames = 1

        def _should_refresh_ball_with_sam2(self) -> bool:
            return False

    tracker = Tracker()
    tracker.update(_frame(), image_capture_mono=time.monotonic())
    tracker.update(_frame(), image_capture_mono=time.monotonic())
    assert _wait_until(lambda: not tracker.status()["ball_mask_sam3_refresh_running"], timeout_s=2.0)

    status = tracker.status()
    assert calls
    assert status["ball_mask_sam3_async_frame_to_request_s"] is not None
    assert status["ball_mask_sam3_async_request_to_response_s"] is not None
    assert status["ball_mask_sam3_async_request_to_response_s"] >= 0.0
    assert status["ball_mask_sam3_async_frame_to_request_history_s"]
    assert status["ball_mask_sam3_inference_hz"] is not None
    assert status["ball_mask_sam3_inference_hz"] >= 3.0
    assert status["ball_mask_sam3_inference_hz_meets_target"] is True


def test_success_tracker_tracks_sam2_request_timings(monkeypatch) -> None:
    module = _load_so101_web_intervene()
    cup_mask = np.zeros((360, 640), dtype=np.uint8)
    cup_mask[100:180, 240:320] = 255

    def fake_post(url, json, timeout):
        del timeout
        if "track_image" in str(url):
            time.sleep(0.02)
            mask = np.zeros((360, 640), dtype=np.uint8)
            mask[132:157, 272:297] = 255
            ok, encoded = cv2.imencode(".png", mask)
            assert ok

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "top_mask": {
                            "tracked": True,
                            "source": "sam2_video_track",
                            "score": 0.88,
                            "area": int((mask > 0).sum()),
                            "box_xyxy": [272, 132, 296, 156],
                            "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
                        }
                    }

            return Response()

        ok, encoded = cv2.imencode(".png", cup_mask)
        assert ok

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "top_mask": {
                        "score": 0.97,
                        "area_px": int((cup_mask > 0).sum()),
                        "box_xyxy": [240, 100, 319, 179],
                        "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                }

        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"
            self.ball_mask_sam3_every_n_frames = 0
            self.ball_mask_sam2_every_n_frames = 1

    tracker = Tracker()
    tracker.update(_frame(), image_capture_mono=time.monotonic())
    tracker.update(_frame(), image_capture_mono=time.monotonic())
    assert _wait_until(lambda: not tracker.status()["ball_mask_sam2_refresh_running"], timeout_s=2.0)

    status = tracker.status()
    assert status["ball_mask_sam2_async_frame_to_request_s"] is not None
    assert status["ball_mask_sam2_async_request_to_response_s"] is not None
    assert status["ball_mask_sam2_async_request_to_response_s"] >= 0.0
    assert status["ball_mask_sam2_async_frame_to_request_history_s"]


def test_success_tracker_tracks_sam2_vs_sam3_alignment_rate(monkeypatch) -> None:
    module = _load_so101_web_intervene()
    module.SUCCESS_BALL_SAM2_MIN_PRIOR_IOU = 0.0
    module.SUCCESS_BALL_SAM2_MAX_CENTER_SHIFT_PX = 9999.0
    calls = []

    def fake_post(url, json, timeout):
        del timeout
        calls.append({"url": url})
        aligned_mask = np.zeros((360, 640), dtype=np.uint8)
        aligned_mask[130:155, 270:295] = 255
        shifted_mask = np.zeros((360, 640), dtype=np.uint8)
        shifted_mask[200:225, 300:325] = 255

        if len(calls) == 1:
            data_mask = aligned_mask
            frame_idx = 1
        else:
            data_mask = shifted_mask
            frame_idx = 2

        ok, encoded = cv2.imencode(".png", data_mask)
        assert ok

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "top_mask": {
                        "tracked": True,
                        "source": "sam2_video_track",
                        "frame_idx": frame_idx,
                        "score": 0.77,
                        "area": int((data_mask > 0).sum()),
                        "box_xyxy": [0, 0, 1, 1],
                        "mask_png_b64": module.base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                }

        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)

    class Tracker(module.LiveCupSuccessTracker):
        def __init__(self):
            self.sam3_ball_calls = 0
            super().__init__()
            self.ball_sam2_url = "http://sam2.test/api/track_image"
            self.ball_mask_sam3_every_n_frames = 0
            self.ball_mask_sam2_every_n_frames = 1

        def _calculate_sam3_cup_mask(self, rgb, default_area):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[100:180, 240:320] = True
            self.cup_mask_score = 1.0
            self._set_sam3_mask_status("accepted", "test")
            return mask, "sam3:test cup"

        def _calculate_sam3_ball_mask(self, rgb):
            self.sam3_ball_calls += 1
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[130:155, 270:295] = True
            self.ball_mask_score = 0.9
            self._set_ball_sam3_status("accepted", "test")
            return mask, "sam3:blue rubber ball score=0.90"

    tracker = Tracker()
    tracker.update(_frame())
    tracker.update(_frame())
    assert _wait_until(lambda: len(calls) >= 1, timeout_s=2.0)

    updates = 0
    while updates < 6 and tracker.status()["ball_mask_sam2_vs_sam3_align_total"] < 2:
        tracker.update(_frame())
        updates += 1
    assert _wait_until(lambda: len(calls) >= 2, timeout_s=2.0)
    assert _wait_until(lambda: not tracker.status()["ball_mask_sam2_refresh_running"], timeout_s=2.0)

    status = tracker.status()
    assert status["ball_mask_sam2_vs_sam3_align_total"] == 2
    assert status["ball_mask_sam2_vs_sam3_align_count"] == 1
    assert status["ball_mask_sam2_vs_sam3_align_rate"] == 0.5
    assert status["ball_mask_sam2_vs_sam3_last_aligned"] is False
    assert status["ball_mask_sam2_vs_sam3_last_iou"] is not None
    assert status["ball_mask_sam2_vs_sam3_last_iou"] < 0.25
    assert len(status["ball_mask_sam2_vs_sam3_iou_history"]) == 2
    assert status["ball_mask_sam2_inference_hz"] is not None
    assert status["ball_mask_sam2_inference_hz"] >= 3.0
    assert status["ball_mask_sam2_inference_hz_meets_target"] is True
