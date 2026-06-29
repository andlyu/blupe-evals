from scripts.apply_lerobot_opencv_transient_read_tolerance_patch import OLD_BLOCK, patch_text


def test_opencv_read_tolerance_patch_updates_failure_policy() -> None:
    patched, changed = patch_text(OLD_BLOCK)

    assert changed is True
    assert "failure_start_time = None" in patched
    assert "max_failure_duration_s = 5.0" in patched
    assert "time.sleep(0.02)" in patched
    assert "exceeded maximum read failure duration" in patched


def test_opencv_read_tolerance_patch_is_idempotent() -> None:
    patched, changed = patch_text(OLD_BLOCK)
    patched_again, changed_again = patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched
