from scripts.apply_lelab_recording_stability_patch import (
    DATASET_CONFIG_OLD,
    EXCEPT_OLD,
    IMPORT_OLD,
    STATUS_OLD,
    WORKER_LOG_OLD,
    patch_text,
)


def _source() -> str:
    return "\n".join(
        [
            IMPORT_OLD,
            DATASET_CONFIG_OLD,
            WORKER_LOG_OLD,
            EXCEPT_OLD,
            STATUS_OLD,
        ]
    )


def test_recording_stability_patch_caps_encoder_and_logs_config() -> None:
    patched, changed = patch_text(_source())

    assert changed is True
    assert "import traceback" in patched
    assert 'vcodec="auto"' in patched
    assert "encoder_threads=2" in patched
    assert "encoder_queue_maxsize=180" in patched
    assert "Recording encoder settings" in patched


def test_recording_stability_patch_exposes_exception_details() -> None:
    patched, _ = patch_text(_source())

    assert "traceback.format_exc()" in patched
    assert "Recording exception detail" in patched
    assert "error_type" in patched
    assert 'status["traceback"]' in patched


def test_recording_stability_patch_is_idempotent() -> None:
    patched, changed = patch_text(_source())
    patched_again, changed_again = patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched
