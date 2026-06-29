from scripts.apply_lelab_no_browser_patch import (
    DEV_OPEN_OLD,
    HELPER_BLOCK,
    OPEN_HELPER_OLD,
    patch_text,
)


def _source() -> str:
    return "\n".join(
        [
            "import os",
            "",
            "",
            "def _wait_for_port(port):",
            "    return True",
            "",
            "",
            OPEN_HELPER_OLD + "    pass\n",
            "",
            "def _run_dev():",
            DEV_OPEN_OLD,
        ]
    )


def test_no_browser_patch_adds_env_gate() -> None:
    patched, changed = patch_text(_source())

    assert changed is True
    assert 'LELAB_OPEN_BROWSER", "1"' in patched
    assert "Browser auto-open disabled" in patched
    assert "if _should_open_browser()" in patched


def test_no_browser_patch_is_idempotent() -> None:
    patched, changed = patch_text(_source())
    patched_again, changed_again = patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched
