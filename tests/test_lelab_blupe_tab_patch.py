from scripts.apply_lelab_blupe_tab_patch import (
    DEFAULT_BLUPE_EVALS_URL,
    SOURCE_TRAINING_CARD,
    patch_dist_bundle_text,
    patch_landing_source,
)


def test_blupe_tab_patch_updates_landing_source() -> None:
    source = (
        '  const handleTrainingClick = () => navigate("/training");\n'
        '<div className="grid grid-cols-1 md:grid-cols-3 gap-3">\n'
        f"{SOURCE_TRAINING_CARD}"
    )

    patched, changed = patch_landing_source(source)

    assert changed is True
    assert "handleBlupeEvalsClick" in patched
    assert "VITE_BLUPE_EVALS_URL" in patched
    assert DEFAULT_BLUPE_EVALS_URL in patched
    assert "md:grid-cols-4" in patched
    assert "BluPe Evals" in patched
    assert "Open Evals" in patched


def test_blupe_tab_patch_is_idempotent_for_landing_source() -> None:
    source = (
        '  const handleTrainingClick = () => navigate("/training");\n'
        '<div className="grid grid-cols-1 md:grid-cols-3 gap-3">\n'
        f"{SOURCE_TRAINING_CARD}"
    )
    patched, changed = patch_landing_source(source)
    patched_again, changed_again = patch_landing_source(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_blupe_tab_patch_updates_production_bundle_text() -> None:
    bundle = (
        "F=()=>window.location.assign(`http://127.0.0.1:8092/episodes`),"
        "(0,V.jsxs)(`div`,{className:`grid grid-cols-1 md:grid-cols-3 gap-3`,children:["
        "(0,V.jsxs)(`div`,{className:`bg-gray-800 rounded-lg border border-gray-700 p-3 "
        "flex flex-col gap-2`,children:[(0,V.jsx)(`h3`,{className:`font-semibold text-lg "
        "text-left h-10 flex items-center`,children:`Create a model`}),(0,V.jsx)(G,{onClick:P,"
        "className:`w-full bg-green-500 hover:bg-green-600 text-white`,children:`Training`})]})"
        "]})"
    )

    patched, changed = patch_dist_bundle_text(bundle)

    assert changed is True
    assert f"window.location.assign(`{DEFAULT_BLUPE_EVALS_URL}`)" in patched
    assert "md:grid-cols-4" in patched
    assert "BluPe Evals" in patched
    assert "Open Evals" in patched
