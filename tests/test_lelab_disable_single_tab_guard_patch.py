from scripts.apply_lelab_disable_single_tab_guard_patch import (
    BUNDLE_POPUP_TEXT,
    _patch_dist_index,
    patch_dist_bundle_text,
    patch_source_text,
)


SOURCE = """import { useCallback, useEffect, useRef, useState, ReactNode } from "react";
import { Button } from "@/components/ui/button";

type Peer = { id: string; openedAt: number; lastSeen: number };

const SingleTabGuard = ({ children }: { children: ReactNode }) => {
  const [isPrimary, setIsPrimary] = useState(true);
  const takeOver = useCallback(() => {}, []);

  return (
    <>
      {children}
      {!isPrimary && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
        >
          <div className="mx-4 max-w-md space-y-4 rounded-lg border bg-background p-6 text-center shadow-lg">
            <h2 className="text-lg font-semibold">
              LeLab is already open in another tab
            </h2>
            <p className="text-sm text-muted-foreground">
              Only one tab can control the robot at a time. Switch back to the
              original tab, or take over here — the other tab will lock.
            </p>
            <Button onClick={takeOver}>Use this tab</Button>
          </div>
        </div>
      )}
    </>
  );
};

export default SingleTabGuard;
"""

BUNDLE = (
    "let c=(0,_.useCallback)(()=>{a.current=0},[s]);"
    "return(0,V.jsxs)(V.Fragment,{children:[e,!t&&(0,V.jsx)(`div`,"
    "{className:`fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm`,"
    "role:`dialog`,\"aria-modal\":`true`,children:(0,V.jsxs)(`div`,{children:["
    "(0,V.jsx)(`h2`,{children:`LeLab is already open in another tab`}),"
    "(0,V.jsx)(G,{onClick:c,children:`Use this tab`})]})})]})"
)


def test_disable_single_tab_guard_patch_updates_source() -> None:
    patched, changed = patch_source_text(SOURCE)

    assert changed is True
    assert 'import { ReactNode } from "react";' in patched
    assert "useCallback" not in patched
    assert "@/components/ui/button" not in patched
    assert BUNDLE_POPUP_TEXT not in patched
    assert "return <>{children}</>;" in patched


def test_disable_single_tab_guard_patch_source_is_idempotent() -> None:
    patched, changed = patch_source_text(SOURCE)
    patched_again, changed_again = patch_source_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_disable_single_tab_guard_patch_updates_production_bundle_text() -> None:
    patched, changed = patch_dist_bundle_text(BUNDLE)

    assert changed is True
    assert "children:[e,false&&(0,V.jsx)(`div`" in patched
    assert "children:[e,!t&&(0,V.jsx)(`div`" not in patched
    assert BUNDLE_POPUP_TEXT in patched


def test_disable_single_tab_guard_patch_bundle_is_idempotent() -> None:
    patched, changed = patch_dist_bundle_text(BUNDLE)
    patched_again, changed_again = patch_dist_bundle_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched_again == patched


def test_disable_single_tab_guard_patch_updates_dist_index(tmp_path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    bundle = assets / "index-test-blupe-stop1.js"
    bundle.write_text("patched bundle")
    index = dist / "index.html"
    index.write_text('<script type="module" crossorigin src="/assets/index-test-blupe-stop1.js"></script>')

    changed = _patch_dist_index(dist)

    assert str(assets / "index-test-blupe-stop1-blupe-notab1.js") in changed
    assert str(index) in changed
    assert (assets / "index-test-blupe-stop1-blupe-notab1.js").read_text() == "patched bundle"
    assert 'src="/assets/index-test-blupe-stop1-blupe-notab1.js?blupe_notab=1"' in index.read_text()
