---
name: coding-practices
description: Project coding and deployment hygiene for blupe-evals/LeLab/Jetson work. Use when Codex is modifying blupe-evals, patching installed LeLab, changing Jetson setup, adding scripts, running services, or deciding where code/configuration should live. Enforces that durable code lives in the repository, environment changes are reproducible from repo-owned scripts or patches, and Jetson state can be wiped and rebuilt without losing work.
---

# Coding Practices

## Core Rule

Keep durable project work in `blupe-evals`.

Installed Jetson files, Python site-packages, generated frontend bundles, running services, local caches, and hand-edited environment state are disposable. Treat them as deployment targets, not sources of truth.

## Required Practice

- Put source code, patches, scripts, configs, tests, and docs in `blupe-evals`.
- If a change must touch installed LeLab or Jetson-local state, also add or update a repo-owned script/patch that can reproduce it.
- Prefer idempotent apply/setup scripts for Jetson changes.
- Keep commands needed to rebuild the Jetson setup documented in repo scripts or docs.
- Do not leave the only copy of a fix inside `/home/andrew/miniforge3/.../site-packages`, frontend `dist/`, `/tmp`, or a running shell session.
- When copying files to the Jetson, copy from repo-owned sources and avoid making the Jetson copy the canonical version.
- Before saying work is done, verify the repo-owned source and the deployed Jetson state are consistent enough that the Jetson could be wiped and restored.

## Jetson And LeLab Work

- LeLab installed-source edits are acceptable only as deployment steps.
- Make LeLab patches durable through repo-owned patch/apply scripts such as `scripts/apply_*.py` or `patches/lelab/*.patch`.
- If frontend production bundles must be rebuilt, keep the source patch and rebuild command reproducible from the repo.
- If services must run on the Jetson, prefer repo-owned service scripts and config files over ad hoc commands.
- If a local tunnel is required for browser access, call it out explicitly as runtime state, not durable setup.

## LeLab Jetson Camera Previews

- Treat Jetson-attached robot cameras as backend resources. When LeLab is viewed through a Mac/browser tunnel, recording previews must come from Jetson backend routes such as `/camera-preview/{index}.jpg`, not browser `getUserMedia`.
- Do not automatically scan cameras while previewing or recording. Routes such as `/available-cameras` may briefly open V4L2 devices and race with preview or recording. Gate scans to explicit rescan actions or only when no saved cameras exist.
- Preview UI must recover from transient camera-open failures. A single `404` or `503` from a startup race must not permanently blank a camera card; add bounded retry logic with cache-busted preview URLs.
- Remove stale browser camera hooks from patched recording camera cards. It is not enough to swap the rendered element if the compiled bundle still calls a browser stream hook in the same component.
- Cache-bust patched LeLab frontend bundles with a new generated asset name or query string so the browser cannot keep stale preview code.
- Make these changes durable through a repo-owned apply script and focused tests that check backend routes, frontend/source patching, compiled bundle patching, retry logic, stale hook removal, and index asset replacement.

## SO101 Eval UI Camera Relay

- Keep LeLab recording UI and SO101 eval UI camera paths distinct. LeLab recording previews use `:8000` backend routes such as `/camera-preview/{index}.jpg`; the eval UI on `:8092` proxies semantic camera URLs from a robot-side MJPEG relay.
- The eval UI launcher must own the camera stack. Do not rely on a remembered manual relay process or stale defaults.
- Start the SO101 camera relay before launching eval UI with the known-good shape: `YAM_control/camera_relay.py --devices 0 2 4 --port 8089 --width 640 --height 360 --fps 30`.
- Pass explicit semantic camera mappings to eval UI: `front=http://127.0.0.1:8089/0`, `side=http://127.0.0.1:8089/2`, and `wrist=http://127.0.0.1:8089/4`.
- Do not use the stale eval defaults `http://127.0.0.1:8080/cam0.mjpg`, `cam1.mjpg`, or `cam2.mjpg` unless a repo-owned service actually provides those exact routes.
- Treat camera resolution as part of the contract. A camera can open but fail to produce frames at the wrong resolution; for the current SO101 three-camera eval path, verify `640x360`, not just that `/dev/video*` opened.
- Keep policy camera selection separate from UI camera availability. The policy may consume only `front,wrist`, while the UI should still expose and verify `front`, `side`, and `wrist`.

## Verification Before Done

Check at least one durable source artifact and one runtime behavior:

- Durable source: `git diff`, relevant script/config/patch, or a focused test.
- Runtime behavior: service health, reachable URL, expected route, dataset listing, frame/image response, or other user-visible behavior.
- For dataset/media UI work, verify the actual media URLs the browser uses, not only the HTML shell or metadata APIs. Check at least one real frame/image/video response for the active dataset and each relevant camera path when cameras are part of the UI.
- For LeLab Jetson camera-preview work, verify the served page loads the patched bundle and all configured camera preview endpoints return real image bytes. At minimum check `/`, the loaded asset name, stale-hook absence in the loaded bundle, and `/camera-preview/0.jpg`, `/camera-preview/2.jpg`, `/camera-preview/4.jpg` when front/side/wrist are configured.
- For SO101 eval UI camera work, verify the relay streams and the eval UI proxy routes. At minimum check `:8089/0`, `:8089/2`, `:8089/4`, then `:8092/camera/front.jpg`, `:8092/camera/side.jpg`, and `:8092/camera/wrist.jpg`; each must return real JPEG bytes before calling the UI ready.
- When imported datasets can represent frames differently than native recordings, test both the metadata row and the media endpoint. Numeric frame ids, filename frame ids, and manifest `path` fields must resolve to real media before calling the UI done.

If a runtime change was made manually on the Jetson and there is no repo-owned way to recreate it, do not call the task complete. Add the missing script/patch first or clearly report the gap.
