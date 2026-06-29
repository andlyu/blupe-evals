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

## LeLab Mac-Local Camera Previews

- Browser `deviceId` values are opaque per-origin values. Do not seed Mac-local LeLab robot records with camera names as `device_id`; seed `camera_index` and let the UI bind to browser `deviceId` at runtime.
- Verify both browser-facing camera discovery and Python/OpenCV capture. `/available-cameras` can list AVFoundation devices even when macOS privacy blocks `cv2.VideoCapture`.
- Browser camera permission is per-origin. Google Meet camera access does not imply Chrome has granted `http://127.0.0.1:8000` or `http://localhost:8000` access.
- Before calling Mac-local recording ready, run a direct OpenCV frame-read check for the selected indices. If it reports camera access denied, grant Camera permission to the app launching LeLab/Python in macOS System Settings, then retest.

## SO101 Eval UI Camera Relay

- Keep LeLab recording UI and SO101 eval UI camera paths distinct. LeLab recording previews use `:8000` backend routes such as `/camera-preview/{index}.jpg`; the eval UI on `:8092` proxies semantic camera URLs from a robot-side MJPEG relay.
- The eval UI launcher must own the camera stack. Do not rely on a remembered manual relay process or stale defaults.
- On macOS, list AVFoundation cameras by name before choosing indices: `ffmpeg -f avfoundation -list_devices true -i ""`. Do not blindly probe camera indices because this can activate FaceTime or Continuity/iPhone cameras.
- Start the SO101 camera relay with USB camera indices only. For the current Mac SO101 setup, the three EMEET USB cameras are AVFoundation indices `0`, `1`, and `2`: `YAM_control/camera_relay.py --devices 0 1 2 --port 8089 --width 640 --height 360 --fps 30`.
- Pass explicit semantic camera mappings to eval UI. For the current Mac SO101 setup: `front=http://127.0.0.1:8089/2`, `side=http://127.0.0.1:8089/1`, and `wrist=http://127.0.0.1:8089/0`.
- Do not use the stale eval defaults `http://127.0.0.1:8080/cam0.mjpg`, `cam1.mjpg`, or `cam2.mjpg` unless a repo-owned service actually provides those exact routes.
- Treat camera resolution as part of the contract. A camera can open but fail to produce frames at the wrong resolution; for the current SO101 three-camera eval path, verify `640x360`, not just that `/dev/video*` opened.
- Keep policy camera selection separate from UI camera availability. The policy may consume only `front,wrist`, while the UI should still expose and verify `front`, `side`, and `wrist`.

## MolmoAct2 A100 Inference And Eval UI

- Start the SO101 eval UI and the A100 policy server independently. The UI at `http://127.0.0.1:8092/#setup` should not wait for the A100 model to finish loading; it only needs `--policy-url http://127.0.0.1:8202` configured before the operator clicks Start MolmoAct.
- Treat the local `:8202` SSH tunnel as runtime state. Verify both the Mac side (`curl http://127.0.0.1:8202/health`) and the A100 side when debugging, because a listening tunnel does not prove the remote policy server is healthy.
- Use the checkpoint loader that matches the artifact format. MolmoAct2 training exports such as `step250-merged` contain `config.yaml` and `model.pt`; run them through `--checkpoint-path ... --norm-tag <training_tag>`, not `--policy-path`, which expects a LeRobot policy directory with `config.json`.
- For released base checkpoints such as `allenai/MolmoAct2-SO100_101`, use the checkpoint path plus the official norm tag, usually `so100_so101_molmoact2`.
- For actual LeRobot-saved policy directories with `config.json`, use `--policy-path`.
- Pick the inference norm tag deliberately. For custom SO101 ball/cup runs, use the custom dataset tag when that tag carries the trained prompt/camera metadata, even if its state/action stats were overridden to the base SO100/SO101 stats during training.
- Do not infer service health from `lsof` alone. A stale or wedged listener can appear bound while HTTP requests fail. Verify the route the browser or robot code will actually call.
- For realtime chunking, compare A100 query latency against action-chunk execution time before changing behavior. With ~1.5-2.0s MolmoAct2 queries and 30 actions, 30 Hz execution leaves unavoidable underruns; 7.5 Hz makes the chunk last ~4s, so dispatching the next query halfway through leaves enough time for it to finish.
- Keep policy timing knobs in the dashboard, not hardcoded only in env vars. The operator should be able to adjust execution Hz, realtime chunking on/off, and query fraction before clicking Start MolmoAct or Start Recording/Eval.

## SO101 Dataset And Media Viewers

- Keep raw episode viewers and compacted LeRobot dataset viewers conceptually separate. If `/episodes` is serving raw JSONL/JPEG frames, do not assume compacted MP4 behavior or smooth playback.
- Use clear SO101 dataset storage roles. Raw editable/eval episodes belong under `episodes/<dataset_slug>/...`; compacted LeRobot exports belong under `datasets/lerobot/<dataset_slug>`; imported HF recordings may live under `recordings/`. Do not move files just to make the editor work; fix the editor/API to understand these roles.
- Expose dataset identity in episode APIs. Raw local eval episodes must return `dataset_name`, `dataset_slug`, and `episodes_root` so the editor can group `episodes/so101-ball-cup-eval/*` as one dataset instead of treating each episode as its own source or confusing it with an HF repo.
- Keep the editor's "dataset repo" input semantics explicit. A Hugging Face repo id such as `andlyu/foo` is not the same thing as a local dataset slug such as `so101-ball-cup-eval`; if the UI accepts both, label/status them distinctly and verify the source picker shows all episodes for the selected local slug.
- Show an explicit compacted state in dataset/editor UIs. Episodes should indicate `Compacted` when `compressed_dataset` points to a local LeRobot root, and `Dataset not compacted` when the viewer is using raw frame folders.
- Do not build episode pickers by parsing every sample or frame JSON row. Use bounded directory discovery and cheap line counts for summaries; load full sample/frame JSON only for the selected episode.
- Avoid recursive scans through camera frame folders when discovering recordings. Limit discovery to known recording roots and one dataset-folder level unless a task specifically requires a deeper import.
- When a frame appears missing at a timestamp, verify both the metadata row and the exact browser media route before blaming compaction. For 30 Hz data, timestamp `6.600s` should map to frame index `198`; test `/episode_frame?...&frame=198` or the actual URL the browser uses.
- Validate compacted LeRobot videos directly with `ffprobe` or equivalent. Check video keys, frame count, fps, duration, and dimensions before concluding an export is corrupt.
- Preserve LeRobot dataset video schema on append. A dataset written with `front,wrist` video keys must keep appending `front,wrist`; include optional `side` only when creating a compatible new dataset or when the existing dataset already has `side`.

## Verification Before Done

Check at least one durable source artifact and one runtime behavior:

- Durable source: `git diff`, relevant script/config/patch, or a focused test.
- Runtime behavior: service health, reachable URL, expected route, dataset listing, frame/image response, or other user-visible behavior.
- For dataset/media UI work, verify the actual media URLs the browser uses, not only the HTML shell or metadata APIs. Check at least one real frame/image/video response for the active dataset and each relevant camera path when cameras are part of the UI.
- For SO101 dataset/editor changes, verify `/api/episodes` responds quickly, local dataset slugs such as `so101-ball-cup-eval` group all expected raw episodes, a selected raw frame URL returns image bytes, and any compacted dataset MP4s have the expected frame count/fps/dimensions. Do not call playback fixed based only on metadata.
- For LeLab Jetson camera-preview work, verify the served page loads the patched bundle and all configured camera preview endpoints return real image bytes. At minimum check `/`, the loaded asset name, stale-hook absence in the loaded bundle, and `/camera-preview/0.jpg`, `/camera-preview/2.jpg`, `/camera-preview/4.jpg` when front/side/wrist are configured.
- For SO101 eval UI camera work, verify the relay streams and the eval UI proxy routes. At minimum check `:8089/0`, `:8089/1`, `:8089/2`, then `:8092/camera/front.jpg`, `:8092/camera/side.jpg`, and `:8092/camera/wrist.jpg`; each must return real JPEG bytes before calling the UI ready, and visually confirm `side` is the wide USB table/arm view rather than FaceTime or Continuity/iPhone.
- For MolmoAct2 eval launches, verify `:8202/health` reports the intended `loaded_from`, `norm_tag`, `runner_api`, and `image_keys`, and verify `:8092/api/status` or `/` responds before opening `http://127.0.0.1:8092/#setup`.
- When imported datasets can represent frames differently than native recordings, test both the metadata row and the media endpoint. Numeric frame ids, filename frame ids, and manifest `path` fields must resolve to real media before calling the UI done.

If a runtime change was made manually on the Jetson and there is no repo-owned way to recreate it, do not call the task complete. Add the missing script/patch first or clearly report the gap.
