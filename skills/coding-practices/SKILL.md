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

## Verification Before Done

Check at least one durable source artifact and one runtime behavior:

- Durable source: `git diff`, relevant script/config/patch, or a focused test.
- Runtime behavior: service health, reachable URL, expected route, dataset listing, frame/image response, or other user-visible behavior.
- For dataset/media UI work, verify the actual media URLs the browser uses, not only the HTML shell or metadata APIs. Check at least one real frame/image/video response for the active dataset and each relevant camera path when cameras are part of the UI.
- When imported datasets can represent frames differently than native recordings, test both the metadata row and the media endpoint. Numeric frame ids, filename frame ids, and manifest `path` fields must resolve to real media before calling the UI done.

If a runtime change was made manually on the Jetson and there is no repo-owned way to recreate it, do not call the task complete. Add the missing script/patch first or clearly report the gap.
