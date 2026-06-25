# Parallel Workstreams

Use this same README in every terminal/worktree. Each workstream has a separate branch,
but the project goals and data contracts are shared.

## Common Goal

Build the SO101 data loop in `blupe-evals`, using LeLab as the UI/data-collection and
training base:

1. Collect a normal teleop dataset.
2. Collect an intervention dataset.
3. Train/eval MolmoAct2 models with the `front` camera and the `side` camera.

Code is written locally, committed to GitHub, then pulled onto the Jetson. Do not use
manual `scp` for source code.

## Shared Camera Contract

All new recording, editing, and dataset export code should use semantic camera names:

- `front`: zero-shot/task-context view.
- `side`: standardized model/eval view.
- `wrist`: consistent manipulation/contact view.

Use stable LeRobot image keys:

- `observation.images.front`
- `observation.images.side`
- `observation.images.wrist`

Do not add new code that assumes only `cam0` and `cam1`. If old code still uses those
names, adapt it toward dynamic camera metadata.

## Shared Data Contract

Continuous recordings are the raw source of truth:

```text
recordings/session_.../
  session_meta.json
  samples.jsonl
  events.jsonl
  front/
  side/
  wrist/
```

Episode-ised outputs are derived artifacts:

```text
episodes/episode_.../
  episode_meta.json
  episode_result.json
  lerobot_samples.jsonl
  front/
  side/
  wrist/
```

Segment manifests should be JSON and easy to review:

```json
{
  "recording_dir": "recordings/session_...",
  "segments": [
    {
      "start_s": 1.2,
      "end_s": 8.7,
      "task": "toss the ball",
      "outcome": "success",
      "type": "teleop",
      "notes": ""
    }
  ]
}
```

Valid `outcome` values:

- `success`
- `failure`

Valid `type` values:

- `teleop`
- `intervention`

## Worktree Setup

From the base checkout:

```bash
cd /Users/andrew/Projects/blupe-evals
git switch lelab-3cam-dataset-editor
git pull --ff-only

git worktree add ../blupe-evals-3cam -b lelab-3cam-recording lelab-3cam-dataset-editor
git worktree add ../blupe-evals-editor -b lelab-dataset-editor lelab-3cam-dataset-editor
git worktree add ../blupe-evals-training -b lelab-training-eval lelab-3cam-dataset-editor
```

## Workstream A: 3-Camera Collection

Worktree:

```bash
cd /Users/andrew/Projects/blupe-evals-3cam
```

Branch:

```text
lelab-3cam-recording
```

Scope:

- Add/configure 3 semantic cameras: `front`, `side`, `wrist`.
- Update continuous SO101 recording to save all configured cameras.
- Persist camera metadata in recording/session metadata.
- Ensure state/action samples stay timestamped and aligned with camera frames.
- Smoke test one short recording with all available cameras.

Primary files likely involved:

- `scripts/so101_web_intervene.py`
- recording/camera config docs or new config files
- any camera helper/config code added for this workflow

Deliverable:

- A short continuous recording that includes state/action plus `front`, `side`, and `wrist`
  camera metadata/streams, or a clear blocker if the third camera is not physically present.

Avoid:

- Training logic.
- Dataset editor UI beyond what is necessary to verify recording.
- Hardcoded two-camera assumptions.

## Workstream B: Dataset Editor / Episode-isation

Worktree:

```bash
cd /Users/andrew/Projects/blupe-evals-editor
```

Branch:

```text
lelab-dataset-editor
```

Scope:

- Extend the existing episode viewer into a dataset editor.
- Load a continuous recording.
- Show synced playback for cameras defined in metadata, not a hardcoded camera list.
- Add segment editing: start/end, task, outcome, type, notes.
- Save/load segment manifests.
- Export segments into clean episode folders.
- Reuse `scripts/extract_so101_subepisodes.py` where possible.

Primary files likely involved:

- `scripts/so101_web_intervene.py`
- `scripts/extract_so101_subepisodes.py`
- any new manifest/editor helper code

Deliverable:

- A UI or API path that turns a long recording plus segment labels into episode folders.

Avoid:

- Camera device probing/collection internals except where needed to read metadata.
- Training launch logic.
- Rewriting extraction from scratch if existing extraction can be adapted.

## Workstream C: Training / Eval

Worktree:

```bash
cd /Users/andrew/Projects/blupe-evals-training
```

Branch:

```text
lelab-training-eval
```

Scope:

- Define dataset export variants for MolmoAct2:
  - normal teleop, `front`
  - normal teleop, `side`
  - normal plus intervention, `front`
  - normal plus intervention, `side`
- Set up or document `gstack` training launch commands.
- Add scripts/configs for pulling checkpoints periodically.
- Add or document continuous eval loop and metrics logging.
- Compare `front` vs `side`, normal-only vs intervention-augmented.

Primary files likely involved:

- dataset conversion/export scripts
- MolmoAct2/gstack launch scripts/configs
- eval runner scripts/docs

Deliverable:

- A smoke training launch path and an eval/checkpoint loop plan or script.

Avoid:

- Robot camera/recording internals.
- Dataset editor UI, except for consuming exported datasets/manifests.

## Intervention Events

Intervention data should be represented with structured events so the editor can create
episodes later:

```json
{"event": "policy_attempt_start", "t": 0.0}
{"event": "failure_detected", "t": 12.3}
{"event": "policy_stopped", "t": 12.7}
{"event": "intervention_claimed", "t": 12.9}
{"event": "intervention_start", "t": 13.0}
{"event": "intervention_end", "t": 21.5}
{"event": "intervention_outcome", "t": 21.5, "outcome": "success"}
{"event": "policy_resume", "t": 22.0}
```

Workstream B consumes these events for episode-isation. Workstream C consumes the exported
datasets, not the raw intervention UI state.

## Git Rules

Keep each workstream scoped to its branch. Commit only the files that belong to that lane:

```bash
git status
git add <specific files>
git commit -m "<scope>: <summary>"
git push -u origin <branch>
```

Jetson sync is always via Git:

```bash
ssh -i ~/.ssh/id_ed25519_jetson_nopass -o IdentitiesOnly=yes andrew@192.168.0.185
cd ~/blupe-evals
git fetch origin
git switch <branch>
git pull --ff-only
```

Do not commit:

- raw recordings
- episode frame dumps
- LeRobot dataset payloads
- model checkpoints
- secrets or tokens

## Existing Context

- The base plan is in `docs/LELAB-INTEGRATION-PLAN.md`.
- Existing code already includes continuous recording, subepisode extraction, a basic
  episode viewer, and LeRobot dataset conversion.
- The Jetson previously had local SO101 work saved as:

```text
stash@{0}: pre-lelab-branch-jetson-dirty
```

Inspect that stash before assuming a feature is missing.
