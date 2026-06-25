# LeLab Integration Plan

## Ownership

`blupe-evals` is the owning repo for the SO101 dataset workflow. LeLab remains the UI,
training, and data-collection base we integrate with, but task-specific collection,
episode editing, camera naming, and dataset export live here first.

## Goals

1. Collect normal teleop datasets from continuous SO101 sessions.
2. Collect intervention datasets using policy stop plus teleop handoff.
3. Export camera-specific datasets for MolmoAct2 training/eval.

## Camera Contract

Recordings use semantic camera names, not positional `cam0`/`cam1` names:

- `front`: zero-shot/task-context view.
- `side`: standardized model/eval view.
- `wrist`: consistent manipulation/contact view.

LeRobot image keys should be stable:

- `observation.images.front`
- `observation.images.side`
- `observation.images.wrist`

## Dataset Flow

Raw collection is continuous:

```text
recordings/session_.../
  session_meta.json
  samples.jsonl
  front/
  side/
  wrist/
  events.jsonl
```

The editor turns continuous sessions into episode manifests, then exports clean episode
folders and derived LeRobot datasets:

- normal teleop, front camera
- normal teleop, side camera
- normal plus intervention, front camera
- normal plus intervention, side camera

## Integration Order

1. Add 3-camera semantic config.
2. Smoke test one continuous 3-camera recording.
3. Extend the existing episode viewer into a dataset editor.
4. Reuse subepisode extraction for segment export.
5. Reuse dataset conversion for camera-specific LeRobot datasets.
6. Launch MolmoAct2 training/eval from exported datasets.
