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

## Long-Term Fleet Synchronization

The single-Jetson workflow should grow into a fleet workflow where many SO101 stations
run continuously and publish enough state for an operator to intervene only when needed.
Assume 10+ SO101 arms, each with its own Jetson, cameras, policy runner, and local
recording buffer.

### Synchronization Boundaries

- Code and configuration: Git branches and tagged releases. Jetsons pull code with
  `git fetch` plus `git pull --ff-only`; no manual `scp` for source files.
- Raw recordings: stay local first on each Jetson, then upload or rsync as data artifacts.
  Raw camera/state streams should never be committed to Git.
- Curated datasets: publish to Hugging Face dataset repos or equivalent object storage.
- Model checkpoints: publish through the training artifact path, such as Hugging Face model
  repos, W&B artifacts, or `gstack` job outputs.
- Fleet state: a central monitor records station health, current policy, camera status,
  recording status, latest attempt outcome, and whether intervention is requested.

### Station Agent

Each Jetson should run a station agent responsible for:

- registering `station_id`, `robot_id`, camera names, current git SHA, and active model;
- starting/stopping policy eval loops;
- keeping a rolling continuous recording buffer;
- reporting successes, failures, and uncertain states;
- exposing a safe remote intervention lease;
- uploading completed raw sessions, edited episodes, and metrics.

The agent should be able to run unattended. Operators should not need to SSH into a Jetson
for normal collection, intervention, or sync.

### Monitoring UI

The central UI should show all stations in a table:

- online/offline
- arm/camera health
- active policy/checkpoint
- current task
- attempts, successes, failures
- intervention requested
- recording/upload backlog
- current git SHA and dirty/clean status

An operator should be able to click one failing station, inspect its live cameras, claim
intervention control, teleop, release control, and return the station to autonomous eval.

### Intervention Lease Workflow

When a policy fails or enters an uncertain state:

1. The station marks `intervention_requested`.
2. The central UI alerts an operator.
3. The operator claims an intervention lease for that station.
4. The station stops or pauses policy control.
5. The operator teleops the arm while the station records intervention data.
6. The operator releases the lease with an outcome label.
7. The station resumes policy eval or resets for the next attempt.

Every intervention should produce structured events:

- `policy_attempt_start`
- `failure_detected` or `uncertain_detected`
- `policy_stopped`
- `intervention_claimed`
- `intervention_start`
- `intervention_end`
- `intervention_outcome`
- `policy_resume` or `reset`

These events become the source of truth for episode-ising intervention data.

### Data Products

Each station should upload:

- raw continuous sessions;
- proposed episode manifests;
- curated normal episodes;
- curated intervention episodes;
- eval metrics by policy checkpoint and camera subset.

The central dataset builder should then create training datasets by selecting station,
task, camera subset, and intervention count, for example:

- normal-only, front camera;
- normal-only, side camera;
- normal plus 5 intervention episodes, front camera;
- normal plus 10 intervention episodes, side camera.

## Integration Order

1. Add 3-camera semantic config.
2. Smoke test one continuous 3-camera recording.
3. Extend the existing episode viewer into a dataset editor.
4. Reuse subepisode extraction for segment export.
5. Reuse dataset conversion for camera-specific LeRobot datasets.
6. Launch MolmoAct2 training/eval from exported datasets.
