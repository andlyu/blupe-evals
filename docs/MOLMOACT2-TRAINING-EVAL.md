# MolmoAct2 Training / Eval

This is the Workstream C path from `docs/PARALLEL-WORKSTREAMS.md`. It consumes episode folders
from the editor/export lane and never reaches into robot recording internals.

## Dataset Variants

Export the four camera/data variants:

```bash
python scripts/export_molmoact2_variants.py --dry-run
python scripts/export_molmoact2_variants.py --overwrite
```

Outputs are written under `datasets/molmoact2/`:

- `teleop_front`: normal teleop episodes, `observation.images.front`
- `teleop_side`: normal teleop episodes, `observation.images.side`
- `teleop_intervention_front`: teleop plus intervention episodes, `observation.images.front`
- `teleop_intervention_side`: teleop plus intervention episodes, `observation.images.side`

By default, failed episodes are excluded. Add `--include-failures` only for experiments that
intentionally train on failures.

## Single-Instance Full Fine-Tune Launch

Only launch one training instance. Use an A100/H100 box as a persistent worker, then run the
training jobs sequentially inside that instance so cost and environment drift stay bounded.

Recommended first worker:

```text
1x A100 80GB or 1x H100 80GB
```

The first smoke run is full fine-tuning, not LoRA. LeRobot's MolmoAct2 documentation reports
full-model fine-tuning around 48-60 GiB peak memory with gradient checkpointing on H100 80GB,
so avoid 40 GB A100 offers for this path.

Local smoke command shape:

```bash
python scripts/run_molmoact2_full_finetune.py --dry-run
```

Initial blue-ball dataset:

```text
dataset:    andlyu/move_blue_ball_training_v21
revision:   main
checkpoint: allenai/MolmoAct2
steps:      1000
batch:      8
wandb:      project molmoact-so101
images:     observation.images.front, observation.images.wrist
state:      6D SO101 joints, converted to MolmoAct2 v2.1 policy convention
action:     6D SO101 joints, converted to MolmoAct2 v2.1 policy convention
```

Use `--dataset.revision=main` for this dataset. Without an explicit revision, the installed
LeRobot/HF stack can try to resolve a version tag and fail before downloading metadata.
The MolmoAct2 docs on `main` show `--env_eval_freq=-1`, but the installed `lerobot_train`
entrypoint in this checkout accepts `--eval_freq=-1`.

## Experiments LoRA Fine-Tune

The experiments-path LoRA run uses the upstream `allenai/molmoact2/experiments`
launcher with a runtime-registered single LeRobot mixture for:

```text
dataset:    andlyu/move_blue_ball_training_v21
checkpoint: allenai/MolmoAct2-SO100_101
project:    molmoact-so101
mixture:    move_blue_ball
cameras:    observation.images.front, observation.images.wrist
horizon:    30
lora rank:  64
```

Set up the remote worker:

```bash
scp scripts/vast_molmoact2_experiments_setup.sh root@HOST:/workspace/blupe_training/
scp scripts/run_molmoact2_experiments_lora.py root@HOST:/workspace/blupe_training/
ssh root@HOST 'cd /workspace/blupe_training && ./vast_molmoact2_experiments_setup.sh'
```

One-step offline W&B preflight:

```bash
cd /workspace/molmoact2/experiments
/workspace/venv/bin/torchrun --standalone --nproc-per-node=1 \
  /workspace/blupe_training/run_molmoact2_experiments_lora.py \
  --offline-wandb \
  --max-duration 1 \
  --global-batch-size 1 \
  --device-batch-size 1 \
  --num-workers 0 \
  --save-interval 1000 \
  --run-name molmoact2-so101-move-blue-ball-lora-preflight \
  --save-folder /workspace/outputs/molmoact2-so101-move-blue-ball-lora-preflight
```

The validated preflight reached `train/action_flow_loss=0.5294`, used about 34 GiB peak
GPU memory, and saved LoRA and merged LoRA checkpoints.

Real W&B run:

```bash
export WANDB_API_KEY=...       # or copy an existing W&B netrc credential
export WANDB_PROJECT=molmoact-so101
export WANDB_ENTITY=...        # optional

cd /workspace/molmoact2/experiments
/workspace/venv/bin/torchrun --standalone --nproc-per-node=1 \
  /workspace/blupe_training/run_molmoact2_experiments_lora.py \
  --max-duration 1000 \
  --global-batch-size 8 \
  --device-batch-size 1 \
  --num-workers 2 \
  --save-interval 250 \
  --run-name molmoact2-so101-move-blue-ball-lora \
  --save-folder /workspace/outputs/molmoact2-so101-move-blue-ball-lora
```

Keep checkpoint outputs outside this repo, or under ignored artifact storage on the training
machine. Do not commit checkpoints.

## Remote Policy Eval

For hardware evals, keep MolmoAct2 on the A100 and let the Jetson call it over HTTP.
The Jetson remains responsible for SO101 IO, camera capture, safety clipping, teleop
intervention, and recording. The Mac only opens the Jetson browser UI through SSH tunnels.

A100:

```bash
cd /workspace/blupe-evals
export MOLMOACT2_EXPERIMENTS_DIR=/workspace/molmoact2/experiments
export MOLMOACT2_CHECKPOINT_PATH=allenai/MolmoAct2-SO100_101
export MOLMOACT2_NORM_TAG=so100_so101_molmoact2
export MOLMOACT2_IMAGE_KEYS='["observation.images.front","observation.images.wrist"]'
scripts/start_a100_molmoact2_policy_server.sh
```

Use `MOLMOACT2_POLICY_PATH` when the run saved a LeRobot policy directory. Use
`MOLMOACT2_CHECKPOINT_PATH` plus `MOLMOACT2_NORM_TAG` for a base/released MolmoAct2
checkpoint path instead. For the released SO100/SO101 checkpoint, MolmoAct2's LeRobot
docs specify `so100_so101_molmoact2`; the launcher defaults to that tag when
`MOLMOACT2_CHECKPOINT_PATH=allenai/MolmoAct2-SO100_101`.

Jetson, first forward a local port to the A100 policy server:

```bash
ssh -N -L 8202:127.0.0.1:8202 root@A100_HOST
```

Then start the SO101 UI against that forwarded policy URL:

```bash
cd ~/blupe-evals
export SO101_POLICY_URL=http://127.0.0.1:8202
export SO101_POLICY_CAMERAS=front,wrist
scripts/start_jetson_so101_remote_policy.sh
```

Then open LeLab from the Mac and use the Evals tab. The task instruction can stay:

```text
Move to light blue ball, grab it, and move it to the tall black cylinder
```

## Checkpoint Pull

Mirror checkpoints from a gstack machine to the eval host:

```bash
python scripts/pull_molmoact2_checkpoints.py \
  user@gstack-host:/path/to/runs/so101-teleop-front/checkpoints \
  --dest /tmp/molmoact2-checkpoints/teleop_front \
  --interval-s 300
```

## Continuous Eval

Run the eval command once per new checkpoint. The command receives:

- `MOLMOACT2_CHECKPOINT`
- `MOLMOACT2_VARIANT`

Example:

```bash
python scripts/molmoact2_eval_loop.py \
  --checkpoints-root /tmp/molmoact2-checkpoints/teleop_front \
  --variant teleop_front \
  --metrics runs/molmoact2_eval/teleop_front.jsonl \
  --command 'python scripts/so101_web_intervene.py eval-checkpoint --checkpoint "$MOLMOACT2_CHECKPOINT" --camera front' \
  --poll-s 60
```

The metrics file is append-only JSONL with the checkpoint path, variant, return code, elapsed
time, and stdout/stderr tails. Use the same command shape for `side` variants and compare:

- front vs side camera with teleop-only data
- front vs side camera with teleop plus intervention data
- teleop-only vs intervention-augmented for the same camera
