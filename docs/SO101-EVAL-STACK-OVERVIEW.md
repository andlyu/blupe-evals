# SO101 Eval Stack Overview

This repo is the operator/eval layer around SO101 data collection, MolmoAct2
policy execution, SAM-based success tracking, and LeRobot dataset export.

## Runtime Topology

```text
Mac
  Browser -> SO101 eval UI :8092
  Camera relay :8089 -> local USB cameras
  SSH tunnels:
    :8202 -> GPU MolmoAct2 policy server
    :8213 -> GPU SAM3 prompt server
    :8214 -> GPU SAM2 tracker

GPU / Vast 4090
  MolmoAct2 HTTP policy runner :8202
  SAM3 image prompt server :8213
  SAM2 video tracker :8214

SO101 hardware
  follower arm + leader arm connected to the Mac by USB
  front/side/wrist cameras exposed through the local camera relay
```

## Start / Stop

Start the full eval stack:

```bash
scripts/pipeline.sh launch so101-eval
```

Verify the stack before using it:

```bash
scripts/pipeline.sh check so101-eval
```

The check is intentionally a deep check: SAM3 must complete a real
`/api/detect_image` request, not only serve its HTML page. This catches missing
Hugging Face auth for gated `facebook/sam3` before an eval starts.

Stop the full eval stack:

```bash
scripts/pipeline.sh stop so101-eval
```

If the Vast SSH host changes, copy:

```bash
config/so101_eval_stack.env.example
```

to:

```bash
config/so101_eval_stack.local.env
```

and update `SO101_GPU_HOST` / `SO101_GPU_PORT`.

## Main Pieces

- `scripts/so101_web_intervene.py`: live eval UI, robot control, policy calls,
  intervention capture, recording, success tracking, and dataset viewer/export.
- `scripts/launch_so101_eval_ui.sh`: starts local camera relay and the eval UI.
- `scripts/pipeline.sh`: canonical command entrypoint for `launch`, `check`,
  `status`, `stop`, and `restart`.
- `scripts/start_so101_eval_stack.sh`: starts/verifies GPU services, local tunnels,
  camera relay, and eval UI. It syncs the local Hugging Face token to the GPU if
  needed and blocks until SAM3, SAM2, MolmoAct2, cameras, and UI pass readiness.
- `scripts/stop_so101_eval_stack.sh`: stops policy/eval/recording, local processes,
  tunnels, and remote GPU services.
- `scripts/molmoact2_policy_runner.py`: HTTP `/act` and `/health` policy server for
  MolmoAct2.
- `scripts/sam3_prompt_ui.py`: SAM3 image prompt service.
- `scripts/sam2_video_track_ui.py`: SAM2 frame-to-frame mask tracking service.
- `scripts/compress_so101_episodes.py`: compacts raw recordings into LeRobot-style
  datasets with videos.
- `scripts/convert_lerobot_joint_convention.py`: converts recorded SO101 joint data
  to the MolmoAct2 v2.1 joint convention for training.

## Stable Ports

- `8089`: local camera relay
- `8092`: local SO101 eval UI
- `8202`: MolmoAct2 policy server
- `8213`: SAM3 prompt server
- `8214`: SAM2 tracker

## Data Flow

1. Cameras stream through the local relay.
2. The eval UI captures front/wrist frames and current robot state.
3. The UI sends those to MolmoAct2 over `/act`.
4. The UI executes returned joint actions on the follower arm.
5. SAM3 seeds cup masks and periodically re-anchors ball masks; SAM2 refreshes ball masks asynchronously, and the mask MJPEG stream draws fresh camera frames with the latest completed mask.
6. Recordings are written locally, then compacted/exported to a dataset.
7. Converted v2.1 datasets are used for MolmoAct2 LoRA training.

## Related Docs

- `docs/SO101-4090-SAM-EVAL.md`: live eval stack setup and operational details.
- `docs/MOLMOACT2-TRAINING-EVAL.md`: MolmoAct2 training, checkpoints, and inference.
- `docs/PARALLEL-WORKSTREAMS.md`: original workstream plan and division of tasks.
