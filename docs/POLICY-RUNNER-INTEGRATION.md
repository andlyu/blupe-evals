# Policy Runner Integration

LeLab/station code and model policy code should stay separated by a small runtime API.

## Repository Roles

`blupe-evals` owns the robot-side station workflow:

- SO101 camera configuration and semantic names.
- Robot state/action IO.
- Recording raw continuous sessions.
- Teleop intervention and event logging.
- Segment/export tooling.
- A generic policy-runner HTTP client.

LeLab-facing integration code starts in this repo under:

```text
src/blupe_evals/station/
  cameras.py
  policy_client.py
```

SO101-specific scripts consume these modules. If a primitive becomes generally useful across
robots and projects, it can later move upstream into LeLab with a smaller, proven API surface.

The policy/model repo owns model-specific work:

- MolmoAct2 model code and checkpoints.
- Model-specific preprocessing and action decoding.
- Offline policy evaluation against recorded datasets.
- Training and checkpoint publishing.
- An HTTP `/act` runner that implements the station policy protocol.

LeLab should be treated as an upstream framework or dependency. If changes are generally useful
to LeLab, make them upstreamable in a LeLab branch or PR. Keep SO101/Blupe-specific deployment,
camera names, station scripts, and dataset glue in `blupe-evals`.

## Runtime Split

Run two processes on the Jetson when local model inference is feasible:

```text
LeLab station process
  scripts/so101_web_intervene.py --policy-url http://127.0.0.1:8302

Policy runner process
  MolmoAct2 or another policy backend exposing /act and /health
```

If MolmoAct2 needs a remote GPU, keep the station process on the Jetson and point
`--policy-url` at a local proxy or remote policy server. The Jetson still owns safety,
recording, and teleop handoff.

## Policy API

The station sends:

```http
POST /act
```

```json
{
  "schema_version": 1,
  "robot_type": "so101_follower",
  "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
  "instruction": "move the object",
  "state": [0.0, -90.0, 70.0, 0.0, -45.0, 0.0],
  "state_units": "degrees",
  "images": {
    "front": {"encoding": "jpeg_base64", "data": "..."},
    "side": {"encoding": "jpeg_base64", "data": "..."}
  },
  "camera_order": ["front", "side"],
  "action_units": "degrees"
}
```

The runner returns:

```json
{
  "policy": "molmoact2",
  "action_units": "degrees",
  "latency_s": 1.23,
  "actions": [[0.0, -88.0, 69.0, 1.0, -44.0, 5.0]]
}
```

`actions` may contain one action or an action chunk. The station applies the chunk with its
own watchdogs and stops safely if the runner times out, crashes, or returns invalid actions.

## Evaluation Boundary

Model/checkpoint evaluation belongs in the policy repo when it is offline evaluation over
datasets or checkpoints.

Robot-side eval orchestration belongs in `blupe-evals` because it uses live cameras, robot IO,
safety stops, intervention collection, and recording metadata.
