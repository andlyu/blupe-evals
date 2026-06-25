# Central LeLab With Jetson Stations

Use one central LeLab UI on the operator machine. Each robot arm runs a headless station
service on its Jetson.

```text
Operator machine
  LeLab UI
  blupe-evals station hub (:8099)

Jetson per arm
  SO101 station service (:8091)
  camera MJPEG service (:8080)
  optional policy runner (:8302 or remote)
```

## Why This Split

The central UI should not own robot safety. A browser tab can crash, reload, or disconnect.
The Jetson station process must keep ownership of:

- robot serial/CAN IO;
- camera capture and recording;
- policy watchdogs and action application;
- teleop leases and safety stops;
- raw session/event logs.

LeLab should own the operator workflow:

- station list and health;
- live camera views;
- task assignment;
- recording controls;
- intervention claim/release controls;
- dataset/training/eval dashboards.

## Jetson Station

Run one station per robot on the Jetson:

```bash
python scripts/so101_web_intervene.py \
  --host 0.0.0.0 \
  --port 8091 \
  --policy-url http://127.0.0.1:8302 \
  --camera front=http://127.0.0.1:8080/cam0.mjpg \
  --camera side=http://127.0.0.1:8080/cam1.mjpg \
  --camera wrist=http://127.0.0.1:8080/cam2.mjpg
```

The station exposes:

- `GET /api/status`
- `GET /api/health`
- `GET /camera/{front,side,wrist}.jpg`
- `GET /camera/{front,side,wrist}.mjpg`
- `POST /api/record/start`
- `POST /api/record/stop`
- `POST /api/teleop/claim`
- `POST /api/teleop/heartbeat`
- `POST /api/teleop/release`
- `POST /api/eval/start`
- `POST /api/eval/stop`
- `POST /api/eval/resume`
- `POST /api/eval/clear`

## Operator Station Hub

Create a station config:

```json
{
  "stations": [
    {
      "id": "so101-1",
      "name": "SO101 Station 1",
      "base_url": "http://192.168.0.185:8091",
      "robot_type": "so101_follower",
      "cameras": ["front", "side", "wrist"]
    }
  ]
}
```

Start the hub on the operator machine:

```bash
python scripts/lelab_station_hub.py --stations config/stations.example.json --port 8099
```

Open the first LeLab-facing station dashboard at:

```text
http://localhost:8099/dashboard
```

The dashboard talks only to the hub API. It lists configured stations, polls each station's
health/status through the hub, shows live hub-proxied `front`/`side`/`wrist` camera streams,
and exposes record start/stop plus teleop claim/release controls.

The hub exposes one local API for LeLab:

- `GET /api/stations`
- `GET /api/stations/{station_id}`
- `GET /api/stations/{station_id}/status`
- `GET /api/stations/{station_id}/health`
- `GET /api/stations/{station_id}/camera/{camera}.jpg`
- `GET /api/stations/{station_id}/camera/{camera}.mjpg`
- `POST /api/stations/{station_id}/record/start`
- `POST /api/stations/{station_id}/record/stop`
- `POST /api/stations/{station_id}/teleop/claim`
- `POST /api/stations/{station_id}/teleop/heartbeat`
- `POST /api/stations/{station_id}/teleop/release`
- `POST /api/stations/{station_id}/eval/start`
- `POST /api/stations/{station_id}/eval/stop`
- `POST /api/stations/{station_id}/eval/resume`
- `POST /api/stations/{station_id}/eval/clear`

## LeLab Integration

Short term, keep the station hub in `blupe-evals` and point LeLab UI work at the hub.
This avoids modifying LeLab's robot-local SO101 assumptions before the fleet contract is stable.

Long term, upstream the generic pieces into LeLab:

- station registry;
- remote station health cards;
- live camera panels using hub camera URLs;
- remote recording controls;
- intervention claim/release controls.

Keep Blupe-specific deployment and SO101 camera naming in `blupe-evals`.
