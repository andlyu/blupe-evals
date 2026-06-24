# xr-bridge — the XR input appliance (Mac)

Runs XRoboToolkit's Linux-only **PC Service** + **xrobotoolkit_sdk** in an arm64 container
(native on Apple Silicon, same arch+OS as the Orin → replays `docs/ORIN-SETUP.md`), plus
`bridge.py`, which republishes XR state per **`docs/XR-INPUT-BRIDGE.md`**. The eval runs
natively on macOS with `XR_INPUT=bridge` (see `scripts/xrobotoolkit_sdk.py`).

```
Quest ──Wi-Fi──> Mac:63901 ──docker──> PC Service ──127.0.0.1:60061──> sdk ──> bridge.py
                                                                                  │ :8765
                                                            eval (native macOS) <─┘
```

## Build & run

```bash
cd docker/xr-bridge
docker build -t xr-bridge .
docker run --rm --name xr-bridge -p 63901:63901 -p 8765:8765 xr-bridge
```

Then on the Quest: **Network → enter this Mac's Wi-Fi IP** → Controller + Send ON.
(The service's UDP discovery broadcast does not traverse Docker NAT, so the in-headset
auto-connect prompt will NOT appear — manual IP entry is required. That's our normal flow.)

## Smoke test (no Quest needed)

```bash
# bridge alive? (hello + ticks; fields null/zero until a Quest connects)
python3 - <<'EOF'
import socket
s = socket.create_connection(("127.0.0.1", 8765), timeout=5)
f = s.makefile("rb")
for i in range(3):
    print(f.readline().decode().strip()[:120])
EOF
# service port reachable?
nc -z 127.0.0.1 63901 && echo "63901 OK"
```

Container logs (`docker logs -f xr-bridge`) show service restarts, SDK init state, and
client connects. The bridge serves null ticks while the SDK/service is down — consumers
read that as "input lost" (clutch released) per the spec.

## Eval against the bridge (native macOS)

```bash
XR_INPUT=bridge .venv/bin/python scripts/mac_quest_bridge.py --quest-ip <quest-ip> \
    --serve-host <orin-ip>   # omit serve-host for sim-only
```
