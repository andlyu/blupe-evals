#!/bin/bash
# Supervise the PC Service (restart on exit, it's a closed blob) and run the bridge in
# the foreground (the container lives and dies with the bridge).
APP=/opt/apps/roboticsservice

echo "[entrypoint] starting RoboticsServiceProcess (headless, offscreen)"
(
  cd "$APP" || exit 1
  while true; do
    ./RoboticsServiceProcess
    echo "[entrypoint] service exited (rc=$?); restarting in 2s" >&2
    sleep 2
  done
) &

# Wait (<=30s) for the service's local SDK port before starting the bridge — the bridge
# retries init() forever anyway, this just makes the logs read in order.
for _ in $(seq 1 30); do
  if (exec 3<>/dev/tcp/127.0.0.1/60061) 2>/dev/null; then
    exec 3>&-
    echo "[entrypoint] service is up (127.0.0.1:60061)"
    break
  fi
  sleep 1
done

exec python3 -u /bridge.py
