#!/usr/bin/env bash
# Run XRoboToolkit teleop on the Jetson Orin.
#
# Ensures the headless PC Service is running, sets up the (fiddly) library + display
# env, and launches a teleop example in MuJoCo. Run this from a terminal ON THE ORIN'S
# DESKTOP (so DISPLAY=:0 and X authority just work) — not over plain SSH.
#
# Usage:
#   bash ~/blupe-evals/scripts/orin/run_teleop.sh                 # default: dual UR5e example
#   bash ~/blupe-evals/scripts/orin/run_teleop.sh <script.py>     # any teleop script
#   CHECK=1 bash ~/blupe-evals/scripts/orin/run_teleop.sh         # preflight only (no viewer)
set -euo pipefail

APP="$HOME/roboticsservice/opt/apps/roboticsservice"
TELEOP="$HOME/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python"
EXAMPLE="${1:-$TELEOP/scripts/simulation/teleop_dual_ur5e_mujoco.py}"

# 1. Ensure the PC Service (the bridge the Quest connects to) is running, headless.
if ! pgrep -f RoboticsServiceProcess >/dev/null; then
  echo "[run_teleop] starting PC Service (headless)..."
  ( cd "$APP" && \
    LD_LIBRARY_PATH="$APP:$APP/lib:$APP/SDK/arm64" \
    QT_PLUGIN_PATH="$APP/plugins/" QT_QPA_PLATFORM=offscreen \
    nohup ./RoboticsServiceProcess > "$HOME/roboticsservice/service.log" 2>&1 & )
  sleep 4
fi
if pgrep -f RoboticsServiceProcess >/dev/null; then
  echo "[run_teleop] PC Service: UP (Quest -> Network -> $(hostname -I | awk '{print $1}'), Controller + Send ON)"
else
  echo "[run_teleop] PC Service FAILED to start; see $HOME/roboticsservice/service.log" >&2
  exit 1
fi

# 2. Conda env (xr, Python 3.10).
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate xr

# 3. Library + display env:
#    - conda's libstdc++ FIRST so meshcat/icu find CXXABI_1.3.15
#    - the deb's bundled dirs so xrobotoolkit_sdk's libPXREARobotSDK.so + deps resolve
#    - DISPLAY=:0 + XAUTHORITY -> the Orin's attached monitor (X11). The viewer needs a real
#      GL display; ~/.Xauthority lets it open :0 even when launched over SSH.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$APP:$APP/lib:$APP/SDK/arm64"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

if [[ "${CHECK:-0}" == "1" ]]; then
  echo "[run_teleop] preflight: importing the teleop stack (no viewer)..."
  python - <<'PY'
import xrobotoolkit_sdk, mujoco, placo, tyro, meshcat
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController
print("[run_teleop] imports OK — ready to run")
PY
  exit 0
fi

echo "[run_teleop] launching: $EXAMPLE  (DISPLAY=$DISPLAY)"
echo "[run_teleop] hold the grip to clutch; move the controller to drive the arm."
exec python "$EXAMPLE"
