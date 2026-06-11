#!/usr/bin/env bash
# Bring up the YAM's CAN bus (the gs_usb CANable = can0). Needs sudo (CAP_NET_ADMIN).
# A down interface holds no bitrate, so one must be given: YAM Damiao motors = 1 Mbps.
# Run after a reboot or a CANable replug (the interface comes back DOWN).
#
#   bash YAM_control/setup_can.sh            # brings up can0 @ 1 Mbps
#   bash YAM_control/setup_can.sh can1       # a different interface
set -euo pipefail

IFACE="${1:-can0}"
BITRATE="${2:-1000000}"

echo "[can] bringing up $IFACE @ $BITRATE ..."
sudo ip link set "$IFACE" down 2>/dev/null || true
sudo ip link set "$IFACE" up type can bitrate "$BITRATE"
ip -br link show "$IFACE"
echo "[can] done — expect state UP above."
