#!/usr/bin/env bash
# ==============================================================================
# Autonomous Line-Following Drone — Diagnostic Runner (Raspberry Pi)
# ==============================================================================
# Executes read-only health checks on camera capture and MAVLink telemetry.
# NEVER arms motors, commands takeoff, or moves vehicle actuators.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/pi_venv/bin/python"

# Use virtual environment python if present, else fallback to system python3
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_CMD="$VENV_PYTHON"
else
    PYTHON_CMD="python3"
fi

echo "======================================================================"
echo " Running Raspberry Pi Drone Diagnostics (Read-Only Mode)"
echo "======================================================================"

exec "$PYTHON_CMD" "$SCRIPT_DIR/pi_main.py" --mode diagnostics "$@"
