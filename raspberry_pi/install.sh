#!/usr/bin/env bash
# ==============================================================================
# Autonomous Line-Following Drone — Raspberry Pi Installation Script
# ==============================================================================
# IMPORTANT: This script is intended to run on the physical Raspberry Pi under
# Linux (Raspberry Pi OS). DO NOT EXECUTE ON WINDOWS.
# ==============================================================================

set -euo pipefail

# 1. Guard against running on non-Linux platforms (e.g. Windows Git Bash or Cygwin)
if [[ "${OSTYPE:-}" != "linux"* ]]; then
    echo "======================================================================" >&2
    echo "ERROR: install.sh is designed for Raspberry Pi OS (Linux)." >&2
    echo "Detected platform: ${OSTYPE:-unknown}." >&2
    echo "Aborting execution. Do not run this script on Windows." >&2
    echo "======================================================================" >&2
    exit 1
fi

echo "======================================================================"
echo " Starting Raspberry Pi Drone Environment Setup"
echo "======================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/pi_venv"

echo "[1/5] Updating APT package index and installing system prerequisites..."
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-serial \
    libcamera-tools \
    v4l-utils

echo "[2/5] Creating dedicated Python virtual environment at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    # --system-site-packages allows using system python3-opencv and python3-numpy with libcamera support
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

echo "[3/5] Activating virtual environment and upgrading pip..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

echo "[4/5] Installing Raspberry Pi Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements-pi.txt"

echo "[5/5] Configuring user permissions for UART serial access..."
# Ensure current user has dialout permissions for /dev/ttyAMA0 or /dev/ttyACM0
if ! groups "$USER" | grep -q '\bdialout\b'; then
    echo "Adding $USER to 'dialout' group for serial port access..."
    sudo usermod -a -G dialout "$USER"
    echo "NOTE: You must log out and log back in for dialout group changes to take effect."
fi

echo "======================================================================"
echo " Raspberry Pi Setup Completed Successfully!"
echo "======================================================================"
echo "Next Steps:"
echo " 1. Edit '$SCRIPT_DIR/config.yaml' to set your actual serial port,"
echo "    camera device index, and battery parameters."
echo " 2. Run diagnostics without flight commands:"
echo "    cd '$SCRIPT_DIR' && ./run_diagnostics.sh"
echo "======================================================================"
