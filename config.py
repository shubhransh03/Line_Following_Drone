"""
Centralized configuration for the Line Following Drone project.
All tunable parameters are defined here — no more hardcoded values.
"""

import os
import sys

# ==============================================================================
# CONNECTION SETTINGS (Pixhawk ↔ Companion Computer / Dev PC)
# ==============================================================================

# Serial port or connection string for flight controller:
# Defaults to Raspberry Pi onboard UART (/dev/ttyAMA0).
# Can be overridden via DRONE_CONNECTION_STRING or DRONE_SERIAL_PORT environment variable.
DEFAULT_SERIAL_PORT = '/dev/ttyAMA0'
SERIAL_PORT = os.environ.get('DRONE_CONNECTION_STRING',
                             os.environ.get('DRONE_SERIAL_PORT', DEFAULT_SERIAL_PORT))
BAUD_RATE = int(os.environ.get('DRONE_BAUD_RATE', 57600))  # Baud rate for MAVLink communication
CONNECT_TIMEOUT = 30               # Seconds to wait for Pixhawk connection

# ==============================================================================
# CAMERA SETTINGS
# ==============================================================================

CAMERA_WIDTH = 640                 # Frame width in pixels
CAMERA_HEIGHT = 480                # Frame height in pixels
CAMERA_FPS = 30                    # Frames per second
CAMERA_SOURCE = os.environ.get('CAMERA_SOURCE', 'gstreamer')  # 'gstreamer', 'opencv', or device index

# GStreamer pipeline for Raspberry Pi Camera Module (libcamera stack)
GSTREAMER_PIPELINE = (
    f"libcamerasrc ! "
    f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
    f"framerate={CAMERA_FPS}/1 ! "
    f"videoconvert ! appsink"
)

# ==============================================================================
# VISION / LINE DETECTION SETTINGS
# ==============================================================================

THRESHOLD_VALUE = 60               # Binary threshold (0-255). Pixels darker than
                                   # this become white in the mask (line detected)
BLUR_KERNEL_SIZE = 5               # Gaussian blur kernel size (must be odd)
MIN_CONTOUR_AREA = 500             # Minimum contour area in pixels² to filter noise
SHOW_DEBUG_FRAMES = True           # Show OpenCV debug windows (set False for headless)

# ==============================================================================
# PID CONTROLLER GAINS
# ==============================================================================
# Start conservative and tune in the field:
#   - Increase Kp until the drone starts oscillating, then back off ~30%
#   - Add Ki slowly to eliminate steady-state error
#   - Add Kd to dampen oscillations

PID_KP = 0.3                      # Proportional gain
PID_KI = 0.0                      # Integral gain (start at 0)
PID_KD = 0.1                      # Derivative gain
PID_INTEGRAL_LIMIT = 50.0         # Anti-windup clamp for integral term

# ==============================================================================
# FLIGHT SETTINGS
# ==============================================================================

TARGET_ALTITUDE = 2.0              # Takeoff altitude in meters
FORWARD_SPEED = 0.3                # Forward velocity in m/s during line following
MAX_LATERAL_SPEED = 0.5            # Maximum lateral correction velocity in m/s
VELOCITY_CMD_INTERVAL = 0.1        # Seconds between velocity command updates
TAKEOFF_TIMEOUT = 30.0             # Maximum climb duration in seconds before takeoff timeout
TAKEOFF_POLL_INTERVAL = 0.5        # Altitude polling interval during takeoff climb

# ==============================================================================
# SAFETY SETTINGS
# ==============================================================================

# Battery Monitoring & Safety Settings
LOW_BATTERY_VOLTAGE = 10.5         # Trigger RTL/Land below this voltage (for 3S LiPo)
BATTERY_MIN_VALID_VOLTAGE = 5.0    # Reject sensor readings below this voltage (noise/disconnect)
BATTERY_MAX_VALID_VOLTAGE = 26.0   # Reject sensor readings above this voltage (up to 6S LiPo)
BATTERY_LOW_CONSECUTIVE_COUNT = 3  # Consecutive low readings required to confirm LOW state
BATTERY_STALE_TIMEOUT = 3.0        # Seconds without valid battery telemetry before marked STALE
BATTERY_HYSTERESIS_VOLTAGE = 0.3   # Voltage recovery margin above threshold (10.5 + 0.3 = 10.8V)
BATTERY_RECOVERY_COUNT = 3         # Consecutive readings above recovery margin before returning to NORMAL

LINE_LOST_TIMEOUT = 3.0            # Seconds without line detection before landing
LANDING_TIMEOUT = 60.0             # Maximum duration in seconds for landing touchdown before disarm timeout
RC_OVERRIDE_CHANNEL = 5            # RC channel to monitor for manual override
                                   # (PWM > 1800 triggers override)
RC_OVERRIDE_PWM_THRESHOLD = 1800   # PWM threshold for RC override detection

# Camera Failure & Frame-Drop Safety
CAMERA_FRAME_TIMEOUT = 1.0         # Seconds without a valid frame before commanding zero-velocity hover
CAMERA_FAULT_TIMEOUT = 3.0         # Seconds without a valid frame before aborting and landing
CAMERA_MAX_CONSECUTIVE_FAILURES = 10  # Consecutive dropped reads threshold before warning
CAMERA_RETRY_SLEEP = 0.05          # Seconds to sleep on dropped frame to prevent 100% CPU spin
