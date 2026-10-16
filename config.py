"""
Centralized configuration for the Line Following Drone project.
All tunable parameters are defined here — no more hardcoded values.
"""

# ==============================================================================
# CONNECTION SETTINGS (Pixhawk ↔ Raspberry Pi)
# ==============================================================================

SERIAL_PORT = '/dev/ttyAMA0'      # Serial port for Pixhawk connection
BAUD_RATE = 57600                  # Baud rate for MAVLink communication
CONNECT_TIMEOUT = 30               # Seconds to wait for Pixhawk connection

# ==============================================================================
# CAMERA SETTINGS
# ==============================================================================

CAMERA_WIDTH = 640                 # Frame width in pixels
CAMERA_HEIGHT = 480                # Frame height in pixels
CAMERA_FPS = 30                    # Frames per second

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

# ==============================================================================
# SAFETY SETTINGS
# ==============================================================================

LOW_BATTERY_VOLTAGE = 10.5         # Trigger RTL below this voltage (for 3S LiPo)
LINE_LOST_TIMEOUT = 3.0            # Seconds without line detection before landing
RC_OVERRIDE_CHANNEL = 5            # RC channel to monitor for manual override
                                   # (PWM > 1800 triggers override)
RC_OVERRIDE_PWM_THRESHOLD = 1800   # PWM threshold for RC override detection
