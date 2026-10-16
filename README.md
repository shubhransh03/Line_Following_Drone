# 🚁 Line Following Drone

An autonomous quadcopter that uses computer vision to detect and follow a line on the ground. Built with a Raspberry Pi companion computer, Pixhawk flight controller, and OpenCV.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Raspberry Pi                       │
│                                                      │
│  Pi Camera V4 ──► OpenCV ──► PID Controller          │
│                  (line       (error → velocity)      │
│                  detection)         │                 │
│                                     ▼                │
│                                 DroneKit             │
│                                (MAVLink)             │
└──────────────────────────────────┬───────────────────┘
                                   │ Serial /dev/ttyAMA0
                                   │ 57600 baud
                                   ▼
                            ┌─────────────┐
                            │   Pixhawk   │──► 4x Motors
                            │     FC      │
                            └─────────────┘
                                   ▲
                            RC Transmitter
                           (manual override)
```

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Frame | S500 quadcopter |
| Flight Controller | Pixhawk (with MS5611 barometer) |
| Companion Computer | Raspberry Pi 4/5 |
| Camera | Raspberry Pi Camera Module V4 |
| Motors | 4× Brushless + 4× ESCs |
| Battery | ~3300 mAh LiPo (3S) |
| RC | Transmitter + Receiver |
| Calibration | QGroundControl |

## Software Requirements

- Python 3.11+
- OpenCV (`python3-opencv`)
- DroneKit (`dronekit`)
- PyMAVLink (`pymavlink`)
- GStreamer (`gstreamer1.0-tools` + plugins)
- pyserial
- v4l-utils

## Project Structure

```
Line_Following_Drone/
├── config.py              # All tunable parameters (PID, camera, flight, safety)
├── pid_controller.py      # PID controller class with anti-windup
├── flight_controller.py   # DroneKit wrapper (connect, takeoff, velocity, land)
├── main.py                # Integrated autonomous flight script
├── cam_test.py            # Camera test script
├── drone_test.py          # Pixhawk connection test script
├── line_detection.py      # Vision pipeline test script
└── README.md              # This file
```

## Setup (Raspberry Pi)

```bash
# 1. System updates
sudo apt update && sudo apt upgrade -y

# 2. Install system dependencies
sudo apt install python3-opencv python3-venv v4l-utils \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad -y

# 3. Create virtual environment
python3 -m venv drone_env
source drone_env/bin/activate

# 4. Install Python packages
pip install dronekit pymavlink pyserial future

# 5. Add user to dialout group (for serial access)
sudo usermod -a -G dialout $USER

# 6. Reboot
sudo reboot
```

## Usage

### Step 1: Test the Camera

```bash
python cam_test.py
```
Verify you see a live video feed. Press ESC to exit.

### Step 2: Test Pixhawk Connection

```bash
python drone_test.py
```
Verify telemetry is displayed (battery, GPS, mode, etc.).

### Step 3: Test Line Detection

```bash
python line_detection.py
```
Point the camera at a dark line on a light surface. Verify the red centroid dot tracks the line and the error value is printed.

### Step 4: Vision-Only Mode (No Drone)

```bash
python main.py --no-fly
```
Runs the full vision + PID pipeline without connecting to the drone. Useful for tuning the threshold and PID gains before real flight.

### Step 5: Autonomous Flight

> ⚠️ **WARNING**: Ensure you are in a safe, open area. Keep the RC transmitter ready for manual override.

```bash
python main.py
```

The drone will:
1. Connect to Pixhawk
2. Arm and takeoff to the configured altitude
3. Follow the detected line using PID-controlled velocity commands
4. Land automatically if the line is lost for too long
5. Emergency land on low battery

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_ALTITUDE` | 2.0 m | Takeoff altitude |
| `FORWARD_SPEED` | 0.3 m/s | Forward velocity during line following |
| `MAX_LATERAL_SPEED` | 0.5 m/s | Maximum lateral correction speed |
| `PID_KP` | 0.3 | Proportional gain |
| `PID_KI` | 0.0 | Integral gain |
| `PID_KD` | 0.1 | Derivative gain |
| `THRESHOLD_VALUE` | 60 | Binary threshold for line detection |
| `MIN_CONTOUR_AREA` | 500 | Minimum contour area (noise filter) |
| `LINE_LOST_TIMEOUT` | 3.0 s | Land if line lost for this long |
| `LOW_BATTERY_VOLTAGE` | 10.5 V | Emergency land threshold |

## PID Tuning Guide

1. Start with `Kp=0.3, Ki=0, Kd=0` in `--no-fly` mode
2. Watch the PID output on screen — it should respond proportionally to error
3. Increase `Kp` until the output oscillates, then reduce by ~30%
4. Add `Kd=0.1` to dampen oscillations
5. Only add `Ki` if there's persistent steady-state error (start very small: 0.01)
6. Test in flight at low altitude (2m) with very slow forward speed (0.2 m/s)

## Safety Features

- **Low Battery**: Automatically lands when voltage drops below threshold
- **Line Lost Timeout**: Hovers then lands if line is lost for too long
- **RC Override**: Detects when pilot takes manual control via RC switch
- **Mode Change Detection**: Stops autonomous control if mode changes externally
- **Ctrl+C Handling**: Clean shutdown with guaranteed landing sequence
- **Fallback Landing**: Attempts RTL if normal landing fails
