# Raspberry Pi Deployment Package — Autonomous Line-Following Drone

This package contains the companion computer software stack for the Autonomous Line-Following Drone, designed to run on a Raspberry Pi mounted on the aircraft and connected to the Pixhawk flight controller via UART/USB and to a downward-facing camera.

---

## 1. Supported Operating Systems & Python Versions

- **Primary Supported OS**: Raspberry Pi OS (64-bit), Debian 12 (Bookworm) or Debian 11 (Bullseye).
- **Supported Python Versions**: Python 3.9, 3.10, 3.11.
  - *Note on Python 3.11+*: Uses standard `collections.abc` shims for DroneKit/PyMAVLink compatibility.
- **Still Requires Hardware Confirmation**:
  - Exact Raspberry Pi model (Pi 4B 2GB/4GB or Pi 5).
  - Physical camera hardware: CSI Ribbon Camera (Pi Camera Module v2 / v3) via `libcamera` vs USB UVC Camera via V4L2 (`/dev/video0`).
  - Flight controller serial interface: Pixhawk `TELEM2` connected to Raspberry Pi GPIO UART (pins 14/15, `/dev/ttyAMA0` / `/dev/serial0`) vs USB cable (`/dev/ttyACM0`).

---

## 2. Directory Structure

```
raspberry_pi/
├── __init__.py
├── pi_main.py             # Main entry point (--mode diagnostics | mock | hardware)
├── camera_source.py       # OpenCV & Synthetic camera abstraction
├── line_perception.py     # Vision perception pipeline with dual-ROI analysis
├── pid_controller.py      # Reusable PID controller
├── mavlink_adapter.py     # Abstract MAVLink telemetry & command interface
├── safety_monitor.py      # Multi-subsystem safety monitor & watchdog
├── config_loader.py       # YAML parser and strict configuration validator
├── config.yaml            # Centralized deployment configuration
├── requirements-pi.txt    # Pi-specific Python dependencies
├── install.sh             # Linux installation script (run on Pi only)
├── run_diagnostics.sh     # Diagnostic runner (read-only, no flight commands)
├── README.md              # This deployment guide
└── tests/                 # Comprehensive offline test suite
    ├── __init__.py
    ├── test_config_validation.py
    ├── test_camera_source.py
    ├── test_line_perception.py
    ├── test_mavlink_adapter.py
    ├── test_safety_monitor.py
    └── test_pi_main.py
```

---

## 3. Installation on the Raspberry Pi

> [!WARNING]
> Do NOT run `install.sh` on a Windows PC. It is strictly meant for the Linux environment on the Raspberry Pi.

### Step 1: Copy or Clone Repository to the Pi
From your development machine, copy the project directory to the Raspberry Pi:
```bash
# Example via rsync:
rsync -avz --exclude '.diag_venv' --exclude '__pycache__' \
    C:/Users/Admin/Downloads/Drones/ pi@raspberrypi.local:~/Drones/
```

### Step 2: Run Installation Script
SSH into the Raspberry Pi:
```bash
ssh pi@raspberrypi.local
cd ~/Drones/raspberry_pi
chmod +x install.sh run_diagnostics.sh
./install.sh
```

The script will:
1. Install APT system packages: `python3-opencv`, `python3-numpy`, `python3-yaml`, `python3-serial`, `python3-venv`, `libcamera-tools`.
2. Create a virtual environment at `raspberry_pi/pi_venv` with `--system-site-packages`.
3. Install required Python packages from `requirements-pi.txt`.
4. Add the current user to the `dialout` group for serial port access.

Log out and back in if `dialout` group membership was updated.

---

## 4. Hardware Configuration (`config.yaml`)

Edit `config.yaml` to specify your actual physical vehicle parameters:

### 1. Serial Port Identification
To find the physical port connected to the flight controller:
```bash
# If using GPIO UART (TELEM2 connected to Pi GPIO 14/15):
ls -l /dev/serial0 /dev/ttyAMA0

# If using USB cable from Pixhawk to Pi:
ls -l /dev/ttyACM* /dev/ttyUSB*
dmesg | grep -i tty
```
In `config.yaml`, update `telemetry.port`:
```yaml
telemetry:
  port: "/dev/ttyAMA0"   # or "/dev/ttyACM0"
  baud_rate: 57600       # Match ArduPilot SERIAL2_BAUD parameter
```

### 2. Camera Configuration
- **For USB Camera**: Check device path:
  ```bash
  v4l2-ctl --list-devices
  ```
  Set in `config.yaml`:
  ```yaml
  camera:
    source_type: "usb"
    device_index: 0      # or "/dev/video0"
  ```
- **For CSI Pi Camera (libcamera)**: Verify camera detection:
  ```bash
  libcamera-hello --list-cameras
  ```
  Set in `config.yaml`:
  ```yaml
  camera:
    source_type: "csi"
    device_index: "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink"
  ```

### 3. Battery Configuration
Specify physical LiPo battery parameters:
```yaml
safety:
  battery:
    cell_count: 3        # 3 for 3S LiPo, 4 for 4S LiPo
    low_voltage: 10.5    # Critical landing threshold (10.5V for 3S; 14.0V for 4S)
```

---

## 5. Execution Modes

### Mode 1: Diagnostics (Safe, Read-Only)
Inspects camera capture and MAVLink telemetry without arming motors or moving actuators:
```bash
./run_diagnostics.sh
# Or directly:
python3 pi_main.py --mode diagnostics
```

### Mode 2: Mock Autonomous Simulation (Offline Testing)
Runs full autonomous flight loop against synthetic camera and mock MAVLink:
```bash
python3 pi_main.py --mode mock --cycles 50
```

### Mode 3: Hardware Flight Mode (Guarded)
> [!CAUTION]
> Hardware mode is strictly guarded. It will refuse to start if any required hardware setting is unconfigured.
```bash
python3 pi_main.py --mode hardware --config config.yaml
```

---

## 6. Pre-Flight Physical Safety Verification Checklist

Before enabling hardware flight mode:
1. **PROPELLERS REMOVED**: Always remove all propellers during bench and telemetry testing.
2. **RC Transmitter Manual Override**:
   - Power on the radio transmitter.
   - Verify Channel 5 or flight mode switch switches ArduPilot between `GUIDED` and `STABILIZE` / `ALT_HOLD`.
   - Confirm in diagnostics that flipping the switch immediately releases autonomous control.
3. **Camera Orientation & Focus**:
   - Verify the camera is pointed straight down.
   - Verify that image top corresponds to vehicle nose forward.
   - Verify lens focus and contrast against the tracking tape.
4. **Indoor Non-GPS Positioning**:
   - In indoor environments without GPS, ArduPilot requires an optical flow sensor (e.g. HereFlow) and a downward rangefinder (lidar/sonar) calibrated and healthy (`EKF_OK = True`) to execute `GUIDED` takeoff and velocity control.
5. **Autopilot Failsafe Settings**:
   - Verify `FS_GCS_ENABLE` in ArduPilot is set to `1` (RTL or Land on telemetry link loss).

---

## 7. Running Offline Tests

Run all unit tests in the deployment package without physical hardware:
```bash
python3 -m unittest discover -v -s tests -p "test_*.py"
```
