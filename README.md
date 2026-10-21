# 🚁 Autonomous Line Following Drone

An autonomous quadcopter that uses onboard computer vision to detect and follow a ground line track. Designed for an onboard Raspberry Pi companion computer communicating via MAVLink with a Pixhawk flight controller running ArduPilot.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Raspberry Pi 4 / Companion Computer                          │
│                                                                             │
│  Downard Camera ──► LineDetector (OpenCV) ──► PIDController                 │
│  (libcamera)        (Moments, Contours,       (Cross-track Error →          │
│                      Defensive Guards)         Lateral Velocity)            │
│                                                       │                     │
│  BatteryMonitor ◄── Telemetry Polling                 ▼                     │
│  (State Machine,    (Voltage, Heartbeat,      FlightController              │
│   Outlier Filter,    Altitude, EKF)          (Bounded Takeoff/Land,         │
│   Hysteresis)                                 Body Velocity, Safety)        │
│                                                       │                     │
│                                                       ▼                     │
│                                                   DroneKit / MAVLink        │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ Serial UART (/dev/ttyAMA0 or USB)
                                        │ 57600 baud
                                        ▼
                             ┌─────────────────────┐
                             │ Pixhawk Flight Ctrl │──► 4× Motors & ESCs
                             │ (ArduPilot Firmware)│
                             └─────────────────────┘
                                        ▲
                                 RC Transmitter
                                (Manual Override)
```

---

## Project Structure

```
Drones/
├── config.py                 # Centralized configuration & environment variable overrides
├── line_detector.py          # Unified computer vision pipeline & contour analyzer
├── battery_monitor.py        # Robust battery telemetry state machine & noise filter
├── pid_controller.py         # Discrete PID controller with anti-windup clamping
├── flight_controller.py      # DroneKit MAVLink interface with bounded pre-arm & takeoff
├── main.py                   # Integrated autonomous flight script & testable loop
├── cam_test.py               # Camera sanity check
├── drone_test.py             # Passive telemetry & heartbeat diagnostic
├── line_detection.py         # Vision pipeline test wrapper
├── raspberry_pi/             # Dedicated Raspberry Pi companion computer deployment package
│   ├── pi_main.py            # Pi entry point (diagnostics, mock, guarded hardware)
│   ├── camera_source.py      # Camera abstraction (OpenCV & Synthetic)
│   ├── line_perception.py    # Enhanced vision with dual-ROI heading estimation
│   ├── pid_controller.py     # Reusable PID controller
│   ├── mavlink_adapter.py    # MAVLink telemetry & command abstraction
│   ├── safety_monitor.py     # Multi-subsystem safety watchdog
│   ├── config_loader.py      # Lightweight YAML loader & strict hardware validator
│   ├── config.yaml           # Deployment configuration with explicit placeholders
│   ├── requirements-pi.txt   # Pi-specific Python dependencies
│   ├── install.sh            # Guarded Linux installation script (run on Pi only)
│   ├── run_diagnostics.sh    # Read-only diagnostic runner
│   ├── README.md             # Complete Pi setup and operating guide
│   └── tests/                # 41 offline unit tests for Pi package
├── simulation/               # Three.js 3D web-based racetrack flight simulator
├── requirements.txt          # Production companion-computer requirements
├── requirements-dev.txt      # Development & offline test requirements
├── test_config.py            # Unit tests for configuration & overrides
├── test_line_detector.py     # Deterministic unit tests for vision pipeline
├── test_battery_monitor.py   # Unit tests for battery telemetry filtering
├── test_camera_failure.py    # Unit tests for camera frame-drop safety
├── test_takeoff_and_prearm.py# Unit tests for takeoff bounding & pre-arm checks
└── test_offline_integration.py# End-to-end integration test suite
```

---

## Offline Testing & Verification

The entire software pipeline can be tested deterministically offline without connecting to physical hardware, opening serial ports, or running live motors.

### Running All Tests
Using the project's dedicated virtual environment (`.diag_venv`):

```powershell
# Run all tests across the repository (99 tests)
& ".\.diag_venv\Scripts\python.exe" -m unittest discover -v -p "test_*.py"

# Or run Raspberry Pi package tests only (41 tests)
& ".\.diag_venv\Scripts\python.exe" -m unittest discover -v -s raspberry_pi/tests -p "test_*.py"
```

All 99 unit and integration tests execute in <0.6 seconds:
- **Root Regression Suite (58 tests)**:
  - `test_config.py`: 4 tests (zero side-effect import, environment overrides, sanity bounds)
  - `test_line_detector.py`: 8 tests (moments, contour filtering, centering, sign convention)
  - `test_battery_monitor.py`: 11 tests (state machine, transient dips, sustained low, hysteresis, stale detection, heartbeat link freshness)
  - `test_camera_failure.py`: 7 tests (frame drop counters, hover timeouts, fault aborts)
  - `test_takeoff_and_prearm.py`: 16 tests (pre-arm sensors, bounded climb, timeout non-actuation, mode/arming failures, bounded landing, disarm confirmation, armed close warning)
  - `test_offline_integration.py`: 12 tests (full flight loop across tracking, line lost, drops, RC override, distinct LOW/STALE/UNKNOWN battery aborts)
- **Raspberry Pi Deployment Suite (41 tests)**:
  - `test_config_validation.py`: 8 tests (YAML parser, section validation, strict hardware gating, warning reporting)
  - `test_camera_source.py`: 6 tests (start/stop, synthetic patterns, frame dimensions, queued drops, recovery, device error handling)
  - `test_line_perception.py`: 7 tests (centered line, left/right sign convention, curved lookahead heading, blank frame, intersection detection, defensive guards)
  - `test_mavlink_adapter.py`: 7 tests (telemetry snapshot, heartbeat staleness, RC override, command recording, disconnect, hardware gating)
  - `test_safety_monitor.py`: 9 tests (nominal CONTINUE, camera hover/abort/recovery, stale telemetry abort, low battery abort, RC override, mode divergence, line loss watchdog)
  - `test_pi_main.py`: 4 tests (diagnostics mode zero flight commands, mock autonomous loop, hardware gating rejection, takeoff failure clean teardown)

---

## Safety Systems & Hardened Logic

1. **Pre-Arm Telemetry Guards**:
   - Telemetry heartbeat must be fresh (<3.0s).
   - Autopilot `is_armable` must report healthy.
   - Relative altitude telemetry must be available.
   - EKF status must report healthy.
   - Initial state must be disarmed.
   - Battery voltage must be valid and above threshold.

2. **Bounded Flight Sequences**:
   - **Takeoff**: Bounded by `TAKEOFF_TIMEOUT` (30.0s). If target altitude is not reached, a non-actuating fault status (`TIMEOUT`) is recorded without issuing unverified flight commands.
   - **Landing**: Bounded by `LANDING_TIMEOUT` (60.0s). Waits for vehicle touchdown and disarm with monotonic timeout to prevent infinite loops.

3. **Multi-Stage Camera Failure Policy**:
   - **Dropped Frame (<1.0s)**: Reset PID derivative/integral terms, throttle CPU spin (`CAMERA_RETRY_SLEEP`), maintain altitude.
   - **Stage 1 Timeout (`CAMERA_FRAME_TIMEOUT` = 1.0s)**: Command zero body velocity (`vx=0, vy=0, vz=0`) to hover safely in place.
   - **Stage 2 Fault (`CAMERA_FAULT_TIMEOUT` = 3.0s)**: Abort flight loop and trigger clean landing.
   - **Continuous Safety Checks**: RC manual override and battery monitors continue executing on every dropped frame iteration.

4. **Battery Telemetry State Machine**:
   - Rejects non-finite values (NaN/Inf), negative voltages, and out-of-range spikes ($<5\text{V}$ or $>26\text{V}$).
   - Filters out transient motor voltage sags by requiring `BATTERY_LOW_CONSECUTIVE_COUNT` (3) sustained low readings before triggering `LOW`.
   - Requires hysteresis margin (`BATTERY_HYSTERESIS_VOLTAGE` = 0.3V) above threshold before recovering to `NORMAL`.
   - Transitions to `STALE` if telemetry packets cease for $>3.0\text{s}$.

5. **Pilot Manual Override**:
   - Monitored continuously via RC channel PWM threshold (`RC_OVERRIDE_PWM_THRESHOLD` = 1800) and autopilot mode changes away from `GUIDED`.

---

## Simulation vs. Real Flight Architecture Note

In `simulation/app.js`, the drone simulator uses track-spline tangent vectors directly from the 3D world (`this.trackHeading`) to smoothly turn the drone's yaw heading along curves.

On the real physical quadcopter:
- The flight controller currently commands body velocities $v_x$ (forward) and $v_y$ (lateral error correction) with fixed yaw ($\omega_z = 0$).
- For straight lines, this follows the line accurately.
- For curves and closed-loop tracks, the physical drone will drift sideways ("crab") unless yaw rotation is actively commanded.
- **Path Forward**: Use `LineDetector.detect().angle` (the orientation angle of the line contour in the camera frame) fed into a dedicated Yaw Rate PID controller to unmask and command angular rate $\omega_z$ in MAVLink `set_position_target_local_ned_encode`.

---

## Live Hardware Pre-Flight Safety Protocol

When bench testing with physical hardware:
1. **PROPELLERS MUST BE REMOVED** at all times during bench testing and calibration.
2. Ensure the vehicle remains disarmed until field-ready.
3. Telemetry connection port can be set via `DRONE_CONNECTION_STRING` (e.g. `COM11` on Windows or `/dev/ttyAMA0` on Linux).
4. Always test RC manual override switch prior to flight.
