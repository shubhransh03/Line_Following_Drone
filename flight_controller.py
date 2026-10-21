"""
Flight Controller wrapper for the Line Following Drone.

Provides high-level flight operations on top of DroneKit:
  - Connection with retry logic
  - Arm and takeoff sequence
  - Body-frame velocity commands for line following
  - Safety monitoring (battery, RC override)
  - Clean landing and shutdown
"""

import time
import sys
import collections
import collections.abc

# Compatibility shim for Python 3.10+ where collections.MutableMapping moved to collections.abc
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping

try:
    from dronekit import connect, VehicleMode
    DRONEKIT_AVAILABLE = True
except ImportError:
    connect = None
    VehicleMode = None
    DRONEKIT_AVAILABLE = False
from pymavlink import mavutil

import config
from battery_monitor import BatteryMonitor, BatteryState


class FlightController:
    """
    High-level interface to the Pixhawk flight controller via DroneKit.
    """

    def __init__(self, battery_monitor=None):
        self.vehicle = None
        self._connected = False
        self.last_takeoff_status = None
        self.last_landing_status = None
        self.last_prearm_issues = []
        self.last_prearm_missing = []
        self.battery_monitor = battery_monitor if battery_monitor is not None else BatteryMonitor()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, port=None, baud=None, timeout=None):
        """
        Connect to the Pixhawk flight controller.

        Args:
            port: Serial port (default: config.SERIAL_PORT)
            baud: Baud rate (default: config.BAUD_RATE)
            timeout: Connection timeout in seconds (default: config.CONNECT_TIMEOUT)

        Returns:
            True if connected successfully, False otherwise
        """
        port = port or config.SERIAL_PORT
        baud = baud or config.BAUD_RATE
        timeout = timeout or config.CONNECT_TIMEOUT

        if not DRONEKIT_AVAILABLE:
            print("[FC] ERROR: dronekit is not installed. Install dronekit for physical flight connection.")
            self._connected = False
            return False

        print(f"[FC] Connecting to Pixhawk on {port} at {baud} baud...")

        try:
            self.vehicle = connect(port, baud=baud, wait_ready=True,
                                   heartbeat_timeout=timeout)
            self._connected = True
            print("[FC] Connected successfully!")
            self.print_telemetry()
            return True
        except Exception as e:
            print(f"[FC] Connection failed: {e}")
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def print_telemetry(self):
        """Print current vehicle telemetry."""
        if not self._check_connected():
            return

        v = self.vehicle
        print(f"[FC] ---- Telemetry ----")
        print(f"[FC]  Battery  : {v.battery}")
        print(f"[FC]  GPS      : {v.gps_0}")
        print(f"[FC]  Mode     : {v.mode.name}")
        print(f"[FC]  Armed    : {v.armed}")
        print(f"[FC]  Heading  : {v.heading}°")
        if v.location.global_relative_frame:
            print(f"[FC]  Altitude : {v.location.global_relative_frame.alt:.1f} m")
        print(f"[FC]  EKF OK   : {v.ekf_ok}")
        print(f"[FC] -------------------")

    def get_telemetry(self):
        """
        Return telemetry as a dictionary.

        Returns:
            dict with keys: battery_voltage, battery_level, gps_fix,
            gps_num_sat, mode, armed, heading, altitude, ekf_ok
        """
        if not self._check_connected():
            return None

        v = self.vehicle
        alt = 0.0
        if v.location.global_relative_frame:
            alt = v.location.global_relative_frame.alt or 0.0

        return {
            'battery_voltage': v.battery.voltage,
            'battery_level': v.battery.level,
            'battery_state': self.battery_monitor.state,
            'gps_fix': v.gps_0.fix_type,
            'gps_num_sat': v.gps_0.satellites_visible,
            'mode': v.mode.name,
            'armed': v.armed,
            'heading': v.heading,
            'altitude': alt,
            'ekf_ok': v.ekf_ok,
        }

    # ------------------------------------------------------------------
    # Pre-Arm Health Checks
    # ------------------------------------------------------------------

    def check_prearm_health(self):
        """
        Evaluate pre-arm sensor and telemetry health indicators.

        Returns:
            Tuple of (is_healthy: bool, issues: list of str, missing_telemetry: list of str)
        """
        issues = []
        missing = []

        if not self._check_connected():
            issues.append("Vehicle not connected")
            return False, issues, missing

        v = self.vehicle

        # 1. Heartbeat freshness
        last_hb = getattr(v, 'last_heartbeat', None)
        if last_hb is None:
            missing.append("Heartbeat telemetry unavailable")
        elif last_hb > 3.0:
            issues.append(f"Stale heartbeat: {last_hb:.1f}s since last packet")

        # 2. Must be disarmed initially
        if getattr(v, 'armed', False):
            issues.append("Vehicle is already armed prior to takeoff sequence")

        # 3. Autopilot pre-arm check
        is_armable = getattr(v, 'is_armable', None)
        if is_armable is None:
            missing.append("Autopilot is_armable telemetry unavailable")
        elif not is_armable:
            issues.append("Autopilot internal pre-arm check failed (is_armable=False)")

        # 4. Battery telemetry
        batt = getattr(v, 'battery', None)
        voltage = getattr(batt, 'voltage', None) if batt else None
        if voltage is None:
            missing.append("Battery voltage telemetry unavailable")
        else:
            is_valid, reason = self.battery_monitor.validate_voltage(voltage)
            if not is_valid:
                issues.append(f"Invalid battery telemetry: {reason}")
            elif voltage < config.LOW_BATTERY_VOLTAGE:
                issues.append(f"Battery critically low: {voltage:.2f}V (threshold {config.LOW_BATTERY_VOLTAGE}V)")
            else:
                self.battery_monitor.update(voltage)

        # 5. Altitude telemetry
        loc = getattr(v, 'location', None)
        rel_frame = getattr(loc, 'global_relative_frame', None) if loc else None
        alt = getattr(rel_frame, 'alt', None) if rel_frame else None
        if alt is None:
            missing.append("Relative altitude telemetry unavailable")

        # 6. EKF status (if reported)
        ekf_ok = getattr(v, 'ekf_ok', None)
        if ekf_ok is None:
            missing.append("EKF health telemetry unavailable")
        elif not ekf_ok:
            issues.append("Autopilot EKF reporting not healthy (ekf_ok=False)")

        is_healthy = (len(issues) == 0 and len(missing) == 0)
        self.last_prearm_issues = issues
        self.last_prearm_missing = missing

        return is_healthy, issues, missing

    # ------------------------------------------------------------------
    # Arm & Takeoff
    # ------------------------------------------------------------------

    def arm_and_takeoff(self, target_altitude=None, timeout=None,
                         clock_fn=time.monotonic, sleep_fn=time.sleep):
        """
        Perform pre-arm health checks, arm motors, and takeoff to target altitude with bounded timeout.

        Args:
            target_altitude: Altitude in meters (default: config.TARGET_ALTITUDE)
            timeout: Maximum climb duration in seconds (default: config.TAKEOFF_TIMEOUT)
            clock_fn: Monotonic clock callable (injectable for testing)
            sleep_fn: Sleep/polling callable (injectable for testing)

        Returns:
            True if target altitude reached within timeout, False otherwise.
        """
        if not self._check_connected():
            self.last_takeoff_status = "NOT_CONNECTED"
            return False

        target_altitude = target_altitude or config.TARGET_ALTITUDE
        timeout = timeout or config.TAKEOFF_TIMEOUT

        print("[FC] Running pre-arm health checks...")
        healthy, issues, missing = self.check_prearm_health()
        if not healthy:
            print(f"[FC] ERROR: Pre-arm health check failed! Issues: {issues}, Missing: {missing}")
            self.last_takeoff_status = "PREARM_FAILED"
            return False

        print("[FC] Setting GUIDED mode...")
        if VehicleMode is not None:
            self.vehicle.mode = VehicleMode("GUIDED")
        else:
            self.vehicle.mode = "GUIDED"

        # Wait for mode change with bounded timeout (10s)
        timeout_start = clock_fn()
        while getattr(self.vehicle.mode, 'name', self.vehicle.mode) != "GUIDED":
            if clock_fn() - timeout_start > 10.0:
                print("[FC] ERROR: Failed to switch to GUIDED mode!")
                self.last_takeoff_status = "MODE_CHANGE_FAILED"
                return False
            sleep_fn(0.5)

        print("[FC] Arming motors...")
        self.vehicle.armed = True

        timeout_start = clock_fn()
        while not self.vehicle.armed:
            if clock_fn() - timeout_start > 10.0:
                print("[FC] ERROR: Arming timed out!")
                self.last_takeoff_status = "ARMING_FAILED"
                return False
            print("[FC]  Waiting for arming...")
            sleep_fn(1.0)

        print(f"[FC] Taking off to {target_altitude}m (timeout: {timeout:.1f}s)...")
        self.vehicle.simple_takeoff(target_altitude)

        # Wait until target altitude is reached with bounded timeout
        climb_start = clock_fn()
        target_threshold = target_altitude * 0.95
        altitude_reached = False

        while clock_fn() - climb_start < timeout:
            loc = getattr(self.vehicle, 'location', None)
            rel_frame = getattr(loc, 'global_relative_frame', None) if loc else None
            current_alt = getattr(rel_frame, 'alt', None) if rel_frame else None

            if current_alt is None:
                print("[FC] WARNING: Altitude telemetry unavailable during climb...")
            else:
                print(f"[FC]  Climb Altitude: {current_alt:.1f}m / {target_altitude:.1f}m")
                if current_alt >= target_threshold:
                    altitude_reached = True
                    print("[FC] Target altitude reached successfully!")
                    break

            sleep_fn(config.TAKEOFF_POLL_INTERVAL)

        if not altitude_reached:
            elapsed = clock_fn() - climb_start
            print(f"[FC] ERROR: Takeoff timed out after {elapsed:.1f}s without reaching {target_threshold:.1f}m target!")
            # Non-actuating fault result: record status and return False without issuing unverified LAND
            self.last_takeoff_status = "TIMEOUT"
            return False

        self.last_takeoff_status = "SUCCESS"
        return True

    # ------------------------------------------------------------------
    # Velocity Control
    # ------------------------------------------------------------------

    def send_body_velocity(self, vx, vy, vz=0):
        """
        Send velocity command in body frame (relative to drone heading).

        This is the primary control method for line following:
          - vx > 0 → forward (nose direction)
          - vx < 0 → backward
          - vy > 0 → right
          - vy < 0 → left
          - vz > 0 → down (NED convention)
          - vz < 0 → up

        Args:
            vx: Forward/backward velocity in m/s
            vy: Left/right velocity in m/s
            vz: Up/down velocity in m/s (default: 0 = maintain altitude)
        """
        if not self._check_connected():
            return

        # Clamp lateral velocity
        vy = max(-config.MAX_LATERAL_SPEED,
                 min(config.MAX_LATERAL_SPEED, vy))

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms (not used)
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Body-relative frame
            0b0000111111000111,  # type_mask (only velocities enabled)
            0, 0, 0,            # x, y, z positions (not used)
            vx, vy, vz,         # x, y, z velocity in m/s
            0, 0, 0,            # x, y, z acceleration (not used)
            0, 0                 # yaw, yaw_rate (not used)
        )
        self.vehicle.send_mavlink(msg)

    # ------------------------------------------------------------------
    # Landing
    # ------------------------------------------------------------------

    def land(self, timeout=None, clock_fn=time.monotonic, sleep_fn=time.sleep):
        """
        Switch to LAND mode and wait with bounded timeout for the vehicle to disarm.

        Args:
            timeout: Maximum seconds to wait for disarm (default: config.LANDING_TIMEOUT)
            clock_fn: Monotonic clock callable (injectable for testing)
            sleep_fn: Sleep/polling callable (injectable for testing)

        Returns:
            True if landed and disarmed within timeout, False otherwise.
        """
        if not self._check_connected():
            self.last_landing_status = "NOT_CONNECTED"
            return False

        timeout = timeout or config.LANDING_TIMEOUT
        print("[FC] Landing...")
        if VehicleMode is not None:
            self.vehicle.mode = VehicleMode("LAND")
        else:
            self.vehicle.mode = "LAND"

        start_time = clock_fn()
        # Bounded wait for landing (vehicle disarms when landed)
        while getattr(self.vehicle, 'armed', False):
            elapsed = clock_fn() - start_time
            if elapsed > timeout:
                print(f"[FC] ERROR: Landing timed out after {elapsed:.1f}s without disarming!")
                self.last_landing_status = "TIMEOUT"
                return False

            loc = getattr(self.vehicle, 'location', None)
            rel_frame = getattr(loc, 'global_relative_frame', None) if loc else None
            current_alt = getattr(rel_frame, 'alt', None) if rel_frame else None
            alt_str = f"{current_alt:.1f}m" if current_alt is not None else "unknown"
            print(f"[FC]  Landing... Altitude: {alt_str} (elapsed: {elapsed:.1f}s)")
            sleep_fn(1.0)

        print("[FC] Landed and disarmed.")
        self.last_landing_status = "SUCCESS"
        return True

    def return_to_launch(self):
        """Switch to RTL mode."""
        if not self._check_connected():
            return

        print("[FC] Returning to launch!")
        if VehicleMode is not None:
            self.vehicle.mode = VehicleMode("RTL")
        else:
            self.vehicle.mode = "RTL"

    # ------------------------------------------------------------------
    # Safety Checks
    # ------------------------------------------------------------------

    def check_battery(self, clock_fn=None):
        """
        Check if battery voltage is above safe threshold using BatteryMonitor.
        Guards against stale telemetry link where DroneKit retains cached voltage.

        Returns:
            True if battery is OK, False if critically low, stale, or link lost
        """
        if not self._check_connected():
            return False

        # Link freshness check: if autopilot heartbeat has stopped, cached battery data is invalid
        last_hb = getattr(self.vehicle, 'last_heartbeat', None)
        if last_hb is not None and last_hb > config.BATTERY_STALE_TIMEOUT:
            print(f"[FC] WARNING: Autopilot heartbeat is stale ({last_hb:.1f}s)! Battery data unverified.")
            self.battery_monitor.update(None)
            return False

        batt = getattr(self.vehicle, 'battery', None)
        voltage = getattr(batt, 'voltage', None) if batt else None

        state = self.battery_monitor.update(voltage)
        if state == BatteryState.LOW:
            print(f"[FC] WARNING: Battery critically low! {voltage:.1f}V (consecutive low: {self.battery_monitor.consecutive_low_count})")
            return False
        elif state == BatteryState.STALE:
            print("[FC] WARNING: Battery telemetry is stale!")
            return False

        return self.battery_monitor.is_ok()

    def check_rc_override(self):
        """
        Check if RC override switch is activated.

        Returns:
            True if pilot has triggered manual override, False otherwise
        """
        if not self._check_connected():
            return False

        try:
            channels = self.vehicle.channels
            ch = str(config.RC_OVERRIDE_CHANNEL)
            if ch in channels and channels[ch] is not None:
                if int(channels[ch]) > config.RC_OVERRIDE_PWM_THRESHOLD:
                    print("[FC] RC override detected! Pilot taking control.")
                    return True
        except Exception:
            pass  # RC data may not be available

        return False

    def is_mode_changed(self):
        """
        Check if the flight mode has been changed externally (e.g., by RC).

        Returns:
            True if no longer in GUIDED mode, False otherwise
        """
        if not self._check_connected():
            return True

        return self.vehicle.mode.name != "GUIDED"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Cleanly disconnect from the vehicle."""
        if self.vehicle is not None:
            is_armed = getattr(self.vehicle, 'armed', False)
            if is_armed:
                print("[FC] WARNING: Closing telemetry connection while vehicle is still armed!")
            print("[FC] Closing vehicle connection...")
            try:
                self.vehicle.close()
            except Exception as e:
                print(f"[FC] Warning during close: {e}")
            self.vehicle = None
        self._connected = False
        self.battery_monitor.reset()
        print("[FC] Disconnected.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_connected(self):
        """Check if vehicle is connected."""
        if not self._connected or self.vehicle is None:
            print("[FC] ERROR: Not connected to vehicle!")
            return False
        return True
