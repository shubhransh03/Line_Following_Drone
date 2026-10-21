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

from dronekit import connect, VehicleMode
from pymavlink import mavutil

import config


class FlightController:
    """
    High-level interface to the Pixhawk flight controller via DroneKit.
    """

    def __init__(self):
        self.vehicle = None
        self._connected = False

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
            'gps_fix': v.gps_0.fix_type,
            'gps_num_sat': v.gps_0.satellites_visible,
            'mode': v.mode.name,
            'armed': v.armed,
            'heading': v.heading,
            'altitude': alt,
            'ekf_ok': v.ekf_ok,
        }

    # ------------------------------------------------------------------
    # Arm & Takeoff
    # ------------------------------------------------------------------

    def arm_and_takeoff(self, target_altitude=None):
        """
        Perform pre-arm checks, arm motors, and takeoff to target altitude.

        Args:
            target_altitude: Altitude in meters (default: config.TARGET_ALTITUDE)

        Returns:
            True if takeoff was successful, False otherwise
        """
        if not self._check_connected():
            return False

        target_altitude = target_altitude or config.TARGET_ALTITUDE

        print("[FC] Running pre-arm checks...")
        timeout_start = time.time()
        while not self.vehicle.is_armable:
            if time.time() - timeout_start > config.CONNECT_TIMEOUT:
                print("[FC] ERROR: Pre-arm checks timed out!")
                return False
            print("[FC]  Waiting for vehicle to initialise...")
            time.sleep(1)

        print("[FC] Setting GUIDED mode...")
        self.vehicle.mode = VehicleMode("GUIDED")

        # Wait for mode change
        timeout_start = time.time()
        while self.vehicle.mode.name != "GUIDED":
            if time.time() - timeout_start > 10:
                print("[FC] ERROR: Failed to switch to GUIDED mode!")
                return False
            time.sleep(0.5)

        print("[FC] Arming motors...")
        self.vehicle.armed = True

        timeout_start = time.time()
        while not self.vehicle.armed:
            if time.time() - timeout_start > 10:
                print("[FC] ERROR: Arming timed out!")
                return False
            print("[FC]  Waiting for arming...")
            time.sleep(1)

        print(f"[FC] Taking off to {target_altitude}m...")
        self.vehicle.simple_takeoff(target_altitude)

        # Wait until target altitude is reached
        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt
            print(f"[FC]  Altitude: {current_alt:.1f}m / {target_altitude}m")

            if current_alt >= target_altitude * 0.95:
                print("[FC] Target altitude reached!")
                break
            time.sleep(1)

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

    def land(self):
        """Switch to LAND mode and wait for the vehicle to disarm."""
        if not self._check_connected():
            return

        print("[FC] Landing...")
        self.vehicle.mode = VehicleMode("LAND")

        # Wait for landing (vehicle disarms when landed)
        while self.vehicle.armed:
            current_alt = self.vehicle.location.global_relative_frame.alt
            print(f"[FC]  Landing... Altitude: {current_alt:.1f}m")
            time.sleep(1)

        print("[FC] Landed and disarmed.")

    def return_to_launch(self):
        """Switch to RTL mode."""
        if not self._check_connected():
            return

        print("[FC] Returning to launch!")
        self.vehicle.mode = VehicleMode("RTL")

    # ------------------------------------------------------------------
    # Safety Checks
    # ------------------------------------------------------------------

    def check_battery(self):
        """
        Check if battery voltage is above safe threshold.

        Returns:
            True if battery is OK, False if critically low
        """
        if not self._check_connected():
            return False

        voltage = self.vehicle.battery.voltage
        if voltage is not None and voltage < config.LOW_BATTERY_VOLTAGE:
            print(f"[FC] WARNING: Battery critically low! {voltage:.1f}V")
            return False
        return True

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
            print("[FC] Closing vehicle connection...")
            try:
                self.vehicle.close()
            except Exception as e:
                print(f"[FC] Warning during close: {e}")
            self.vehicle = None
        self._connected = False
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
