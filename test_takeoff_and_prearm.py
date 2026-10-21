"""
test_takeoff_and_prearm.py — Deterministic Unit Tests for Takeoff & Pre-Arm Safety

Tests bounded takeoff climb, altitude telemetry validation, pre-arm sensor guards,
and non-actuating timeout faults using mocked vehicle objects and simulated clocks.
"""

import unittest
import config
from flight_controller import FlightController


class MockRelativeFrame:
    def __init__(self, alt=0.0):
        self.alt = alt


class MockLocation:
    def __init__(self, alt=0.0):
        self.global_relative_frame = MockRelativeFrame(alt) if alt is not None else None


class MockBattery:
    def __init__(self, voltage=12.4, level=90):
        self.voltage = voltage
        self.level = level


class MockVehicleMode:
    def __init__(self, name="STABILIZE"):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"MockVehicleMode('{self.name}')"


class MockVehicle:
    """Mock DroneKit vehicle simulating ArduPilot state and telemetry."""
    def __init__(self, initial_alt=0.0, initial_mode="STABILIZE"):
        self.is_armable = True
        self._armed = False
        self._mode = MockVehicleMode(initial_mode)
        self.allow_mode_change = True
        self.allow_arming = True
        self.location = MockLocation(initial_alt)
        self.battery = MockBattery(voltage=12.4, level=90)
        self.last_heartbeat = 0.5
        self.ekf_ok = True
        self.takeoff_alt_commanded = None
        self.mode_changes = []
        self.messages_sent = []

    @property
    def armed(self):
        return self._armed

    @armed.setter
    def armed(self, val):
        if getattr(self, 'allow_arming', True):
            self._armed = bool(val)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        if getattr(self, 'allow_mode_change', True):
            if isinstance(val, str):
                self._mode = MockVehicleMode(val)
            else:
                self._mode = val

    def simple_takeoff(self, target_alt):
        self.takeoff_alt_commanded = target_alt

    def send_mavlink(self, msg):
        self.messages_sent.append(msg)


class SimulatedClock:
    """Monotonic clock simulator advancing deterministically on demand."""
    def __init__(self, start_time=100.0, step=0.1):
        self.time = start_time
        self.step = step

    def __call__(self):
        t = self.time
        self.time += self.step
        return t

    def advance(self, seconds):
        self.time += seconds


def dummy_sleep(seconds):
    """No-op sleep replacement for fast test execution."""
    pass


class TestTakeoffAndPreArm(unittest.TestCase):
    """Unit tests for FlightController takeoff bounding and pre-arm health checks."""

    def setUp(self):
        self.fc = FlightController()
        self.fc.vehicle = MockVehicle(initial_alt=0.0, initial_mode="STABILIZE")
        self.fc._connected = True

    # -------------------------------------------------------------------------
    # Test 1: Takeoff reaches altitude criterion before timeout
    # -------------------------------------------------------------------------
    def test_takeoff_reaches_criterion_successfully(self):
        """Takeoff succeeds when vehicle reaches 95% of target altitude within timeout."""
        # Simulated climb: vehicle starts at 0m, climb step brings it to 1.95m (target 2.0m * 0.95 = 1.90m)
        clock = SimulatedClock(step=0.2)

        def climbing_sleep(seconds):
            # On each poll, drone climbs 0.5m
            current = self.fc.vehicle.location.global_relative_frame.alt
            self.fc.vehicle.location.global_relative_frame.alt = current + 0.5

        success = self.fc.arm_and_takeoff(
            target_altitude=2.0,
            timeout=10.0,
            clock_fn=clock,
            sleep_fn=climbing_sleep
        )

        self.assertTrue(success, "Takeoff must return True when altitude is reached")
        self.assertEqual(self.fc.last_takeoff_status, "SUCCESS")
        self.assertAlmostEqual(self.fc.vehicle.takeoff_alt_commanded, 2.0)
        self.assertTrue(self.fc.vehicle.armed)
        self.assertEqual(self.fc.vehicle.mode.name, "GUIDED")

    # -------------------------------------------------------------------------
    # Test 2: Takeoff times out if altitude never reaches criterion
    # -------------------------------------------------------------------------
    def test_takeoff_times_out_if_altitude_stalled(self):
        """Takeoff must time out and return False if altitude never reaches target threshold."""
        # Altitude stays stalled at 0.4m (never reaches 1.9m)
        self.fc.vehicle.location.global_relative_frame.alt = 0.4
        clock = SimulatedClock(step=1.0)  # each poll advances 1.0s

        success = self.fc.arm_and_takeoff(
            target_altitude=2.0,
            timeout=5.0,  # 5-second timeout
            clock_fn=clock,
            sleep_fn=dummy_sleep
        )

        self.assertFalse(success, "Takeoff must return False on timeout")
        self.assertEqual(self.fc.last_takeoff_status, "TIMEOUT")

    # -------------------------------------------------------------------------
    # Test 3: Missing altitude telemetry handled safely during climb
    # -------------------------------------------------------------------------
    def test_missing_altitude_telemetry_handled_safely(self):
        """If altitude telemetry becomes None during climb, takeoff handles it without crashing."""
        clock = SimulatedClock(step=1.0)
        # Set altitude frame to None
        self.fc.vehicle.location.global_relative_frame = None

        success = self.fc.arm_and_takeoff(
            target_altitude=2.0,
            timeout=3.0,
            clock_fn=clock,
            sleep_fn=dummy_sleep
        )

        self.assertFalse(success)
        # Pre-arm health check should reject missing altitude before even arming!
        self.assertEqual(self.fc.last_takeoff_status, "PREARM_FAILED")

    # -------------------------------------------------------------------------
    # Test 4: Invalid telemetry cannot falsely indicate successful takeoff
    # -------------------------------------------------------------------------
    def test_invalid_telemetry_cannot_falsely_indicate_success(self):
        """NaN or negative altitudes cannot trigger false success."""
        clock = SimulatedClock(step=1.0)
        # Altitude is negative (e.g. barometer ground drift below launch point)
        self.fc.vehicle.location.global_relative_frame.alt = -0.5

        success = self.fc.arm_and_takeoff(
            target_altitude=2.0,
            timeout=3.0,
            clock_fn=clock,
            sleep_fn=dummy_sleep
        )

        self.assertFalse(success)
        self.assertEqual(self.fc.last_takeoff_status, "TIMEOUT")

    # -------------------------------------------------------------------------
    # Test 5: Pre-arm health checks reject missing or stale data
    # -------------------------------------------------------------------------
    def test_prearm_rejects_missing_or_stale_data(self):
        """Pre-arm health check must fail if heartbeat is stale or battery is critically low."""
        # 1. Stale heartbeat (> 3.0s)
        self.fc.vehicle.last_heartbeat = 5.5
        healthy, issues, missing = self.fc.check_prearm_health()
        self.assertFalse(healthy)
        self.assertTrue(any("Stale heartbeat" in s for s in issues))

        # Reset heartbeat, test low battery
        self.fc.vehicle.last_heartbeat = 0.2
        self.fc.vehicle.battery.voltage = 9.8  # Below 10.5V
        healthy, issues, missing = self.fc.check_prearm_health()
        self.assertFalse(healthy)
        self.assertTrue(any("Battery critically low" in s for s in issues))

        # Test already armed prior to sequence
        self.fc.vehicle.battery.voltage = 12.4
        self.fc.vehicle.armed = True
        healthy, issues, missing = self.fc.check_prearm_health()
        self.assertFalse(healthy)
        self.assertTrue(any("already armed" in s for s in issues))

    # -------------------------------------------------------------------------
    # Test 6: Pre-arm health checks accept valid mock data
    # -------------------------------------------------------------------------
    def test_prearm_accepts_valid_mock_data(self):
        """All sensors healthy, valid telemetry present, vehicle disarmed -> passes pre-arm."""
        self.fc.vehicle.is_armable = True
        self.fc.vehicle.armed = False
        self.fc.vehicle.battery.voltage = 12.4
        self.fc.vehicle.last_heartbeat = 0.3
        self.fc.vehicle.ekf_ok = True
        self.fc.vehicle.location.global_relative_frame.alt = 0.0

        healthy, issues, missing = self.fc.check_prearm_health()
        self.assertTrue(healthy)
        self.assertEqual(len(issues), 0)
        self.assertEqual(len(missing), 0)

    # -------------------------------------------------------------------------
    # Test 7: Exceptions/Interrupts do not leave unbounded loop
    # -------------------------------------------------------------------------
    def test_exception_in_climb_does_not_loop_infinitely(self):
        """An exception raised inside the climb loop terminates cleanly."""
        clock = SimulatedClock(step=0.5)

        def failing_sleep(seconds):
            raise KeyboardInterrupt("Simulated user Ctrl+C during takeoff climb")

        with self.assertRaises(KeyboardInterrupt):
            self.fc.arm_and_takeoff(
                target_altitude=2.0,
                timeout=10.0,
                clock_fn=clock,
                sleep_fn=failing_sleep
            )

    # -------------------------------------------------------------------------
    # Test 8: Timeout does not trigger unverified automatic LAND or disarm
    # -------------------------------------------------------------------------
    def test_timeout_does_not_issue_unverified_land_or_disarm(self):
        """On timeout, arm_and_takeoff returns False without altering vehicle mode to LAND."""
        self.fc.vehicle.location.global_relative_frame.alt = 0.5
        clock = SimulatedClock(step=1.0)

        # Vehicle is in GUIDED mode during climb
        success = self.fc.arm_and_takeoff(
            target_altitude=2.0,
            timeout=3.0,
            clock_fn=clock,
            sleep_fn=dummy_sleep
        )

        self.assertFalse(success)
        self.assertEqual(self.fc.last_takeoff_status, "TIMEOUT")
        # Ensure mode was NOT changed to LAND or RTL by arm_and_takeoff
        self.assertEqual(self.fc.vehicle.mode.name, "GUIDED")
        # Ensure no MAVLink messages were sent by takeoff timeout
        self.assertEqual(len(self.fc.vehicle.messages_sent), 0)

    # -------------------------------------------------------------------------
    # Test 9: Land bounded wait succeeds when vehicle disarms
    # -------------------------------------------------------------------------
    def test_land_completes_when_disarmed(self):
        """land() switches mode to LAND and returns True once vehicle disarms."""
        self.fc.vehicle.armed = True
        self.fc.vehicle.location.global_relative_frame.alt = 1.0
        clock = SimulatedClock(step=0.5)

        def landing_sleep(seconds):
            # Simulate descent and touchdown disarm after 2 steps
            current = self.fc.vehicle.location.global_relative_frame.alt
            if current > 0.2:
                self.fc.vehicle.location.global_relative_frame.alt = current - 0.5
            else:
                self.fc.vehicle.armed = False

        result = self.fc.land(timeout=10.0, clock_fn=clock, sleep_fn=landing_sleep)
        self.assertTrue(result, "land() must return True when vehicle successfully disarms")
        self.assertEqual(self.fc.last_landing_status, "SUCCESS")
        self.assertFalse(self.fc.vehicle.armed)
        self.assertEqual(self.fc.vehicle.mode.name, "LAND")

    # -------------------------------------------------------------------------
    # Test 10: Land times out safely if vehicle fails to disarm
    # -------------------------------------------------------------------------
    def test_land_times_out_if_vehicle_remains_armed(self):
        """land() must not loop indefinitely if vehicle fails to disarm."""
        self.fc.vehicle.armed = True
        self.fc.vehicle.location.global_relative_frame.alt = 0.5
        clock = SimulatedClock(step=1.0)

        result = self.fc.land(timeout=5.0, clock_fn=clock, sleep_fn=dummy_sleep)
        self.assertFalse(result, "land() must return False on timeout")
        self.assertEqual(self.fc.last_landing_status, "TIMEOUT")
        self.assertTrue(self.fc.vehicle.armed)
        self.assertEqual(self.fc.vehicle.mode.name, "LAND")

    # -------------------------------------------------------------------------
    # Test 11: Land handles missing altitude telemetry safely
    # -------------------------------------------------------------------------
    def test_land_handles_none_altitude_safely(self):
        """land() handles None relative altitude telemetry without crashing."""
        self.fc.vehicle.armed = True
        self.fc.vehicle.location.global_relative_frame = None
        clock = SimulatedClock(step=1.0)

        def disarm_immediately(seconds):
            self.fc.vehicle.armed = False

        result = self.fc.land(timeout=5.0, clock_fn=clock, sleep_fn=disarm_immediately)
        self.assertTrue(result)
        self.assertEqual(self.fc.last_landing_status, "SUCCESS")
        self.assertFalse(self.fc.vehicle.armed)


    # -------------------------------------------------------------------------
    # Test 12: Land returns False when vehicle is disconnected
    # -------------------------------------------------------------------------
    def test_land_returns_false_when_not_connected(self):
        """land() immediately returns False if vehicle is not connected."""
        self.fc._connected = False
        self.fc.vehicle = None
        result = self.fc.land()
        self.assertFalse(result)
        self.assertEqual(self.fc.last_landing_status, "NOT_CONNECTED")

    # -------------------------------------------------------------------------
    # Test 13: Close warns when vehicle is still armed
    # -------------------------------------------------------------------------
    def test_close_warns_when_vehicle_armed(self):
        """close() warns if vehicle is armed when telemetry connection is closed."""
        import io
        import sys
        self.fc.vehicle.armed = True
        captured = io.StringIO()
        orig_stdout = sys.stdout
        try:
            sys.stdout = captured
            self.fc.close()
        finally:
            sys.stdout = orig_stdout

        self.assertIn("WARNING: Closing telemetry connection while vehicle is still armed!", captured.getvalue())
        self.assertFalse(self.fc._connected)
        self.assertIsNone(self.fc.vehicle)

    # -------------------------------------------------------------------------
    # Test 14: Takeoff returns False when vehicle is not connected
    # -------------------------------------------------------------------------
    def test_takeoff_returns_false_when_not_connected(self):
        """arm_and_takeoff() returns False immediately if not connected."""
        self.fc._connected = False
        self.fc.vehicle = None
        result = self.fc.arm_and_takeoff()
        self.assertFalse(result)
        self.assertEqual(self.fc.last_takeoff_status, "NOT_CONNECTED")

    # -------------------------------------------------------------------------
    # Test 15: Takeoff records MODE_CHANGE_FAILED on GUIDED mode failure
    # -------------------------------------------------------------------------
    def test_takeoff_mode_change_failure_recorded(self):
        """arm_and_takeoff() records MODE_CHANGE_FAILED when mode switch fails."""
        self.fc.vehicle.allow_mode_change = False
        clock = SimulatedClock(step=2.0)
        result = self.fc.arm_and_takeoff(clock_fn=clock, sleep_fn=dummy_sleep)
        self.assertFalse(result)
        self.assertEqual(self.fc.last_takeoff_status, "MODE_CHANGE_FAILED")

    # -------------------------------------------------------------------------
    # Test 16: Takeoff records ARMING_FAILED on arming failure
    # -------------------------------------------------------------------------
    def test_takeoff_arming_failure_recorded(self):
        """arm_and_takeoff() records ARMING_FAILED when arming fails."""
        self.fc.vehicle.allow_arming = False
        clock = SimulatedClock(step=2.0)
        result = self.fc.arm_and_takeoff(clock_fn=clock, sleep_fn=dummy_sleep)
        self.assertFalse(result)
        self.assertEqual(self.fc.last_takeoff_status, "ARMING_FAILED")


if __name__ == '__main__':
    unittest.main()
