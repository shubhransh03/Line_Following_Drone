"""
test_battery_monitor.py — Offline Unit Tests for BatteryMonitor

Deterministic tests covering state machine transitions, noise filtering,
invalid/stale telemetry rejection, and hysteresis recovery.
"""

import unittest
import math
from battery_monitor import BatteryMonitor, BatteryState


class SimulatedClock:
    """Monotonic clock simulator advancing deterministically."""
    def __init__(self, start_time=100.0):
        self.time = start_time

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


class TestBatteryMonitor(unittest.TestCase):
    """Unit tests for BatteryMonitor."""

    def setUp(self):
        self.clock = SimulatedClock(start_time=100.0)
        self.monitor = BatteryMonitor(
            low_threshold=10.5,
            consecutive_low_required=3,
            stale_timeout=3.0,
            min_valid_voltage=5.0,
            max_valid_voltage=26.0,
            hysteresis=0.3,
            recovery_count=3,
            window_size=5,
            clock_fn=self.clock
        )

    # -------------------------------------------------------------------------
    # Initial State
    # -------------------------------------------------------------------------
    def test_initial_state_is_unknown(self):
        """Monitor starts in UNKNOWN state before any readings arrive."""
        self.assertEqual(self.monitor.state, BatteryState.UNKNOWN)
        self.assertIsNone(self.monitor.last_voltage)
        self.assertIsNone(self.monitor.rolling_average)
        self.assertFalse(self.monitor.is_ok(allow_unknown=False))
        self.assertTrue(self.monitor.is_ok(allow_unknown=True))

    # -------------------------------------------------------------------------
    # Normal Operation
    # -------------------------------------------------------------------------
    def test_valid_voltage_transitions_to_normal(self):
        """A valid reading above threshold immediately transitions to NORMAL."""
        state = self.monitor.update(12.4)
        self.assertEqual(state, BatteryState.NORMAL)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)
        self.assertAlmostEqual(self.monitor.last_voltage, 12.4)
        self.assertTrue(self.monitor.is_ok())

    # -------------------------------------------------------------------------
    # Noise & Transient Dip Filtering
    # -------------------------------------------------------------------------
    def test_transient_dip_does_not_immediately_trigger_low(self):
        """
        A single momentary voltage sag (e.g. rapid motor acceleration)
        must NOT trigger LOW state when consecutive_low_required = 3.
        """
        # Baseline normal
        self.monitor.update(12.0)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)

        # Sag 1: 10.2V (below 10.5V)
        self.clock.advance(0.1)
        self.monitor.update(10.2)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL,
                         "Single dip must not transition to LOW")
        self.assertEqual(self.monitor.consecutive_low_count, 1)

        # Sag 2: 10.3V
        self.clock.advance(0.1)
        self.monitor.update(10.3)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL,
                         "Two dips must not transition to LOW when 3 required")
        self.assertEqual(self.monitor.consecutive_low_count, 2)

        # Recovery before 3rd sample: back to 11.8V
        self.clock.advance(0.1)
        self.monitor.update(11.8)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)
        self.assertEqual(self.monitor.consecutive_low_count, 0,
                         "Recovery resets low counter")

    # -------------------------------------------------------------------------
    # Sustained Low Voltage
    # -------------------------------------------------------------------------
    def test_sustained_low_triggers_low_state(self):
        """Sustained low readings matching consecutive_low_required triggers LOW."""
        self.monitor.update(12.0)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)

        # 3 consecutive readings below 10.5V
        self.clock.advance(0.1)
        self.monitor.update(10.4)
        self.clock.advance(0.1)
        self.monitor.update(10.3)
        self.clock.advance(0.1)
        state = self.monitor.update(10.2)

        self.assertEqual(state, BatteryState.LOW)
        self.assertEqual(self.monitor.state, BatteryState.LOW)
        self.assertFalse(self.monitor.is_ok())

    # -------------------------------------------------------------------------
    # Hysteresis Recovery
    # -------------------------------------------------------------------------
    def test_hysteresis_recovery_from_low_state(self):
        """
        Once LOW, recovering to NORMAL requires voltage >= (low_threshold + hysteresis)
        for recovery_count consecutive samples.
        """
        # Trigger LOW: threshold 10.5V, hysteresis 0.3V -> recovery requires >= 10.8V
        for v in [10.2, 10.1, 10.0]:
            self.clock.advance(0.1)
            self.monitor.update(v)
        self.assertEqual(self.monitor.state, BatteryState.LOW)

        # Reading 10.6V is above 10.5V but below 10.8V (within hysteresis band)
        self.clock.advance(0.1)
        self.monitor.update(10.6)
        self.assertEqual(self.monitor.state, BatteryState.LOW,
                         "Voltage inside hysteresis margin must not clear LOW state")

        # Readings >= 10.8V: need 3 consecutive
        self.clock.advance(0.1)
        self.monitor.update(11.0)
        self.assertEqual(self.monitor.state, BatteryState.LOW)

        self.clock.advance(0.1)
        self.monitor.update(11.2)
        self.assertEqual(self.monitor.state, BatteryState.LOW)

        self.clock.advance(0.1)
        self.monitor.update(11.1)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL,
                         "3 sustained readings above recovery threshold clear LOW state")

    # -------------------------------------------------------------------------
    # Invalid Reading Rejection
    # -------------------------------------------------------------------------
    def test_rejection_of_invalid_inputs(self):
        """Rejects None, strings, NaN, Inf, negative, zero, and extreme spike readings."""
        self.monitor.update(12.0)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)
        initial_rejected = self.monitor.rejected_readings_count

        invalid_samples = [
            None,
            "12.4V",
            float('nan'),
            float('inf'),
            float('-inf'),
            -1.0,
            0.0,
            3.2,    # below 5.0V min_valid_voltage
            35.0,   # above 26.0V max_valid_voltage
        ]

        for sample in invalid_samples:
            self.clock.advance(0.1)
            self.monitor.update(sample)
            # Last valid voltage should remain unchanged
            self.assertAlmostEqual(self.monitor.last_voltage, 12.0)

        self.assertEqual(self.monitor.rejected_readings_count,
                         initial_rejected + len(invalid_samples))
        # State remains NORMAL because staleness timeout (3.0s) has not elapsed
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)

    # -------------------------------------------------------------------------
    # Stale Telemetry Detection
    # -------------------------------------------------------------------------
    def test_stale_telemetry_detection(self):
        """Advancing clock beyond stale_timeout without valid updates transitions to STALE."""
        self.monitor.update(12.0)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)

        # Advance 2.5s (< 3.0s timeout)
        self.clock.advance(2.5)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)
        self.assertTrue(self.monitor.is_ok())

        # Advance another 1.0s (total 3.5s > 3.0s)
        self.clock.advance(1.0)
        self.assertEqual(self.monitor.state, BatteryState.STALE)
        self.assertFalse(self.monitor.is_ok())

        # Fresh valid reading arrives -> recovers immediately to NORMAL
        self.monitor.update(12.2)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)
        self.assertTrue(self.monitor.is_ok())

    # -------------------------------------------------------------------------
    # Rolling Average
    # -------------------------------------------------------------------------
    def test_rolling_average_calculation(self):
        """Rolling average tracks the mean of the most recent valid window."""
        readings = [12.0, 12.2, 11.8, 12.0, 12.0]
        for r in readings:
            self.clock.advance(0.1)
            self.monitor.update(r)

        expected = sum(readings) / len(readings)
        self.assertAlmostEqual(self.monitor.rolling_average, expected, places=3)

        # Push one more reading (pushes out the oldest 12.0)
        self.clock.advance(0.1)
        self.monitor.update(12.4)
        new_expected = (12.2 + 11.8 + 12.0 + 12.0 + 12.4) / 5
        self.assertAlmostEqual(self.monitor.rolling_average, new_expected, places=3)

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    def test_reset_functionality(self):
        """Reset clears state, counters, history, and timestamps back to UNKNOWN."""
        self.monitor.update(12.0)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)

        self.monitor.reset()
        self.assertEqual(self.monitor.state, BatteryState.UNKNOWN)
        self.assertIsNone(self.monitor.last_voltage)
        self.assertIsNone(self.monitor.last_valid_time)
        self.assertIsNone(self.monitor.rolling_average)
        self.assertEqual(self.monitor.consecutive_low_count, 0)
        self.assertEqual(self.monitor.rejected_readings_count, 0)


    # -------------------------------------------------------------------------
    # Interleaved Invalid Readings
    # -------------------------------------------------------------------------
    def test_interleaved_invalid_readings_behavior(self):
        """Invalid readings do not count towards low count, nor do they clear it."""
        self.monitor.update(12.0)
        self.assertEqual(self.monitor.state, BatteryState.NORMAL)

        # 1 valid low
        self.clock.advance(0.1)
        self.monitor.update(10.2)
        self.assertEqual(self.monitor.consecutive_low_count, 1)

        # 1 invalid reading (NaN)
        self.clock.advance(0.1)
        self.monitor.update(float('nan'))
        self.assertEqual(self.monitor.consecutive_low_count, 1,
                         "Invalid reading must not increment low count")

        # 2nd valid low
        self.clock.advance(0.1)
        self.monitor.update(10.1)
        self.assertEqual(self.monitor.consecutive_low_count, 2)

        # 1 valid normal reading clears low counter
        self.clock.advance(0.1)
        self.monitor.update(11.5)
        self.assertEqual(self.monitor.consecutive_low_count, 0,
                         "Valid normal reading must reset low counter to 0")

    # -------------------------------------------------------------------------
    # FC Integration: Heartbeat Freshness in check_battery
    # -------------------------------------------------------------------------
    def test_fc_check_battery_rejects_stale_heartbeat_despite_cached_voltage(self):
        """
        FlightController.check_battery must return False if heartbeat is stale,
        preventing stale cached DroneKit voltage from falsely reporting healthy battery.
        """
        from flight_controller import FlightController

        class MockBatteryData:
            def __init__(self, v):
                self.voltage = v

        class MockVehicleWithHB:
            def __init__(self):
                self.battery = MockBatteryData(12.4)  # Cached healthy voltage
                self.last_heartbeat = 0.5            # Fresh initially

        fc = FlightController()
        fc.vehicle = MockVehicleWithHB()
        fc._connected = True

        # Initially fresh -> returns True
        self.assertTrue(fc.check_battery())

        # Heartbeat stops arriving (5.0s > 3.0s timeout) while voltage remains 12.4 in memory
        fc.vehicle.last_heartbeat = 5.0
        self.assertFalse(fc.check_battery(),
                         "check_battery must return False when telemetry heartbeat is stale")


if __name__ == '__main__':
    unittest.main()
