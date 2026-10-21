"""
test_safety_monitor.py — Unit tests for multi-subsystem SafetyMonitor evaluation.
"""

import unittest
from raspberry_pi.safety_monitor import SafetyMonitor, SafetyAction, SafetyDecision
from raspberry_pi.mavlink_adapter import MockMAVLinkAdapter
from battery_monitor import BatteryMonitor


class SimulatedClock:
    def __init__(self, start=100.0, step=0.1):
        self.time = start
        self.step = step

    def __call__(self):
        t = self.time
        self.time += self.step
        return t

    def advance(self, dt):
        self.time += dt


class TestSafetyMonitor(unittest.TestCase):
    """Test suite for flight safety evaluation, watchdog bounds, and failsafe triggers."""

    def setUp(self):
        self.clock = SimulatedClock(start=100.0)
        self.mavlink = MockMAVLinkAdapter(initial_mode="GUIDED", initial_voltage=12.4, clock_fn=self.clock)
        self.safety = SafetyMonitor(
            camera_frame_timeout=1.0,
            camera_fault_timeout=3.0,
            telemetry_timeout=3.0,
            line_lost_timeout=3.0,
            rc_override_channel=5,
            rc_override_threshold=1800,
            rc_override_enabled=True,
            clock_fn=self.clock
        )
        self.safety.start_monitoring()

    def test_nominal_flight_conditions(self):
        """All systems healthy produces CONTINUE action."""
        dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.CONTINUE)
        self.assertTrue(dec.safe_to_proceed)
        self.assertIsNone(dec.fault_code)

    def test_camera_frame_timeout_triggers_hover(self):
        """Camera drop between 1.0s and 3.0s mandates zero-velocity HOVER."""
        # Advance clock by 1.5s without frame success
        self.clock.advance(1.5)
        dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.HOVER)
        self.assertTrue(dec.safe_to_proceed)
        self.assertEqual(dec.fault_code, "CAMERA_DROP_HOVER")

    def test_camera_fault_timeout_triggers_abort(self):
        """Persistent camera feed loss exceeding 3.0s mandates autonomous ABORT landing."""
        self.clock.advance(3.5)
        self.mavlink.simulate_heartbeat()  # Keep telemetry link fresh to isolate camera fault
        dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.ABORT)
        self.assertFalse(dec.safe_to_proceed)
        self.assertEqual(dec.fault_code, "CAMERA_FAULT")

    def test_camera_recovery_clears_hover(self):
        """A valid frame arrival recovers camera drop and returns action to CONTINUE."""
        self.clock.advance(1.5)
        self.assertEqual(self.safety.evaluate(self.mavlink).action, SafetyAction.HOVER)

        # Frame arrives
        self.safety.record_frame_success()
        dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.CONTINUE)

    def test_stale_telemetry_triggers_abort(self):
        """Heartbeat age exceeding 3.0s mandates autonomous ABORT landing."""
        self.mavlink.simulate_stale_heartbeat(age_sec=4.0)
        dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.ABORT)
        self.assertFalse(dec.safe_to_proceed)
        self.assertEqual(dec.fault_code, "TELEMETRY_STALE")

    def test_low_battery_triggers_abort(self):
        """Confirmed low battery voltage mandates ABORT landing."""
        # Sustained low voltage readings (3 consecutive samples required)
        for _ in range(3):
            self.mavlink.simulate_voltage(10.2)
            dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.ABORT)
        self.assertEqual(dec.fault_code, "LOW_BATTERY")

    def test_pilot_rc_override_surrenders_control(self):
        """Pilot RC switch mandates RELEASE_CONTROL with highest priority."""
        self.mavlink.simulate_rc_channel(channel=5, pwm=1950)
        dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.RELEASE_CONTROL)
        self.assertFalse(dec.safe_to_proceed)
        self.assertEqual(dec.fault_code, "PILOT_OVERRIDE")

    def test_mode_divergence_surrenders_control(self):
        """Autopilot mode changing from GUIDED (e.g. to STABILIZE) surrenders control."""
        self.mavlink.telemetry.mode = "STABILIZE"
        dec = self.safety.evaluate(self.mavlink, line_detected=True)
        self.assertEqual(dec.action, SafetyAction.RELEASE_CONTROL)
        self.assertEqual(dec.fault_code, "MODE_DIVERGED")

    def test_line_loss_watchdog(self):
        """Line lost < 3.0s triggers search HOVER; > 3.0s triggers ABORT."""
        # Transient loss (1.0s): camera frames still arriving, but line not found
        self.clock.advance(1.0)
        self.mavlink.simulate_heartbeat()
        self.safety.record_frame_success()
        dec_transient = self.safety.evaluate(self.mavlink, line_detected=False)
        self.assertEqual(dec_transient.action, SafetyAction.HOVER)

        # Sustained loss (> 3.0s): camera frames still arriving, line continuously lost
        self.clock.advance(2.5)
        self.mavlink.simulate_heartbeat()
        self.safety.record_frame_success()
        dec_abort = self.safety.evaluate(self.mavlink, line_detected=False)
        self.assertEqual(dec_abort.action, SafetyAction.ABORT)
        self.assertEqual(dec_abort.fault_code, "LINE_LOST_TIMEOUT")


if __name__ == '__main__':
    unittest.main()
