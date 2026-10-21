"""
test_mavlink_adapter.py — Unit tests for MAVLinkAdapterBase, MockMAVLinkAdapter, and Hardware guards.
"""

import unittest
from raspberry_pi.mavlink_adapter import (
    MockMAVLinkAdapter,
    HardwareMAVLinkAdapter,
    MAVLinkTelemetry
)


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


class TestMAVLinkAdapter(unittest.TestCase):
    """Test suite for MAVLink telemetry queries, mock command recording, and hardware guards."""

    def test_mock_adapter_telemetry_snapshot(self):
        """Mock adapter accurately reflects simulated telemetry snapshot."""
        adapter = MockMAVLinkAdapter(initial_mode="STABILIZE", initial_voltage=12.5)
        telem = adapter.get_telemetry()
        self.assertTrue(adapter.is_connected())
        self.assertEqual(telem.mode, "STABILIZE")
        self.assertEqual(telem.voltage, 12.5)
        self.assertFalse(telem.armed)
        self.assertTrue(adapter.is_heartbeat_fresh(3.0))

    def test_mock_adapter_stale_heartbeat_detection(self):
        """Simulated heartbeat age beyond threshold flags heartbeat as not fresh."""
        clock = SimulatedClock()
        adapter = MockMAVLinkAdapter(clock_fn=clock)
        self.assertTrue(adapter.is_heartbeat_fresh(3.0))

        # Force stale heartbeat (e.g. 5 seconds ago)
        adapter.simulate_stale_heartbeat(age_sec=5.0)
        self.assertFalse(adapter.is_heartbeat_fresh(3.0))

        # Simulate fresh heartbeat arrives
        adapter.simulate_heartbeat()
        self.assertTrue(adapter.is_heartbeat_fresh(3.0))

    def test_mock_adapter_rc_override_detection(self):
        """RC channel PWM exceeding threshold triggers override detection."""
        adapter = MockMAVLinkAdapter()
        self.assertFalse(adapter.check_rc_override(channel=5, threshold_pwm=1800))

        adapter.simulate_rc_channel(channel=5, pwm=1950)
        self.assertTrue(adapter.check_rc_override(channel=5, threshold_pwm=1800))

        adapter.simulate_rc_channel(channel=5, pwm=1500)
        self.assertFalse(adapter.check_rc_override(channel=5, threshold_pwm=1800))

    def test_mock_adapter_command_recording(self):
        """Takeoff, velocity, mode, and landing commands are faithfully recorded in order."""
        adapter = MockMAVLinkAdapter()

        # Arm and takeoff
        success_to = adapter.arm_and_takeoff(target_altitude=2.0, timeout=15.0)
        self.assertTrue(success_to)
        self.assertTrue(adapter.get_telemetry().armed)
        self.assertEqual(adapter.get_telemetry().mode, "GUIDED")

        # Send velocity
        adapter.send_velocity(vx=0.3, vy=0.1, vz=0.0)

        # Land
        success_land = adapter.land(timeout=30.0)
        self.assertTrue(success_land)
        self.assertFalse(adapter.get_telemetry().armed)
        self.assertEqual(adapter.get_telemetry().mode, "LAND")

        # Verify command history
        actions = [cmd[0] for cmd in adapter.commands_sent]
        self.assertEqual(actions, ["arm_and_takeoff", "velocity", "land"])

    def test_mock_adapter_close_disconnects(self):
        """close() sets connected to False and records action."""
        adapter = MockMAVLinkAdapter()
        self.assertTrue(adapter.is_connected())
        adapter.close()
        self.assertFalse(adapter.is_connected())
        self.assertFalse(adapter.is_heartbeat_fresh())

    def test_hardware_adapter_blocks_offline_execution(self):
        """HardwareMAVLinkAdapter raises PermissionError if allow_hardware is False."""
        with self.assertRaises(PermissionError):
            HardwareMAVLinkAdapter(port="/dev/ttyAMA0", allow_hardware=False)

    def test_hardware_adapter_rejects_empty_port(self):
        """HardwareMAVLinkAdapter raises ValueError if port is None or empty."""
        with self.assertRaises(ValueError):
            HardwareMAVLinkAdapter(port=None, allow_hardware=True)
        with self.assertRaises(ValueError):
            HardwareMAVLinkAdapter(port="UNCONFIGURED", allow_hardware=True)


if __name__ == '__main__':
    unittest.main()
