"""
test_pi_main.py — Integration and regression tests for PiDroneApplication.
"""

import unittest
from raspberry_pi.pi_main import PiDroneApplication
from raspberry_pi.camera_source import SyntheticCameraSource
from raspberry_pi.mavlink_adapter import MockMAVLinkAdapter
from raspberry_pi.config_loader import ConfigValidationError


class TestPiMain(unittest.TestCase):
    """Test suite for top-level application dispatching, mock flight loop, and hardware gating."""

    def test_diagnostics_mode_executes_zero_flight_commands(self):
        """Diagnostics mode executes sensor checks and transmits ZERO flight commands."""
        cam = SyntheticCameraSource(width=320, height=240)
        mavlink = MockMAVLinkAdapter(initial_mode="STABILIZE", initial_voltage=12.4)

        app = PiDroneApplication(
            mode_override="diagnostics",
            camera_source=cam,
            mavlink_adapter=mavlink
        )
        exit_code = app.run()

        # Exit code 0 (both camera and telemetry succeeded)
        self.assertEqual(exit_code, 0)

        # STRICT VERIFICATION: Zero flight commands were transmitted
        cmd_names = [cmd[0] for cmd in mavlink.commands_sent]
        self.assertNotIn("arm_and_takeoff", cmd_names)
        self.assertNotIn("velocity", cmd_names)
        self.assertNotIn("land", cmd_names)
        self.assertFalse(mavlink.get_telemetry().armed)

    def test_mock_mode_full_flight_loop(self):
        """Mock mode performs takeoff, velocity tracking, and landing across bounded cycles."""
        cam = SyntheticCameraSource(width=320, height=240)
        cam.set_pattern("offset", offset_px=20)
        mavlink = MockMAVLinkAdapter(initial_mode="STABILIZE", initial_voltage=12.4)

        app = PiDroneApplication(
            mode_override="mock",
            max_cycles=5,
            camera_source=cam,
            mavlink_adapter=mavlink
        )
        exit_code = app.run()

        self.assertEqual(exit_code, 0)

        cmd_names = [cmd[0] for cmd in mavlink.commands_sent]
        self.assertIn("arm_and_takeoff", cmd_names)
        self.assertIn("velocity", cmd_names)
        self.assertIn("land", cmd_names)
        self.assertIn("close", cmd_names)

    def test_hardware_mode_rejected_when_unconfigured(self):
        """Application refuses initialization in hardware mode when required settings are missing."""
        with self.assertRaises(ConfigValidationError):
            PiDroneApplication(mode_override="hardware")

    def test_takeoff_failure_aborts_cleanly(self):
        """When takeoff fails, application terminates without entering loop and closes connection."""
        cam = SyntheticCameraSource(width=320, height=240)
        mavlink = MockMAVLinkAdapter()
        mavlink.allow_takeoff_success = False  # Simulate takeoff failure

        app = PiDroneApplication(
            mode_override="mock",
            max_cycles=5,
            camera_source=cam,
            mavlink_adapter=mavlink
        )
        exit_code = app.run()

        self.assertEqual(exit_code, 1)
        cmd_names = [cmd[0] for cmd in mavlink.commands_sent]
        self.assertIn("arm_and_takeoff", cmd_names)
        self.assertNotIn("velocity", cmd_names)  # Never entered control loop
        self.assertIn("close", cmd_names)


if __name__ == '__main__':
    unittest.main()
