"""
test_config.py — Cross-Platform Configuration Decoupling Tests

Verifies that config.py imports without hardware side-effects, supports
environment variable overrides for cross-platform deployments (Windows dev vs. Linux Pi),
and maintains valid physical constants.
"""

import os
import sys
import unittest
import importlib
import config


class TestConfig(unittest.TestCase):
    """Unit tests for configuration decoupling and environment overrides."""

    def test_clean_import_without_hardware_side_effects(self):
        """Importing config must never open serial ports, instantiate camera, or execute I/O."""
        # config is imported at module top level without throwing exceptions or opening serial
        self.assertIsNotNone(config.SERIAL_PORT)
        self.assertIsInstance(config.SERIAL_PORT, str)
        self.assertGreater(config.BAUD_RATE, 0)

    def test_environment_variable_overrides_connection_string(self):
        """DRONE_CONNECTION_STRING environment variable overrides default SERIAL_PORT."""
        test_conn = "udpin:127.0.0.1:14550"
        old_env = os.environ.get('DRONE_CONNECTION_STRING')
        try:
            os.environ['DRONE_CONNECTION_STRING'] = test_conn
            # Reload module to verify environment pickup
            reloaded = importlib.reload(config)
            self.assertEqual(reloaded.SERIAL_PORT, test_conn)
        finally:
            if old_env is not None:
                os.environ['DRONE_CONNECTION_STRING'] = old_env
            else:
                os.environ.pop('DRONE_CONNECTION_STRING', None)
            importlib.reload(config)

    def test_environment_variable_overrides_camera_source(self):
        """CAMERA_SOURCE environment variable overrides camera source mode."""
        old_cam = os.environ.get('CAMERA_SOURCE')
        try:
            os.environ['CAMERA_SOURCE'] = 'opencv:0'
            reloaded = importlib.reload(config)
            self.assertEqual(reloaded.CAMERA_SOURCE, 'opencv:0')
        finally:
            if old_cam is not None:
                os.environ['CAMERA_SOURCE'] = old_cam
            else:
                os.environ.pop('CAMERA_SOURCE', None)
            importlib.reload(config)

    def test_battery_and_flight_constants_consistency(self):
        """Verify logical consistency of safety thresholds and timeout bounds."""
        self.assertGreater(config.TARGET_ALTITUDE, 0.0)
        self.assertGreater(config.FORWARD_SPEED, 0.0)
        self.assertGreater(config.MAX_LATERAL_SPEED, 0.0)
        self.assertGreater(config.TAKEOFF_TIMEOUT, 0.0)
        self.assertGreater(config.LANDING_TIMEOUT, 0.0)
        self.assertGreater(config.TAKEOFF_POLL_INTERVAL, 0.0)

        # Battery bounds sanity
        self.assertLess(config.BATTERY_MIN_VALID_VOLTAGE, config.LOW_BATTERY_VOLTAGE)
        self.assertGreater(config.BATTERY_MAX_VALID_VOLTAGE, config.LOW_BATTERY_VOLTAGE)
        self.assertGreater(config.BATTERY_LOW_CONSECUTIVE_COUNT, 0)
        self.assertGreater(config.BATTERY_STALE_TIMEOUT, 0.0)
        self.assertGreater(config.BATTERY_HYSTERESIS_VOLTAGE, 0.0)

        # Camera timeout hierarchy: frame timeout must be smaller than fault timeout
        self.assertLess(config.CAMERA_FRAME_TIMEOUT, config.CAMERA_FAULT_TIMEOUT)
        self.assertGreater(config.CAMERA_MAX_CONSECUTIVE_FAILURES, 0)
        self.assertGreater(config.CAMERA_RETRY_SLEEP, 0.0)

        # Vision parameters
        self.assertGreater(config.THRESHOLD_VALUE, 0)
        self.assertLess(config.THRESHOLD_VALUE, 255)
        self.assertEqual(config.BLUR_KERNEL_SIZE % 2, 1, "Blur kernel size must be odd")
        self.assertGreater(config.MIN_CONTOUR_AREA, 0)


if __name__ == '__main__':
    unittest.main()
