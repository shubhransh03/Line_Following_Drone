"""
test_config_validation.py — Unit tests for YAML configuration loading and safety validation.
"""

import unittest
from raspberry_pi.config_loader import (
    parse_simple_yaml,
    load_config,
    validate_config,
    ConfigValidationError
)


class TestConfigValidation(unittest.TestCase):
    """Test suite for configuration parsing and hardware-mode safety gates."""

    def test_parse_simple_yaml_scalars(self):
        """Verify lightweight YAML parser parses ints, floats, booleans, None, and strings."""
        yaml_text = """
        mode: "diagnostics"
        baud_rate: 57600
        kp: 0.35
        invert: true
        active: false
        port: null
        comment_test: 10 # this is a comment
        """
        parsed = parse_simple_yaml(yaml_text)
        self.assertEqual(parsed["mode"], "diagnostics")
        self.assertEqual(parsed["baud_rate"], 57600)
        self.assertAlmostEqual(parsed["kp"], 0.35)
        self.assertTrue(parsed["invert"])
        self.assertFalse(parsed["active"])
        self.assertIsNone(parsed["port"])
        self.assertEqual(parsed["comment_test"], 10)

    def test_parse_simple_yaml_nested(self):
        """Verify lightweight YAML parser parses nested dictionaries."""
        yaml_text = """
        telemetry:
          port: "/dev/ttyAMA0"
          baud_rate: 57600
        camera:
          width: 640
          height: 480
        """
        parsed = parse_simple_yaml(yaml_text)
        self.assertIn("telemetry", parsed)
        self.assertEqual(parsed["telemetry"]["port"], "/dev/ttyAMA0")
        self.assertEqual(parsed["telemetry"]["baud_rate"], 57600)
        self.assertEqual(parsed["camera"]["width"], 640)
        self.assertEqual(parsed["camera"]["height"], 480)

    def test_load_default_config_file(self):
        """Verify the repository config.yaml loads cleanly."""
        cfg = load_config()
        self.assertIn("mode", cfg)
        self.assertIn("telemetry", cfg)
        self.assertIn("camera", cfg)
        self.assertIn("vision", cfg)
        self.assertIn("guidance", cfg)
        self.assertIn("safety", cfg)

    def test_diagnostics_mode_allows_unconfigured_hardware(self):
        """Diagnostics mode allows unconfigured serial port and returns warnings."""
        cfg = load_config()
        warnings = validate_config(cfg, requested_mode="diagnostics")
        self.assertTrue(any("telemetry.port" in w for w in warnings))

    def test_mock_mode_allows_unconfigured_hardware(self):
        """Mock simulation mode allows unconfigured serial port."""
        cfg = load_config()
        warnings = validate_config(cfg, requested_mode="mock")
        self.assertTrue(any("telemetry.port" in w for w in warnings))

    def test_hardware_mode_strictly_rejected_when_unconfigured(self):
        """Hardware mode must raise ConfigValidationError if port, camera, or battery are unset."""
        cfg = load_config()
        with self.assertRaises(ConfigValidationError) as ctx:
            validate_config(cfg, requested_mode="hardware")
        err_msg = str(ctx.exception)
        self.assertIn("telemetry.port is unconfigured", err_msg)
        self.assertIn("safety.battery.cell_count is unconfigured", err_msg)

    def test_hardware_mode_accepts_valid_hardware_config(self):
        """Hardware mode passes validation when all physical settings are specified."""
        cfg = load_config()
        # Supply valid hardware settings
        cfg["telemetry"]["port"] = "/dev/ttyAMA0"
        cfg["telemetry"]["baud_rate"] = 57600
        cfg["camera"]["device_index"] = 0
        cfg["camera"]["source_type"] = "usb"
        cfg["safety"]["battery"]["cell_count"] = 3
        cfg["safety"]["battery"]["low_voltage"] = 10.5

        warnings = validate_config(cfg, requested_mode="hardware")
        self.assertIsInstance(warnings, list)

    def test_invalid_vision_and_guidance_parameters(self):
        """Invalid vision or guidance parameters raise ConfigValidationError."""
        cfg = load_config()
        cfg["vision"]["blur_kernel_size"] = 4  # Must be odd
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

        cfg["vision"]["blur_kernel_size"] = 5
        cfg["guidance"]["max_lateral_speed"] = -0.5  # Must be positive
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
