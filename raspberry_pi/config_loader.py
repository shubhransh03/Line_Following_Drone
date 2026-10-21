"""
config_loader.py — YAML Configuration Loader and Validator for Raspberry Pi Deployment.

Supports:
  - Loading standard YAML files via PyYAML if installed.
  - Fallback pure-Python lightweight YAML parser when PyYAML is not installed (e.g. offline dev).
  - Explicit schema and constraint validation.
  - Strict hardware-mode gating: prevents starting in hardware mode when required
    physical settings (port, camera, battery cells) remain unconfigured.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple


class ConfigValidationError(ValueError):
    """Raised when configuration values fail validation or violate safety rules."""
    pass


def _parse_yaml_scalar(val_str: str) -> Any:
    """Parse a scalar string into a Python type (int, float, bool, None, str)."""
    val_str = val_str.strip()
    if not val_str or val_str.lower() in ("null", "~", "none"):
        return None
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False
    # Check if quoted string
    if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    # Try integer
    try:
        return int(val_str)
    except ValueError:
        pass
    # Try float
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Lightweight, robust indentation-based YAML parser for dictionaries, nested dicts, and scalars.
    Used as an offline zero-dependency fallback when PyYAML is not installed.
    """
    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in lines:
        # Strip trailing comments while respecting quotes
        line = raw_line
        in_quote = False
        quote_char = ""
        comment_idx = -1
        for idx, char in enumerate(line):
            if char in ('"', "'"):
                if not in_quote:
                    in_quote = True
                    quote_char = char
                elif quote_char == char:
                    in_quote = False
            elif char == '#' and not in_quote:
                comment_idx = idx
                break
        if comment_idx != -1:
            line = line[:comment_idx]

        stripped = line.rstrip()
        if not stripped.strip():
            continue

        indent = len(stripped) - len(stripped.lstrip(' '))
        content = stripped.strip()

        # Key: value pair or key: (nested dict)
        if ":" not in content:
            continue

        parts = content.split(":", 1)
        key = parts[0].strip()
        # Remove any quotes from key
        if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
            key = key[1:-1]
        val_part = parts[1].strip()

        # Adjust stack based on indentation
        while stack and stack[-1][0] >= indent:
            stack.pop()

        current_dict = stack[-1][1]

        if not val_part:
            # Nested dictionary starting
            new_dict: Dict[str, Any] = {}
            current_dict[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current_dict[key] = _parse_yaml_scalar(val_part)

    return root


def load_config(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        file_path: Path to config YAML file. If None, uses default 'config.yaml'
                   in the same directory as this file.

    Returns:
        Nested dictionary of configuration values.
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try PyYAML if available
    try:
        import yaml
        config = yaml.safe_load(content)
        if not isinstance(config, dict):
            raise ConfigValidationError("Configuration file must define a YAML mapping/dictionary.")
        return config
    except ImportError:
        # Fallback to pure-Python YAML parser
        return parse_simple_yaml(content)


def validate_config(cfg: Dict[str, Any], requested_mode: Optional[str] = None) -> List[str]:
    """
    Validate the configuration dictionary against structural and safety rules.

    Args:
        cfg: The parsed configuration dictionary.
        requested_mode: Optional mode override (e.g. from CLI argument).

    Returns:
        List of warning messages (for non-fatal missing settings in safe modes).

    Raises:
        ConfigValidationError: If required settings are invalid or if hardware mode
                               is attempted with unconfigured hardware parameters.
    """
    if not isinstance(cfg, dict):
        raise ConfigValidationError("Configuration root must be a dictionary.")

    warnings: List[str] = []

    # 1. Operating Mode
    mode = requested_mode or cfg.get("mode", "diagnostics")
    valid_modes = ("diagnostics", "mock", "simulation", "hardware")
    if mode not in valid_modes:
        raise ConfigValidationError(
            f"Invalid operating mode '{mode}'. Must be one of: {valid_modes}"
        )

    # 2. Check top-level sections
    required_sections = ["telemetry", "camera", "vision", "guidance", "safety"]
    for sec in required_sections:
        if sec not in cfg or not isinstance(cfg[sec], dict):
            raise ConfigValidationError(f"Missing required configuration section: '{sec}'")

    telemetry = cfg["telemetry"]
    camera = cfg["camera"]
    vision = cfg["vision"]
    guidance = cfg["guidance"]
    safety = cfg["safety"]

    # 3. Vision parameter validation
    thresh = vision.get("threshold_value", 60)
    if not (0 <= int(thresh) <= 255):
        raise ConfigValidationError(f"vision.threshold_value ({thresh}) must be between 0 and 255.")

    blur = vision.get("blur_kernel_size", 5)
    if blur < 1 or blur % 2 == 0:
        raise ConfigValidationError(f"vision.blur_kernel_size ({blur}) must be a positive odd integer.")

    # 4. Guidance parameter validation
    if guidance.get("max_lateral_speed", 0.5) <= 0:
        raise ConfigValidationError("guidance.max_lateral_speed must be positive.")
    if guidance.get("forward_speed", 0.3) < 0:
        raise ConfigValidationError("guidance.forward_speed must be non-negative.")
    if guidance.get("loop_rate_hz", 10.0) <= 0:
        raise ConfigValidationError("guidance.loop_rate_hz must be positive.")

    # 5. Safety parameter validation
    battery = safety.get("battery", {})
    min_v = battery.get("min_valid_voltage", 5.0)
    max_v = battery.get("max_valid_voltage", 26.0)
    if min_v >= max_v:
        raise ConfigValidationError(
            f"safety.battery min_valid_voltage ({min_v}V) must be strictly less than max_valid_voltage ({max_v}V)."
        )

    # 6. Strict Hardware-Mode Verification
    if mode == "hardware":
        hardware_errors: List[str] = []

        port = telemetry.get("port")
        if not port or str(port).strip().upper() in ("NONE", "NULL", "UNCONFIGURED", ""):
            hardware_errors.append(
                "telemetry.port is unconfigured. Specify physical serial port (e.g. /dev/ttyAMA0 or /dev/ttyACM0)."
            )

        baud = telemetry.get("baud_rate")
        if not baud or int(baud) <= 0:
            hardware_errors.append("telemetry.baud_rate is unconfigured or invalid.")

        cam_dev = camera.get("device_index")
        cam_source = camera.get("source_type")
        if cam_dev is None and cam_source is None:
            hardware_errors.append(
                "camera.device_index or camera.source_type is unconfigured. Specify camera source (e.g. 0, /dev/video0, or csi)."
            )

        cell_count = battery.get("cell_count")
        if not cell_count or int(cell_count) not in (3, 4, 6):
            hardware_errors.append(
                "safety.battery.cell_count is unconfigured. Must specify physical battery cell count (3, 4, or 6)."
            )

        low_v = battery.get("low_voltage")
        if low_v is None or float(low_v) <= 0:
            hardware_errors.append(
                "safety.battery.low_voltage is unconfigured. Specify safe landing voltage threshold."
            )

        rc_channel = safety.get("rc_override", {}).get("channel")
        if rc_channel is None:
            warnings.append(
                "safety.rc_override.channel is unset. Pilot manual override switch monitoring will be inactive."
            )

        if hardware_errors:
            error_details = "\n  - " + "\n  - ".join(hardware_errors)
            raise ConfigValidationError(
                f"Cannot start in 'hardware' mode due to unconfigured hardware parameters:{error_details}\n"
                "Please configure these in config.yaml or supply explicit command-line overrides."
            )

    else:
        # In non-hardware modes (diagnostics, mock), check if hardware settings are unconfigured and note warnings
        if telemetry.get("port") is None:
            warnings.append("telemetry.port is unconfigured (acceptable in diagnostics/mock mode).")
        if camera.get("device_index") is None:
            warnings.append("camera.device_index is unconfigured (acceptable in diagnostics/mock mode).")
        if battery.get("cell_count") is None:
            warnings.append("battery.cell_count is unconfigured (acceptable in diagnostics/mock mode).")

    return warnings
