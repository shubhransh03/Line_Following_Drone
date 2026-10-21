"""
pi_main.py — Main Entry Point for Raspberry Pi Autonomous Line-Following Drone.

Execution Modes:
  - 'diagnostics' (DEFAULT): Read-only sensor & telemetry health inspection.
                             NEVER arms motors or sends flight commands.
  - 'mock': Full autonomous flight simulation using synthetic vision & mock MAVLink.
            Runs offline without physical hardware or serial ports.
  - 'hardware': Live flight mode.
                STRICTLY GUARDED: Startup is refused if required physical hardware
                settings (serial port, camera source, battery cells) remain unconfigured.

Usage:
  python pi_main.py --mode diagnostics
  python pi_main.py --mode mock --cycles 50
  python pi_main.py --mode hardware --config config.yaml  # (Guarded)
"""

import argparse
import os
import signal
import sys
import time
from typing import Optional

# Ensure project root is available on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from raspberry_pi.config_loader import load_config, validate_config, ConfigValidationError
from raspberry_pi.camera_source import (
    CameraSourceBase,
    OpenCVCameraSource,
    SyntheticCameraSource,
    CameraDiagnostics
)
from raspberry_pi.line_perception import LinePerception, PerceptionResult
from raspberry_pi.pid_controller import PIDController
from raspberry_pi.mavlink_adapter import (
    MAVLinkAdapterBase,
    MockMAVLinkAdapter,
    HardwareMAVLinkAdapter,
    MAVLinkTelemetry
)
from raspberry_pi.safety_monitor import SafetyMonitor, SafetyAction, SafetyDecision
from battery_monitor import BatteryMonitor


class PiDroneApplication:
    """
    Main application controller managing hardware-independent components,
    diagnostics, and autonomous flight guidance.
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 mode_override: Optional[str] = None,
                 port_override: Optional[str] = None,
                 baud_override: Optional[int] = None,
                 camera_override: Optional[str] = None,
                 max_cycles: Optional[int] = None,
                 # Dependency Injection overrides for offline testing
                 camera_source: Optional[CameraSourceBase] = None,
                 mavlink_adapter: Optional[MAVLinkAdapterBase] = None,
                 perception_engine: Optional[LinePerception] = None,
                 pid_controller: Optional[PIDController] = None,
                 safety_monitor: Optional[SafetyMonitor] = None):
        """
        Initialize drone application with configuration and dependency injection.
        """
        self.config_path = config_path
        self.raw_config = load_config(config_path)

        # Mode determination: CLI override > config file > fallback 'diagnostics'
        self.mode = mode_override or self.raw_config.get("mode", "diagnostics")

        # Apply CLI overrides to config dict
        if port_override:
            self.raw_config["telemetry"]["port"] = port_override
        if baud_override:
            self.raw_config["telemetry"]["baud_rate"] = int(baud_override)
        if camera_override:
            self.raw_config["camera"]["device_index"] = camera_override

        self.max_cycles = max_cycles
        self._running = False
        self._shutdown_requested = False

        # 1. Validate configuration against requested mode
        self.validation_warnings = validate_config(self.raw_config, self.mode)

        # 2. Wire dependencies (use injected objects or construct based on mode)
        self.camera: CameraSourceBase = camera_source or self._build_camera_source()
        self.mavlink: MAVLinkAdapterBase = mavlink_adapter or self._build_mavlink_adapter()
        self.perception: LinePerception = perception_engine or self._build_perception()
        self.pid: PIDController = pid_controller or self._build_pid()
        self.safety: SafetyMonitor = safety_monitor or self._build_safety()

    def _build_camera_source(self) -> CameraSourceBase:
        cam_cfg = self.raw_config["camera"]
        if self.mode == "mock":
            # Deterministic synthetic vision
            return SyntheticCameraSource(
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480)
            )
        else:
            # Diagnostics or Hardware
            dev = cam_cfg.get("device_index", 0)
            return OpenCVCameraSource(
                source=dev if dev is not None else 0,
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                fps=cam_cfg.get("fps", 30),
                retry_sleep_sec=cam_cfg.get("retry_sleep_sec", 0.05)
            )

    def _build_mavlink_adapter(self) -> MAVLinkAdapterBase:
        telem_cfg = self.raw_config["telemetry"]
        if self.mode == "mock":
            return MockMAVLinkAdapter(
                initial_mode="STABILIZE",
                initial_voltage=12.4
            )
        elif self.mode == "diagnostics":
            port = telem_cfg.get("port")
            if port and str(port).strip().upper() not in ("NONE", "NULL", "UNCONFIGURED", ""):
                try:
                    return HardwareMAVLinkAdapter(
                        port=port,
                        baud=telem_cfg.get("baud_rate", 57600),
                        connect_timeout=telem_cfg.get("connect_timeout", 5.0),
                        allow_hardware=True
                    )
                except Exception as e:
                    print(f"[PI_APP] Notice: Hardware port '{port}' unavailable ({e}). Diagnostics falling back to MockMAVLinkAdapter.")
                    return MockMAVLinkAdapter()
            else:
                return MockMAVLinkAdapter()
        elif self.mode == "hardware":
            return HardwareMAVLinkAdapter(
                port=telem_cfg["port"],
                baud=telem_cfg.get("baud_rate", 57600),
                connect_timeout=telem_cfg.get("connect_timeout", 30.0),
                allow_hardware=True
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _build_perception(self) -> LinePerception:
        vis_cfg = self.raw_config["vision"]
        return LinePerception(
            threshold_value=vis_cfg.get("threshold_value", 60),
            blur_kernel_size=vis_cfg.get("blur_kernel_size", 5),
            min_contour_area=vis_cfg.get("min_contour_area", 500),
            invert_threshold=vis_cfg.get("invert_threshold", True),
            near_roi_bottom_pct=vis_cfg.get("near_roi_bottom_pct", 0.30),
            far_roi_top_pct=vis_cfg.get("far_roi_top_pct", 0.30)
        )

    def _build_pid(self) -> PIDController:
        guidance = self.raw_config["guidance"]
        return PIDController(
            kp=guidance.get("kp", 0.3),
            ki=guidance.get("ki", 0.0),
            kd=guidance.get("kd", 0.1),
            integral_limit=guidance.get("integral_limit", 50.0),
            output_limit=guidance.get("max_lateral_speed", 0.5)
        )

    def _build_safety(self) -> SafetyMonitor:
        safety_cfg = self.raw_config["safety"]
        batt_cfg = safety_cfg.get("battery", {})
        cam_cfg = self.raw_config["camera"]
        rc_cfg = safety_cfg.get("rc_override", {})

        batt_mon = BatteryMonitor(
            low_threshold=batt_cfg.get("low_voltage", 10.5),
            consecutive_low_required=batt_cfg.get("consecutive_low_required", 3),
            stale_timeout=batt_cfg.get("stale_timeout", 3.0),
            min_valid_voltage=batt_cfg.get("min_valid_voltage", 5.0),
            max_valid_voltage=batt_cfg.get("max_valid_voltage", 26.0),
            hysteresis=batt_cfg.get("hysteresis", 0.3),
            recovery_count=batt_cfg.get("recovery_count", 3)
        )

        return SafetyMonitor(
            battery_monitor=batt_mon,
            camera_frame_timeout=cam_cfg.get("frame_timeout", 1.0),
            camera_fault_timeout=cam_cfg.get("fault_timeout", 3.0),
            telemetry_timeout=self.raw_config["telemetry"].get("heartbeat_timeout", 3.0),
            line_lost_timeout=safety_cfg.get("line_lost_timeout", 3.0),
            rc_override_channel=rc_cfg.get("channel"),
            rc_override_threshold=rc_cfg.get("pwm_threshold", 1800),
            rc_override_enabled=rc_cfg.get("enabled", False)
        )

    # -------------------------------------------------------------------------
    # Execution Dispatcher
    # -------------------------------------------------------------------------

    def run(self) -> int:
        """Run the application in the configured mode."""
        print(f"================================================================")
        print(f"  Autonomous Line-Following Drone — Raspberry Pi Application   ")
        print(f"  Active Mode: [{self.mode.upper()}]                            ")
        print(f"================================================================")

        for w in self.validation_warnings:
            print(f"[PI_APP] Config Warning: {w}")

        if self.mode == "diagnostics":
            return self.run_diagnostics()
        elif self.mode in ("mock", "hardware"):
            return self.run_flight_loop()
        else:
            print(f"[PI_APP] ERROR: Unrecognized mode '{self.mode}'")
            return 1

    # -------------------------------------------------------------------------
    # Mode 1: Diagnostics (Read-Only, Safe, No Flight Commands)
    # -------------------------------------------------------------------------

    def run_diagnostics(self) -> int:
        """
        Execute read-only health checks on camera and MAVLink telemetry.
        Never arms motors, never commands takeoff, never moves servos.
        """
        print("[DIAG] Starting read-only diagnostics...")
        issues = 0

        # 1. Camera Diagnostic
        print("[DIAG] Inspecting camera source...")
        try:
            opened = self.camera.start()
            if not opened:
                print("  [FAIL] Camera failed to open.")
                issues += 1
            else:
                success, frame, timestamp = self.camera.read_frame()
                if not success or frame is None:
                    print("  [FAIL] Camera opened but frame capture failed.")
                    issues += 1
                else:
                    h, w = frame.shape[:2]
                    print(f"  [PASS] Camera frame captured: {w}x{h} px at t={timestamp:.2f}s")
                    # Test perception
                    res = self.perception.process(frame, draw_debug=False)
                    print(f"  [PASS] Line perception: detected={res.detected}, confidence={res.confidence:.2f}")
        except Exception as e:
            print(f"  [ERROR] Camera exception: {e}")
            issues += 1
        finally:
            self.camera.stop()

        # 2. Telemetry Diagnostic
        print("[DIAG] Inspecting MAVLink telemetry adapter...")
        try:
            connected = self.mavlink.is_connected()
            if not connected:
                print("  [FAIL] Autopilot adapter not connected.")
                issues += 1
            else:
                telem = self.mavlink.get_telemetry()
                fresh = self.mavlink.is_heartbeat_fresh(3.0)
                print(f"  [PASS] Autopilot connected. Mode: {telem.mode}, Armed: {telem.armed}")
                print(f"  [PASS] Heartbeat age: {telem.heartbeat_age_sec:.2f}s (Fresh: {fresh})")
                print(f"  [PASS] Battery: {telem.voltage}V ({telem.battery_percentage}%)")
                print(f"  [PASS] Altitude: {telem.relative_altitude_m}m")
                print(f"  [PASS] RC Channels: {telem.rc_channels}")
        except Exception as e:
            print(f"  [ERROR] Telemetry exception: {e}")
            issues += 1

        print("----------------------------------------------------------------")
        if issues == 0:
            print("[DIAG] All diagnostic health checks PASSED.")
            return 0
        else:
            print(f"[DIAG] Diagnostics completed with {issues} issue(s).")
            return 1

    # -------------------------------------------------------------------------
    # Mode 2 & 3: Autonomous Flight Loop (Mock or Guarded Hardware)
    # -------------------------------------------------------------------------

    def run_flight_loop(self) -> int:
        """
        Execute the autonomous line-following flight loop.
        """
        guidance = self.raw_config["guidance"]
        loop_interval = 1.0 / guidance.get("loop_rate_hz", 10.0)
        forward_speed = guidance.get("forward_speed", 0.3)
        target_alt = guidance.get("target_altitude", 2.0)
        takeoff_timeout = guidance.get("takeoff_timeout", 30.0)

        # 1. Start camera
        if not self.camera.start():
            print("[FLIGHT] ERROR: Camera failed to start. Aborting.")
            return 1

        # 2. Verify autopilot connection
        if not self.mavlink.is_connected():
            print("[FLIGHT] ERROR: Autopilot not connected. Aborting.")
            self.camera.stop()
            return 1

        # 3. Takeoff Sequence
        print(f"[FLIGHT] Initiating arm and takeoff to {target_alt:.1f}m...")
        success = self.mavlink.arm_and_takeoff(target_altitude=target_alt, timeout=takeoff_timeout)
        if not success:
            print("[FLIGHT] ERROR: Takeoff failed! Terminating flight loop.")
            self.camera.stop()
            self.mavlink.close()
            return 1

        print("[FLIGHT] Takeoff successful. Entering line-following control loop...")
        self.safety.start_monitoring()
        self._running = True
        cycle_count = 0
        exit_code = 0

        # Signal handlers for clean shutdown
        def _handle_signal(sig, frame):
            print(f"\n[FLIGHT] Signal {sig} received. Initiating controlled landing...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while self._running and not self._shutdown_requested:
                cycle_start = time.monotonic()
                cycle_count += 1

                # Step A: Read camera frame
                frame_ok, frame, t_frame = self.camera.read_frame()
                if frame_ok and frame is not None:
                    self.safety.record_frame_success(t_frame)
                    # Step B: Process line perception
                    res = self.perception.process(frame, draw_debug=False)
                    line_detected = res.detected
                else:
                    self.safety.record_frame_failure(cycle_start)
                    res = None
                    line_detected = False

                # Step C: Evaluate Safety Monitor
                decision = self.safety.evaluate(self.mavlink, line_detected=line_detected, now=cycle_start)

                if decision.action == SafetyAction.RELEASE_CONTROL:
                    print(f"[FLIGHT] Control surrendered: {decision.reasons}")
                    self._running = False
                    break

                elif decision.action == SafetyAction.ABORT:
                    print(f"[FLIGHT] Safety ABORT triggered ({decision.fault_code}): {decision.reasons}")
                    exit_code = 2
                    self._running = False
                    break

                elif decision.action == SafetyAction.HOVER:
                    # Arrest velocity
                    self.mavlink.send_velocity(0.0, 0.0, 0.0)

                elif decision.action == SafetyAction.CONTINUE:
                    if res is not None and res.detected and res.normalized_error is not None:
                        # PID lateral speed computation
                        # normalized_error is in [-1.0, 1.0]
                        lateral_speed = self.pid.compute(res.normalized_error, dt=loop_interval)
                        self.mavlink.send_velocity(vx=forward_speed, vy=lateral_speed, vz=0.0)
                    else:
                        self.mavlink.send_velocity(0.0, 0.0, 0.0)

                # Check max cycles limit (for test bounds)
                if self.max_cycles is not None and cycle_count >= self.max_cycles:
                    print(f"[FLIGHT] Max test cycles ({self.max_cycles}) reached. Concluding test run.")
                    break

                # Sleep to maintain loop rate
                elapsed = time.monotonic() - cycle_start
                sleep_time = max(0.0, loop_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[FLIGHT] CRITICAL UNHANDLED EXCEPTION in flight loop: {e}")
            exit_code = 3
        finally:
            print("[FLIGHT] Exiting control loop. Ordering landing...")
            self.mavlink.land(timeout=self.raw_config["safety"].get("landing_timeout", 60.0))
            self.camera.stop()
            self.mavlink.close()
            print("[FLIGHT] Teardown complete.")

        return exit_code


def main():
    parser = argparse.ArgumentParser(description="Autonomous Line-Following Drone — Raspberry Pi Application")
    parser.add_argument("--mode", choices=["diagnostics", "mock", "hardware"], default="diagnostics",
                        help="Operating mode (default: diagnostics)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port override (e.g. /dev/ttyAMA0)")
    parser.add_argument("--baud", type=int, default=None,
                        help="Baud rate override (e.g. 57600)")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera source override (e.g. 0 or /dev/video0)")
    parser.add_argument("--cycles", type=int, default=None,
                        help="Maximum control cycles to execute (useful for mock tests)")

    args = parser.parse_args()

    try:
        app = PiDroneApplication(
            config_path=args.config,
            mode_override=args.mode,
            port_override=args.port,
            baud_override=args.baud,
            camera_override=args.camera,
            max_cycles=args.cycles
        )
        sys.exit(app.run())
    except ConfigValidationError as e:
        print(f"\n[PI_MAIN] CONFIGURATION ERROR:\n{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n[PI_MAIN] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
