"""
test_offline_integration.py — End-to-End Offline Integration Test Suite

Tests the integrated flight loop (run_flight_loop) across vision detection,
PID correction, flight controller commanding, and all safety state transitions:
  - Normal line tracking and velocity dispatch
  - Line loss, hover damping, and line recovery
  - Sustained line loss timeout abort
  - Camera frame drop and recovery
  - Camera frame timeout (hover) and persistent fault timeout (abort)
  - RC manual override interception
  - Autopilot mode change interception
  - Battery low telemetry abort (during tracking and during camera failure)
"""

import unittest
import numpy as np
import time

import config
from pid_controller import PIDController
from line_detector import LineDetector
from battery_monitor import BatteryMonitor
from main import run_flight_loop


def make_frame(line_center_x=320, width=640, height=480, line_width=40):
    """Create a synthetic BGR frame with a dark line on a bright background."""
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    if line_center_x is not None:
        x1 = max(0, int(line_center_x - line_width // 2))
        x2 = min(width, int(line_center_x + line_width // 2))
        frame[:, x1:x2] = (10, 10, 10)  # Dark tape line
    return frame


class MockCapture:
    """Mock video capture source yielding predefined frames or sequences."""
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0
        self.released = False

    def read(self):
        if self.index < len(self.sequence):
            item = self.sequence[self.index]
            self.index += 1
            if callable(item):
                return item()
            return item
        return False, None

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True


class MockFC:
    """Mock FlightController capturing velocity commands and simulating safety triggers."""
    def __init__(self, battery_ok=True, rc_override=False, mode="GUIDED", battery_state=None):
        self.battery_ok = battery_ok
        self.rc_override = rc_override
        self.mode_name = mode
        self.velocity_commands = []
        self.land_called = False
        self.closed = False
        self.battery_monitor = BatteryMonitor()
        if battery_state == "STALE":
            self.battery_monitor._last_valid_time = 0.0  # Expired timestamp triggers STALE
            self.battery_monitor._state = "STALE"
        elif battery_state == "UNKNOWN":
            self.battery_monitor._last_valid_time = None
            self.battery_monitor._state = "UNKNOWN"
        elif battery_state == "LOW" or not battery_ok:
            self.battery_monitor._last_valid_time = time.monotonic()
            self.battery_monitor._state = "LOW"

    def send_body_velocity(self, vx, vy, vz=0):
        self.velocity_commands.append((vx, vy, vz))

    def check_battery(self):
        return self.battery_ok

    def check_rc_override(self):
        return self.rc_override

    def is_mode_changed(self):
        return self.mode_name != "GUIDED"

    def land(self, *args, **kwargs):
        self.land_called = True
        return True

    def close(self):
        self.closed = True


class TestOfflineIntegration(unittest.TestCase):
    """End-to-end integration tests for run_flight_loop."""

    def setUp(self):
        self.orig_debug = config.SHOW_DEBUG_FRAMES
        self.orig_retry_sleep = config.CAMERA_RETRY_SLEEP
        config.SHOW_DEBUG_FRAMES = False
        config.CAMERA_RETRY_SLEEP = 0.0  # Fast execution

        self.detector = LineDetector(
            threshold_value=config.THRESHOLD_VALUE,
            blur_kernel_size=config.BLUR_KERNEL_SIZE,
            min_contour_area=config.MIN_CONTOUR_AREA,
            invert_threshold=True
        )

        self.pid = PIDController(
            kp=config.PID_KP,
            ki=config.PID_KI,
            kd=config.PID_KD,
            integral_limit=config.PID_INTEGRAL_LIMIT,
            output_limit=config.MAX_LATERAL_SPEED
        )

    def tearDown(self):
        config.SHOW_DEBUG_FRAMES = self.orig_debug
        config.CAMERA_RETRY_SLEEP = self.orig_retry_sleep

    # -------------------------------------------------------------------------
    # Test 1: Normal Line Tracking
    # -------------------------------------------------------------------------
    def test_normal_tracking_dispatches_forward_and_lateral_velocities(self):
        """
        When line is detected right of center (x=420 > 320),
        positive lateral velocity and forward velocity are sent to FC.
        """
        frame_right = make_frame(line_center_x=420)
        cap = MockCapture([(True, frame_right)] * 5)
        fc = MockFC()

        t = [100.0]
        def mock_clock():
            t[0] += 0.033
            return t[0]

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=mock_clock,
            sleep_fn=lambda s: None,
            max_iterations=5,
            exit_on_override=False
        )

        self.assertEqual(summary['loop_count'], 5)
        self.assertEqual(summary['shutdown_cause'], "MAX_ITERATIONS")
        self.assertEqual(len(fc.velocity_commands), 5)

        for vx, vy, vz in fc.velocity_commands:
            self.assertAlmostEqual(vx, config.FORWARD_SPEED)
            self.assertGreater(vy, 0.0, "Line to right must command positive vy")
            self.assertEqual(vz, 0)

    # -------------------------------------------------------------------------
    # Test 2: Line Loss and Recovery
    # -------------------------------------------------------------------------
    def test_line_loss_commands_hover_and_resumes_on_recovery(self):
        """
        Line lost temporarily commands hover (vx=0, vy=0, vz=0).
        When line reappears, forward velocity resumes.
        """
        track_frame = make_frame(line_center_x=320)
        lost_frame = make_frame(line_center_x=None)  # Blank white

        # Sequence: 2 tracking frames -> 2 lost frames -> 2 tracking frames
        frames = (
            [(True, track_frame)] * 2 +
            [(True, lost_frame)] * 2 +
            [(True, track_frame)] * 2
        )
        cap = MockCapture(frames)
        fc = MockFC()

        t = [100.0]
        def mock_clock():
            t[0] += 0.1
            return t[0]

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=mock_clock,
            sleep_fn=lambda s: None,
            max_iterations=6,
            exit_on_override=False
        )

        self.assertEqual(summary['loop_count'], 6)
        self.assertEqual(len(fc.velocity_commands), 6)

        # Frames 0-1: tracking (vx = FORWARD_SPEED)
        self.assertAlmostEqual(fc.velocity_commands[0][0], config.FORWARD_SPEED)
        self.assertAlmostEqual(fc.velocity_commands[1][0], config.FORWARD_SPEED)

        # Frames 2-3: line lost -> hover commanded (vx=0, vy=0, vz=0)
        self.assertEqual(fc.velocity_commands[2], (0, 0, 0))
        self.assertEqual(fc.velocity_commands[3], (0, 0, 0))

        # Frames 4-5: recovery -> tracking resumes
        self.assertAlmostEqual(fc.velocity_commands[4][0], config.FORWARD_SPEED)
        self.assertAlmostEqual(fc.velocity_commands[5][0], config.FORWARD_SPEED)

    # -------------------------------------------------------------------------
    # Test 3: Sustained Line Loss Timeout
    # -------------------------------------------------------------------------
    def test_sustained_line_loss_triggers_timeout_shutdown(self):
        """Line lost exceeding LINE_LOST_TIMEOUT (3.0s) triggers loop termination."""
        lost_frame = make_frame(line_center_x=None)
        cap = MockCapture([(True, lost_frame)] * 50)
        fc = MockFC()

        t = [100.0]
        def advancing_clock():
            t[0] += 0.5  # 0.5s per frame -> 7 frames = 3.5s > 3.0s
            return t[0]

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=advancing_clock,
            sleep_fn=lambda s: None,
            max_iterations=20,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "LINE_LOST_TIMEOUT")

    # -------------------------------------------------------------------------
    # Test 4: Camera Frame Drop & Recovery
    # -------------------------------------------------------------------------
    def test_camera_drop_and_recovery(self):
        """Dropped camera reads reset PID without aborting if recovered quickly."""
        track_frame = make_frame(line_center_x=320)
        # Sequence: 2 good -> 2 dropped -> 2 good
        frames = (
            [(True, track_frame)] * 2 +
            [(False, None)] * 2 +
            [(True, track_frame)] * 2
        )
        cap = MockCapture(frames)
        fc = MockFC()

        t = [100.0]
        def mock_clock():
            t[0] += 0.05
            return t[0]

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=mock_clock,
            sleep_fn=lambda s: None,
            max_iterations=6,
            exit_on_override=False
        )

        self.assertEqual(summary['loop_count'], 6)
        self.assertEqual(summary['consecutive_frame_failures'], 0,
                         "Counter must reset to 0 after recovery")

    # -------------------------------------------------------------------------
    # Test 5: Camera Frame Timeout Commands Hover
    # -------------------------------------------------------------------------
    def test_camera_frame_timeout_commands_hover(self):
        """Camera drops exceeding CAMERA_FRAME_TIMEOUT command zero velocity."""
        # 1 initial good frame, then drops
        track_frame = make_frame(line_center_x=320)
        frames = [(True, track_frame)] + [(False, None)] * 15
        cap = MockCapture(frames)
        fc = MockFC()

        t = [100.0]
        def mock_clock():
            t[0] += 0.15  # 0.15s per frame
            return t[0]

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=mock_clock,
            sleep_fn=lambda s: None,
            max_iterations=10,
            exit_on_override=False
        )

        # After elapsed >= 1.0s, (0, 0, 0) hover is commanded
        self.assertIn((0, 0, 0), fc.velocity_commands)

    # -------------------------------------------------------------------------
    # Test 6: Camera Persistent Fault Aborts Flight
    # -------------------------------------------------------------------------
    def test_camera_persistent_fault_aborts_flight(self):
        """Camera drops exceeding CAMERA_FAULT_TIMEOUT terminate loop with CAMERA_TIMEOUT."""
        cap = MockCapture([(False, None)] * 50)
        fc = MockFC()

        t = [100.0]
        def mock_clock():
            t[0] += 0.5  # 0.5s per frame -> 7 frames = 3.5s > 3.0s
            return t[0]

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=mock_clock,
            sleep_fn=lambda s: None,
            max_iterations=20,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "CAMERA_TIMEOUT")

    # -------------------------------------------------------------------------
    # Test 7: RC Override Interception
    # -------------------------------------------------------------------------
    def test_rc_override_terminates_loop(self):
        """Pilot triggering RC override switch immediately terminates flight loop."""
        track_frame = make_frame(line_center_x=320)
        cap = MockCapture([(True, track_frame)] * 10)
        fc = MockFC(rc_override=True)

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            max_iterations=10,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "RC_OVERRIDE")

    # -------------------------------------------------------------------------
    # Test 8: External Mode Change Interception
    # -------------------------------------------------------------------------
    def test_external_mode_change_terminates_loop(self):
        """Mode changed from GUIDED (e.g. pilot switched to ALT_HOLD) terminates loop."""
        track_frame = make_frame(line_center_x=320)
        cap = MockCapture([(True, track_frame)] * 10)
        fc = MockFC(mode="ALT_HOLD")

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            max_iterations=10,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "RC_OVERRIDE")

    # -------------------------------------------------------------------------
    # Test 9: Low Battery Telemetry Aborts Flight
    # -------------------------------------------------------------------------
    def test_low_battery_aborts_flight(self):
        """Battery dropping below threshold triggers emergency LOW_BATTERY shutdown."""
        track_frame = make_frame(line_center_x=320)
        cap = MockCapture([(True, track_frame)] * 10)
        fc = MockFC(battery_ok=False)

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            max_iterations=10,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "LOW_BATTERY")

    # -------------------------------------------------------------------------
    # Test 10: Low Battery Caught Even During Camera Failure
    # -------------------------------------------------------------------------
    def test_low_battery_caught_during_camera_failure(self):
        """Battery failure is caught and acted upon even when camera reads fail."""
        cap = MockCapture([(False, None)] * 10)
        fc = MockFC(battery_ok=False)

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            max_iterations=10,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "LOW_BATTERY")

    # -------------------------------------------------------------------------
    # Test 11: Stale Battery Telemetry Distinct from Low Battery
    # -------------------------------------------------------------------------
    def test_stale_battery_telemetry_aborts_flight_with_stale_cause(self):
        """Stale battery telemetry triggers STALE_BATTERY rather than misleading LOW_BATTERY."""
        track_frame = make_frame(line_center_x=320)
        cap = MockCapture([(True, track_frame)] * 10)
        fc = MockFC(battery_ok=False, battery_state="STALE")

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            max_iterations=10,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "STALE_BATTERY")

    # -------------------------------------------------------------------------
    # Test 12: Unknown / Uninitialized Battery Telemetry Aborts Explicitly
    # -------------------------------------------------------------------------
    def test_unknown_battery_telemetry_aborts_flight_with_unknown_cause(self):
        """Uninitialized battery telemetry triggers UNKNOWN_BATTERY shutdown."""
        track_frame = make_frame(line_center_x=320)
        cap = MockCapture([(True, track_frame)] * 10)
        fc = MockFC(battery_ok=False, battery_state="UNKNOWN")

        summary = run_flight_loop(
            cap=cap,
            detector=self.detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            max_iterations=10,
            exit_on_override=False
        )

        self.assertEqual(summary['shutdown_cause'], "UNKNOWN_BATTERY")


if __name__ == '__main__':
    unittest.main()
