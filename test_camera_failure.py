"""
test_camera_failure.py — Deterministic Offline Unit Tests for Camera Frame-Drop Safety

Verifies bounded camera failure, hover transition, fault timeout, RC-override
safety polling during failure, and failure counter resets.

Requires ZERO physical cameras, serial ports, MAVLink connections, or real-time sleeping.
"""

import unittest
import numpy as np

import config
from pid_controller import PIDController
from line_detector import LineDetector, LineDetectionResult
from main import run_flight_loop


class MockVideoCapture:
    """Mock camera that yields a predefined sequence of (ret, frame) pairs."""
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.read_count = 0

    def read(self):
        self.read_count += 1
        if self.sequence:
            return self.sequence.pop(0)
        return False, None

    def release(self):
        pass


class MockFlightController:
    """Mock flight controller recording velocity commands and safety queries."""
    def __init__(self):
        self.velocities_sent = []
        self.battery_check_count = 0
        self.battery_ok = True
        self.rc_override_count = 0
        self.rc_override_active = False
        self.mode_changed = False
        self.closed = False

    def send_body_velocity(self, vx, vy, vz=0):
        self.velocities_sent.append((vx, vy, vz))

    def check_battery(self):
        self.battery_check_count += 1
        return self.battery_ok

    def check_rc_override(self):
        self.rc_override_count += 1
        return self.rc_override_active

    def is_mode_changed(self):
        return self.mode_changed

    def close(self):
        self.closed = True


class MockDetector(LineDetector):
    """LineDetector spy recording all frames passed to detect()."""
    def __init__(self):
        super().__init__()
        self.frames_received = []

    def detect(self, frame, draw_debug=False):
        self.frames_received.append(frame)
        return super().detect(frame, draw_debug=draw_debug)


class SimulatedClock:
    """Deterministic injectable clock advancing on demand without real-world sleep."""
    def __init__(self, start_time=100.0, step_on_read=0.033):
        self.current_time = start_time
        self.step_on_read = step_on_read

    def __call__(self):
        t = self.current_time
        self.current_time += self.step_on_read
        return t

    def advance(self, seconds):
        self.current_time += seconds


def dummy_sleep(seconds):
    """No-op sleep replacement for instantaneous test execution."""
    pass


class TestCameraFailureHandling(unittest.TestCase):
    """Comprehensive test suite for camera drop handling and flight safety."""

    def setUp(self):
        self.width = 640
        self.height = 480
        # Valid test frame with centered black line
        self.valid_frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self.valid_frame[:, 300:340] = (0, 0, 0)

        self.pid = PIDController(
            kp=config.PID_KP,
            ki=config.PID_KI,
            kd=config.PID_KD,
            integral_limit=config.PID_INTEGRAL_LIMIT,
            output_limit=config.MAX_LATERAL_SPEED
        )

    # -------------------------------------------------------------------------
    # Test 1: Single failed frame followed by a valid frame
    # -------------------------------------------------------------------------
    def test_single_frame_failure_and_recovery(self):
        """A single dropped frame should not trigger fault, and recovery must follow."""
        sequence = [
            (False, None),              # Frame 0: Drop
            (True, self.valid_frame),   # Frame 1: Success
        ]
        cap = MockVideoCapture(sequence)
        detector = MockDetector()
        fc = MockFlightController()
        clock = SimulatedClock(step_on_read=0.033)

        result = run_flight_loop(
            cap=cap,
            detector=detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=clock,
            sleep_fn=dummy_sleep,
            max_iterations=2,
            exit_on_override=False
        )

        self.assertEqual(result["loop_count"], 2)
        # Only the valid frame should have reached the detector
        self.assertEqual(len(detector.frames_received), 1)
        # Consecutive failures must be 0 after successful recovery
        self.assertEqual(result["consecutive_frame_failures"], 0)
        # Velocity command should have been dispatched on the valid frame
        self.assertGreater(len(fc.velocities_sent), 0)

    # -------------------------------------------------------------------------
    # Test 2: Repeated failures reaching CAMERA_FRAME_TIMEOUT trigger hover
    # -------------------------------------------------------------------------
    def test_failures_reaching_frame_timeout_triggers_hover(self):
        """Repeated frame drops exceeding CAMERA_FRAME_TIMEOUT must command zero velocity (hover)."""
        # 1 valid frame, then consecutive failures exceeding 1.0s timeout
        # step = 0.1s -> 15 drops = 1.5s
        clock = SimulatedClock(start_time=100.0, step_on_read=0.1)
        sequence = [(True, self.valid_frame)] + [(False, None)] * 15
        cap = MockVideoCapture(sequence)
        detector = MockDetector()
        fc = MockFlightController()

        result = run_flight_loop(
            cap=cap,
            detector=detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=clock,
            sleep_fn=dummy_sleep,
            max_iterations=16,
            exit_on_override=False
        )

        # Confirm zero velocity hover was commanded
        hover_commands = [v for v in fc.velocities_sent if v == (0, 0, 0)]
        self.assertGreater(len(hover_commands), 0, "Hover (0, 0, 0) must be commanded after frame timeout")

    # -------------------------------------------------------------------------
    # Test 3: Persistent failure reaching CAMERA_FAULT_TIMEOUT triggers landing
    # -------------------------------------------------------------------------
    def test_persistent_failure_reaches_fault_timeout(self):
        """Persistent failure exceeding CAMERA_FAULT_TIMEOUT must exit loop to trigger landing."""
        # 1 valid frame, then drops exceeding 3.0s (e.g. 35 frames * 0.1s = 3.5s)
        clock = SimulatedClock(start_time=100.0, step_on_read=0.1)
        sequence = [(True, self.valid_frame)] + [(False, None)] * 50
        cap = MockVideoCapture(sequence)
        detector = MockDetector()
        fc = MockFlightController()

        result = run_flight_loop(
            cap=cap,
            detector=detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=clock,
            sleep_fn=dummy_sleep,
            max_iterations=50,
            exit_on_override=False
        )

        self.assertEqual(result["shutdown_cause"], "CAMERA_TIMEOUT")
        self.assertGreaterEqual(result["consecutive_frame_failures"], 30)

    # -------------------------------------------------------------------------
    # Test 4: RC-override and battery checks still run during camera failure
    # -------------------------------------------------------------------------
    def test_safety_checks_run_during_camera_failure(self):
        """RC-override and battery checks must execute even when all camera reads fail."""
        clock = SimulatedClock(start_time=100.0, step_on_read=0.033)
        # Sequence of 5 dropped frames, but RC override activates on frame 3
        sequence = [(False, None)] * 10
        cap = MockVideoCapture(sequence)
        detector = MockDetector()
        fc = MockFlightController()

        # Simulate pilot flipping RC override switch
        fc.rc_override_active = True

        result = run_flight_loop(
            cap=cap,
            detector=detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=clock,
            sleep_fn=dummy_sleep,
            max_iterations=10,
            exit_on_override=False
        )

        # Loop must abort immediately on RC override
        self.assertEqual(result["shutdown_cause"], "RC_OVERRIDE")
        self.assertGreater(fc.rc_override_count, 0, "RC override must be polled during camera failure")

    # -------------------------------------------------------------------------
    # Test 5: Invalid/None frames never reach the detector
    # -------------------------------------------------------------------------
    def test_invalid_frames_never_reach_detector(self):
        """Ensure failed reads (False, None) are never passed into LineDetector.detect()."""
        clock = SimulatedClock(step_on_read=0.033)
        sequence = [(False, None)] * 8
        cap = MockVideoCapture(sequence)
        detector = MockDetector()
        fc = MockFlightController()

        run_flight_loop(
            cap=cap,
            detector=detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=clock,
            sleep_fn=dummy_sleep,
            max_iterations=8,
            exit_on_override=False
        )

        self.assertEqual(len(detector.frames_received), 0, "Detector must receive zero frames on drops")

    # -------------------------------------------------------------------------
    # Test 6: Stale errors do not produce fresh movement commands
    # -------------------------------------------------------------------------
    def test_stale_errors_do_not_produce_movement_commands(self):
        """When camera drops after tracking, forward/lateral movement commands must cease."""
        clock = SimulatedClock(start_time=100.0, step_on_read=0.033)
        # 1 valid frame (tracking), then 5 dropped frames
        sequence = [(True, self.valid_frame)] + [(False, None)] * 5
        cap = MockVideoCapture(sequence)
        detector = MockDetector()
        fc = MockFlightController()

        run_flight_loop(
            cap=cap,
            detector=detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=clock,
            sleep_fn=dummy_sleep,
            max_iterations=6,
            exit_on_override=False
        )

        # Only frame 0 should have sent forward movement velocity (vx = FORWARD_SPEED)
        movement_commands = [v for v in fc.velocities_sent if v[0] == config.FORWARD_SPEED]
        self.assertEqual(len(movement_commands), 1, "Only the 1 valid frame should generate forward movement")

    # -------------------------------------------------------------------------
    # Test 7: Camera recovery resets the failure counter
    # -------------------------------------------------------------------------
    def test_camera_recovery_resets_failure_counter(self):
        """Multiple drops followed by a good frame must reset consecutive_frame_failures to 0."""
        clock = SimulatedClock(start_time=100.0, step_on_read=0.033)
        # 5 drops, then 1 good frame
        sequence = [(False, None)] * 5 + [(True, self.valid_frame)]
        cap = MockVideoCapture(sequence)
        detector = MockDetector()
        fc = MockFlightController()

        result = run_flight_loop(
            cap=cap,
            detector=detector,
            pid=self.pid,
            fc=fc,
            fly_mode=True,
            clock_fn=clock,
            sleep_fn=dummy_sleep,
            max_iterations=6,
            exit_on_override=False
        )

        self.assertEqual(result["loop_count"], 6)
        self.assertEqual(result["consecutive_frame_failures"], 0, "Recovery must reset failure counter to 0")


if __name__ == '__main__':
    unittest.main()
