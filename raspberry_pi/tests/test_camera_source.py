"""
test_camera_source.py — Unit tests for CameraSourceBase and SyntheticCameraSource.
"""

import unittest
import numpy as np
from raspberry_pi.camera_source import (
    CameraSourceBase,
    SyntheticCameraSource,
    OpenCVCameraSource,
    CameraDiagnostics
)


class SimulatedClock:
    def __init__(self, start=100.0, step=0.033):
        self.time = start
        self.step = step

    def __call__(self):
        t = self.time
        self.time += self.step
        return t

    def advance(self, dt):
        self.time += dt


class TestCameraSource(unittest.TestCase):
    """Test suite for camera frame acquisition and failure accounting."""

    def test_synthetic_camera_start_stop(self):
        """Synthetic camera starts and stops correctly."""
        cam = SyntheticCameraSource(width=640, height=480)
        self.assertFalse(cam.is_opened())
        self.assertTrue(cam.start())
        self.assertTrue(cam.is_opened())
        cam.stop()
        self.assertFalse(cam.is_opened())

    def test_synthetic_camera_frame_dimensions(self):
        """Synthetic camera generates correct BGR NumPy frame dimensions."""
        cam = SyntheticCameraSource(width=320, height=240)
        cam.start()
        success, frame, timestamp = cam.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_synthetic_camera_frame_patterns(self):
        """Synthetic camera correctly generates centered and offset lines."""
        cam = SyntheticCameraSource(width=640, height=480)
        cam.start()

        # Centered line
        cam.set_pattern("center")
        _, frame_center, _ = cam.read_frame()
        # Middle column should have line color (dark/black)
        mid_x = 320
        self.assertTrue(np.all(frame_center[240, mid_x] == (0, 0, 0)))

        # Blank pattern (no line)
        cam.set_pattern("blank")
        _, frame_blank, _ = cam.read_frame()
        self.assertTrue(np.all(frame_blank == (255, 255, 255)))

    def test_synthetic_camera_queued_failures_and_recovery(self):
        """Camera handles queued read drops and resets consecutive counter on recovery."""
        clock = SimulatedClock(start=10.0, step=0.1)
        cam = SyntheticCameraSource(clock_fn=clock)
        cam.start()

        # Queue 3 failures
        cam.queue_failures(3)

        ret1, frame1, t1 = cam.read_frame()
        self.assertFalse(ret1)
        self.assertIsNone(frame1)
        diag = cam.get_diagnostics()
        self.assertEqual(diag.consecutive_drops, 1)

        ret2, frame2, t2 = cam.read_frame()
        self.assertFalse(ret2)
        self.assertEqual(diag.consecutive_drops, 2)

        ret3, frame3, t3 = cam.read_frame()
        self.assertFalse(ret3)
        self.assertEqual(diag.consecutive_drops, 3)

        # 4th read recovers
        ret4, frame4, t4 = cam.read_frame()
        self.assertTrue(ret4)
        self.assertIsNotNone(frame4)
        self.assertEqual(diag.consecutive_drops, 0)
        self.assertEqual(diag.frames_read, 1)
        self.assertEqual(diag.frames_dropped, 3)

    def test_synthetic_camera_custom_frame_injection(self):
        """Injecting a custom frame overrides default pattern generation."""
        cam = SyntheticCameraSource(width=100, height=100)
        cam.start()

        custom = np.zeros((100, 100, 3), dtype=np.uint8)
        custom[50, 50] = (123, 45, 67)
        cam.set_custom_frame(custom)

        success, frame, _ = cam.read_frame()
        self.assertTrue(success)
        self.assertEqual(tuple(frame[50, 50]), (123, 45, 67))

    def test_opencv_camera_source_safe_on_missing_device(self):
        """OpenCVCameraSource on invalid device returns False and handles safely without crashing."""
        clock = SimulatedClock()
        # Device index 99999 should not exist
        cam = OpenCVCameraSource(source=99999, clock_fn=clock)
        # start() returns False when device cannot open
        opened = cam.start()
        self.assertFalse(opened)
        success, frame, _ = cam.read_frame()
        self.assertFalse(success)
        self.assertIsNone(frame)
        self.assertEqual(cam.get_diagnostics().consecutive_drops, 1)


if __name__ == '__main__':
    unittest.main()
