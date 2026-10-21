"""
test_line_detector.py — Deterministic Offline Unit Tests for LineDetector

Tests the modular vision pipeline using synthetic in-memory images.
Requires ZERO physical cameras, serial ports, MAVLink connections, or external files.
"""

import unittest
import math

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV2_AVAILABLE = False

from line_detector import LineDetector, LineDetectionResult


class TestLineDetector(unittest.TestCase):
    """Offline unit test suite for LineDetector using synthetic test frames."""

    @classmethod
    def setUpClass(cls):
        if not CV2_AVAILABLE:
            raise unittest.SkipTest(
                "OpenCV and NumPy are not installed in this environment. "
                "Install opencv-python and numpy to run line detector tests."
            )

    def setUp(self):
        self.width = 640
        self.height = 480
        self.center_x = self.width // 2  # 320
        self.detector = LineDetector(
            threshold_value=60,
            blur_kernel_size=5,
            min_contour_area=500,
            invert_threshold=True
        )

    def _create_synthetic_frame(self, line_center_x, line_width=40):
        """
        Create a synthetic BGR frame with a pure white background (255)
        and a vertical black strip (0) at the specified x center.
        """
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        x1 = max(0, int(line_center_x - line_width // 2))
        x2 = min(self.width, int(line_center_x + line_width // 2))
        frame[:, x1:x2] = (0, 0, 0)
        return frame

    # -------------------------------------------------------------------------
    # 1. Centered Line Test
    # -------------------------------------------------------------------------
    def test_centered_line(self):
        """A line in the exact center of the frame should produce near-zero error."""
        frame = self._create_synthetic_frame(self.center_x, line_width=40)
        result = self.detector.detect(frame)

        self.assertTrue(result.detected, "Centered line should be detected")
        self.assertIsNotNone(result.cx)
        self.assertIsNotNone(result.cy)
        self.assertIsNotNone(result.error)

        # Centroid X should be within ±2 pixels of center_x (320)
        self.assertAlmostEqual(result.cx, self.center_x, delta=2)
        # Lateral error should be approximately 0
        self.assertAlmostEqual(result.error, 0, delta=2)
        # Normalized error should be near 0.0
        self.assertAlmostEqual(result.normalized_error, 0.0, delta=0.01)

    # -------------------------------------------------------------------------
    # 2. Right-Shifted Line (Positive Error)
    # -------------------------------------------------------------------------
    def test_line_to_the_right(self):
        """A line to the right of center should produce positive error."""
        target_x = 420  # +100 px from center (320)
        frame = self._create_synthetic_frame(target_x, line_width=40)
        result = self.detector.detect(frame)

        self.assertTrue(result.detected)
        self.assertGreater(result.error, 0, "Line to right must produce positive error")
        self.assertAlmostEqual(result.cx, target_x, delta=2)
        self.assertAlmostEqual(result.error, 100, delta=2)
        self.assertGreater(result.normalized_error, 0.0)

    # -------------------------------------------------------------------------
    # 3. Left-Shifted Line (Negative Error)
    # -------------------------------------------------------------------------
    def test_line_to_the_left(self):
        """A line to the left of center should produce negative error."""
        target_x = 220  # -100 px from center (320)
        frame = self._create_synthetic_frame(target_x, line_width=40)
        result = self.detector.detect(frame)

        self.assertTrue(result.detected)
        self.assertLess(result.error, 0, "Line to left must produce negative error")
        self.assertAlmostEqual(result.cx, target_x, delta=2)
        self.assertAlmostEqual(result.error, -100, delta=2)
        self.assertLess(result.normalized_error, 0.0)

    # -------------------------------------------------------------------------
    # 4. Backward-Compatibility Tuple Unpacking
    # -------------------------------------------------------------------------
    def test_tuple_unpacking_compatibility(self):
        """Verify result.as_tuple() matches legacy detect_line() return format."""
        frame = self._create_synthetic_frame(self.center_x)
        result = self.detector.detect(frame)

        error, cx, cy, thresh = result.as_tuple()
        self.assertEqual(error, result.error)
        self.assertEqual(cx, result.cx)
        self.assertEqual(cy, result.cy)
        self.assertIsNotNone(thresh)
        self.assertEqual(thresh.shape, (self.height, self.width))

    # -------------------------------------------------------------------------
    # 5. Detection Failure (No Line / Pure White Frame)
    # -------------------------------------------------------------------------
    def test_no_line_blank_white_frame(self):
        """A blank white frame (no dark line) should report failure explicitly."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        result = self.detector.detect(frame)

        self.assertFalse(result.detected)
        self.assertIsNone(result.error)
        self.assertIsNone(result.cx)
        self.assertIsNone(result.cy)
        self.assertEqual(result.contour_area, 0.0)

    # -------------------------------------------------------------------------
    # 6. Noise Rejection (Small Contour Area)
    # -------------------------------------------------------------------------
    def test_noise_contour_below_min_area(self):
        """Small noise specks (< min_contour_area) should be rejected."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        # Draw a tiny 10x10 dot (area = 100 px^2 < min_contour_area = 500)
        frame[100:110, 100:110] = (0, 0, 0)
        result = self.detector.detect(frame)

        self.assertFalse(result.detected, "Contour smaller than min_contour_area must be rejected")
        self.assertIsNone(result.error)

    # -------------------------------------------------------------------------
    # 7. Invalid Inputs & Empty Frames Handling
    # -------------------------------------------------------------------------
    def test_invalid_and_empty_inputs(self):
        """LineDetector must handle None, empty, or degenerate arrays without crashing."""
        # Test None
        res_none = self.detector.detect(None)
        self.assertFalse(res_none.detected)

        # Test empty array
        res_empty = self.detector.detect(np.array([], dtype=np.uint8))
        self.assertFalse(res_empty.detected)

        # Test 1D array
        res_1d = self.detector.detect(np.zeros(100, dtype=np.uint8))
        self.assertFalse(res_1d.detected)

        # Test 0x0 image
        res_0x0 = self.detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertFalse(res_0x0.detected)

    # -------------------------------------------------------------------------
    # 8. Debug Overlay Verification
    # -------------------------------------------------------------------------
    def test_debug_overlay_generation(self):
        """When draw_debug=True, an annotated BGR frame matching input size must be produced."""
        frame = self._create_synthetic_frame(self.center_x)
        result = self.detector.detect(frame, draw_debug=True)

        self.assertIsNotNone(result.debug_frame)
        self.assertEqual(result.debug_frame.shape, frame.shape)
        self.assertEqual(result.debug_frame.dtype, np.uint8)

        # Also verify debug frame on failed detection
        blank = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        res_fail = self.detector.detect(blank, draw_debug=True)
        self.assertIsNotNone(res_fail.debug_frame)
        self.assertEqual(res_fail.debug_frame.shape, blank.shape)


if __name__ == '__main__':
    unittest.main()
