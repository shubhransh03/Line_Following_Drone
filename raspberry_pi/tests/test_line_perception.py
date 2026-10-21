"""
test_line_perception.py — Unit tests for LinePerception engine and dual-ROI tracking.
"""

import unittest
import numpy as np
from raspberry_pi.line_perception import LinePerception, PerceptionResult
from raspberry_pi.camera_source import SyntheticCameraSource


class TestLinePerception(unittest.TestCase):
    """Test suite for vision perception, error conventions, and heading estimation."""

    def setUp(self):
        self.perception = LinePerception(
            threshold_value=60,
            blur_kernel_size=5,
            min_contour_area=300,
            invert_threshold=True,
            near_roi_bottom_pct=0.30,
            far_roi_top_pct=0.30
        )
        self.cam = SyntheticCameraSource(width=640, height=480)
        self.cam.start()

    def test_centered_line_perception(self):
        """Centered vertical line produces near-zero lateral error and high confidence."""
        self.cam.set_pattern("center")
        _, frame, _ = self.cam.read_frame()

        res = self.perception.process(frame, draw_debug=True)
        self.assertTrue(res.detected)
        self.assertIsNotNone(res.lateral_error)
        self.assertAlmostEqual(res.lateral_error, 0, delta=5)
        self.assertAlmostEqual(res.normalized_error, 0.0, delta=0.05)
        self.assertGreater(res.confidence, 0.7)
        self.assertFalse(res.is_intersection)
        self.assertIsNotNone(res.debug_frame)

    def test_left_offset_line_error_convention(self):
        """Line to the LEFT of center must produce negative lateral error (error < 0)."""
        self.cam.set_pattern("offset", offset_px=-100)
        _, frame, _ = self.cam.read_frame()

        res = self.perception.process(frame)
        self.assertTrue(res.detected)
        self.assertIsNotNone(res.lateral_error)
        self.assertAlmostEqual(res.lateral_error, -100, delta=5)
        self.assertLess(res.normalized_error, -0.2)

    def test_right_offset_line_error_convention(self):
        """Line to the RIGHT of center must produce positive lateral error (error > 0)."""
        self.cam.set_pattern("offset", offset_px=+120)
        _, frame, _ = self.cam.read_frame()

        res = self.perception.process(frame)
        self.assertTrue(res.detected)
        self.assertIsNotNone(res.lateral_error)
        self.assertAlmostEqual(res.lateral_error, +120, delta=5)
        self.assertGreater(res.normalized_error, +0.3)

    def test_curved_line_heading_estimation(self):
        """Angled/curved track yields non-zero heading error angle."""
        self.cam.set_pattern("curved", offset_px=0)
        _, frame, _ = self.cam.read_frame()

        res = self.perception.process(frame, draw_debug=True)
        self.assertTrue(res.detected)
        self.assertIsNotNone(res.heading_error_deg)
        # Angled track toward upper-right yields positive heading deviation
        self.assertGreater(res.heading_error_deg, 0.0)

    def test_blank_frame_no_line_detected(self):
        """Blank image produces safe failure result with confidence 0.0."""
        self.cam.set_pattern("blank")
        _, frame, _ = self.cam.read_frame()

        res = self.perception.process(frame)
        self.assertFalse(res.detected)
        self.assertIsNone(res.lateral_error)
        self.assertIsNone(res.normalized_error)
        self.assertEqual(res.confidence, 0.0)

    def test_intersection_detection(self):
        """Cross-tracks or T-junctions trigger is_intersection flag."""
        self.cam.set_pattern("intersection")
        _, frame, _ = self.cam.read_frame()

        res = self.perception.process(frame)
        self.assertTrue(res.detected)
        self.assertTrue(res.is_intersection)

    def test_defensive_handling_of_invalid_frames(self):
        """Defensive guards handle None, empty, or truncated frames without crashing."""
        res_none = self.perception.process(None)
        self.assertFalse(res_none.detected)

        empty_frame = np.array([], dtype=np.uint8)
        res_empty = self.perception.process(empty_frame)
        self.assertFalse(res_empty.detected)


if __name__ == '__main__':
    unittest.main()
