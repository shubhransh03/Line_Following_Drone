"""
line_detector.py — Modular OpenCV Line Detection for Autonomous Drone Guidance

Provides:
  - LineDetectionResult: Structured, strongly-typed result object.
  - LineDetector: Reusable, robust line detection engine.

Coordinate and Sign Conventions:
  - Image frame: Origin (0, 0) is top-left. X points right [0, width), Y points down [0, height).
  - Center X: frame_width // 2.
  - Lateral error: error = cx - center_x
      * error > 0: Line centroid is to the RIGHT of frame center.
                   The vehicle must correct to the right to re-center.
      * error < 0: Line centroid is to the LEFT of frame center.
                   The vehicle must correct to the left to re-center.
      * error == 0: Line centroid is precisely centered.
      * error is None: No valid line detected.
  - Normalized error: error / (frame_width / 2.0), approximately [-1.0, 1.0].
  - Line angle: Angle in degrees of the line's principal axis derived from central moments.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV2_AVAILABLE = False


@dataclass
class LineDetectionResult:
    """Structured result from LineDetector.detect()."""
    detected: bool
    error: Optional[int]              # Lateral pixel error (cx - center_x). Positive = right.
    normalized_error: Optional[float] # Error normalized to [-1.0, 1.0] by frame_half_width
    cx: Optional[int]                 # Centroid X coordinate in pixels
    cy: Optional[int]                 # Centroid Y coordinate in pixels
    angle: Optional[float]            # Estimated line orientation angle in degrees
    contour_area: float               # Area of detected line contour in px^2
    binary_mask: Optional[object]     # Thresholded binary mask (255 = line, 0 = background)
    debug_frame: Optional[object]     # Annotated BGR frame (if requested)

    def as_tuple(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[object]]:
        """
        Backward-compatibility tuple matching legacy detect_line() return:
        (error, cx, cy, thresh)
        """
        return self.error, self.cx, self.cy, self.binary_mask


class LineDetector:
    """
    Modular line detector for downward-facing camera tracking.

    Processes raw BGR camera frames to identify high-contrast lines,
    compute lateral centroid error, and optionally generate debug overlays.
    """

    def __init__(self,
                 threshold_value: int = 60,
                 blur_kernel_size: int = 5,
                 min_contour_area: int = 500,
                 invert_threshold: bool = True):
        """
        Initialize LineDetector with vision parameters.

        Args:
            threshold_value: Binary threshold cutoff (0-255).
            blur_kernel_size: Gaussian blur kernel size (must be positive odd integer).
            min_contour_area: Minimum contour area in px^2 to reject sensor noise.
            invert_threshold: True for dark line on bright background (cv2.THRESH_BINARY_INV),
                              False for bright line on dark background.
        """
        self.threshold_value = int(threshold_value)
        # Ensure blur kernel size is odd and >= 1
        self.blur_kernel_size = blur_kernel_size if (blur_kernel_size % 2 == 1) else (blur_kernel_size + 1)
        self.min_contour_area = max(0, int(min_contour_area))
        self.invert_threshold = bool(invert_threshold)

    def detect(self, frame, draw_debug: bool = False) -> LineDetectionResult:
        """
        Process a single image frame and detect line centroid.

        Args:
            frame: BGR NumPy array from camera or offline source.
            draw_debug: If True, generates annotated debug_frame with overlays.

        Returns:
            LineDetectionResult containing detection status, centroid, error, and masks.
        """
        # Guard: Check OpenCV availability
        if not CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV (cv2) and NumPy are required for LineDetector. "
                "Please install opencv-python and numpy."
            )

        # 1. Guard against invalid, empty, or non-array inputs
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self._empty_result(None)

        if frame.ndim < 2:
            return self._empty_result(None)

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return self._empty_result(None)

        center_x = w // 2

        # 2. Color conversion & Gaussian Blur
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            return self._empty_result(None)

        blur = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # 3. Thresholding
        thresh_type = cv2.THRESH_BINARY_INV if self.invert_threshold else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blur, self.threshold_value, 255, thresh_type)

        # 4. Contour Extraction
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return self._empty_result(thresh, frame if draw_debug else None)

        # 5. Filter by minimum contour area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        if not valid_contours:
            return self._empty_result(thresh, frame if draw_debug else None)

        # 6. Largest contour assumed to be the tracking line
        c = max(valid_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))

        # 7. Spatial Image Moments
        M = cv2.moments(c)
        if M["m00"] == 0:
            return self._empty_result(thresh, frame if draw_debug else None)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Compute lateral error
        error = cx - center_x
        frame_half_width = w / 2.0
        normalized_error = max(-1.0, min(1.0, error / frame_half_width))

        # 8. Principal Axis Orientation from Central Moments
        # mu20 = second central moment X, mu02 = second central moment Y, mu11 = covariance
        mu20 = M.get("mu20", 0)
        mu02 = M.get("mu02", 0)
        mu11 = M.get("mu11", 0)
        angle_deg = None
        if (mu20 - mu02) != 0 or mu11 != 0:
            angle_rad = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
            angle_deg = math.degrees(angle_rad)

        # 9. Optional Debug Overlay Rendering
        debug_frame = None
        if draw_debug:
            debug_frame = self._render_debug_overlay(frame, thresh, cx, cy, center_x, error, True)

        return LineDetectionResult(
            detected=True,
            error=error,
            normalized_error=normalized_error,
            cx=cx,
            cy=cy,
            angle=angle_deg,
            contour_area=area,
            binary_mask=thresh,
            debug_frame=debug_frame
        )

    def _empty_result(self, thresh, frame=None) -> LineDetectionResult:
        """Construct a safe failure result when no valid line is found."""
        debug_frame = None
        if frame is not None and CV2_AVAILABLE:
            h, w = frame.shape[:2]
            center_x = w // 2
            debug_frame = self._render_debug_overlay(frame, thresh, None, None, center_x, None, False)

        return LineDetectionResult(
            detected=False,
            error=None,
            normalized_error=None,
            cx=None,
            cy=None,
            angle=None,
            contour_area=0.0,
            binary_mask=thresh,
            debug_frame=debug_frame
        )

    def _render_debug_overlay(self, frame, thresh, cx, cy, center_x, error, detected):
        """Draw HUD overlay with center line, centroid, and error line."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Center reference vertical line (Green)
        cv2.line(out, (center_x, 0), (center_x, h), (0, 255, 0), 1)

        if detected and cx is not None and cy is not None:
            # Centroid (Red solid circle)
            cv2.circle(out, (cx, cy), 8, (0, 0, 255), -1)

            # Error line from center to centroid (Blue line)
            cv2.line(out, (center_x, cy), (cx, cy), (255, 0, 0), 2)

            # Text overlay
            cv2.putText(out, f"Error: {error:+d}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(out, "LINE LOST", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        status_text = "TRACKING" if detected else "SEARCHING"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(out, status_text, (max(10, w - 180), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return out
