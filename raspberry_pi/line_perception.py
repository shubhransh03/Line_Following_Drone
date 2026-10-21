"""
line_perception.py — Vision Perception Pipeline for Raspberry Pi Drone Guidance.

Reuses the proven LineDetector from line_detector.py and enhances it with:
  - Dual-ROI analysis (Near band for lateral deviation, Far band for lookahead heading).
  - Lookahead heading estimation (relative to vehicle forward axis).
  - Multi-contour intersection and branching detection.
  - Tracking confidence estimation.
  - Comprehensive debug HUD generation.
"""

from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Ensure project root is available on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from line_detector import LineDetector, LineDetectionResult

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


@dataclass
class PerceptionResult:
    """Strongly-typed perception output for guidance and safety controllers."""
    detected: bool
    lateral_error: Optional[int]              # Lateral pixel error (cx - center_x). Positive = right.
    normalized_error: Optional[float]         # Error normalized to [-1.0, 1.0] by frame_half_width
    heading_error_deg: Optional[float]        # Estimated heading error in degrees (0 = straight ahead)
    confidence: float                         # Confidence score [0.0, 1.0]
    is_intersection: bool                     # True if cross-tracks or multiple branches detected
    near_centroid: Optional[Tuple[int, int]]  # (cx, cy) in near band (bottom)
    far_centroid: Optional[Tuple[int, int]]   # (cx, cy) in far band (top)
    binary_mask: Optional[object]             # Binary threshold mask
    debug_frame: Optional[object]             # Annotated debug HUD frame


class LinePerception:
    """
    Advanced line perception engine wrapping and extending LineDetector.
    """

    def __init__(self,
                 threshold_value: int = 60,
                 blur_kernel_size: int = 5,
                 min_contour_area: int = 500,
                 invert_threshold: bool = True,
                 near_roi_bottom_pct: float = 0.30,
                 far_roi_top_pct: float = 0.30):
        """
        Initialize perception engine.

        Args:
            threshold_value: Grayscale threshold cutoff (0-255).
            blur_kernel_size: Gaussian blur kernel size (positive odd integer).
            min_contour_area: Minimum contour area in px^2 to reject sensor noise.
            invert_threshold: True for dark track on light floor, False for bright line.
            near_roi_bottom_pct: Fraction of frame height at bottom for lateral error.
            far_roi_top_pct: Fraction of frame height at top for lookahead heading.
        """
        self.detector = LineDetector(
            threshold_value=threshold_value,
            blur_kernel_size=blur_kernel_size,
            min_contour_area=min_contour_area,
            invert_threshold=invert_threshold
        )
        self.near_pct = float(max(0.1, min(0.5, near_roi_bottom_pct)))
        self.far_pct = float(max(0.1, min(0.5, far_roi_top_pct)))

    def process(self, frame: Optional[np.ndarray], draw_debug: bool = False) -> PerceptionResult:
        """
        Process a camera frame to extract lateral deviation, heading, and confidence.

        Args:
            frame: Input BGR image from camera or synthetic generator.
            draw_debug: If True, renders visual overlays on debug_frame.

        Returns:
            PerceptionResult dataclass.
        """
        # Guard: invalid or empty frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
            return self._empty_result()

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return self._empty_result()

        center_x = w // 2

        # 1. Primary full-frame detection via LineDetector
        det_result: LineDetectionResult = self.detector.detect(frame, draw_debug=False)
        if not det_result.detected or det_result.binary_mask is None:
            return self._empty_result(frame if draw_debug else None)

        mask = det_result.binary_mask

        # 2. Dual-ROI Analysis (Near band & Far band)
        near_y_start = int(h * (1.0 - self.near_pct))
        far_y_end = int(h * self.far_pct)

        near_band = mask[near_y_start:h, :]
        far_band = mask[0:far_y_end, :]

        near_pt = self._find_band_centroid(near_band, y_offset=near_y_start)
        far_pt = self._find_band_centroid(far_band, y_offset=0)

        # Determine primary lateral error:
        # Prefer near band (closest to drone) if available; fallback to global centroid
        if near_pt is not None:
            primary_cx, primary_cy = near_pt
            lat_error = primary_cx - center_x
        elif det_result.cx is not None and det_result.cy is not None:
            primary_cx, primary_cy = det_result.cx, det_result.cy
            lat_error = det_result.error
        else:
            return self._empty_result(frame if draw_debug else None)

        norm_error = max(-1.0, min(1.0, lat_error / (w / 2.0)))

        # 3. Lookahead Heading Calculation
        heading_deg: Optional[float] = None
        if near_pt is not None and far_pt is not None:
            dx = far_pt[0] - near_pt[0]
            dy = near_pt[1] - far_pt[1]  # Inverted because Y points down in images
            if dy > 0:
                heading_deg = math.degrees(math.atan2(dx, dy))

        # 4. Multi-Contour / Intersection Detection
        is_intersection = False
        if CV2_AVAILABLE:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) >= self.detector.min_contour_area]
            # Case A: Multiple prominent contours (branching track)
            if len(valid_contours) >= 2:
                areas = sorted([cv2.contourArea(c) for c in valid_contours], reverse=True)
                if len(areas) >= 2 and areas[1] >= (0.3 * areas[0]):
                    is_intersection = True
            # Case B: Single connected cross-junction / T-track spanning both axes
            if not is_intersection and valid_contours:
                bx, by, bw, bh = cv2.boundingRect(valid_contours[0])
                if bw >= int(0.6 * w) and bh >= int(0.6 * h):
                    is_intersection = True

        # 5. Tracking Confidence Score
        confidence = self._compute_confidence(det_result, near_pt, far_pt, is_intersection)

        # 6. Debug HUD Rendering (if requested)
        debug_frame = None
        if draw_debug and CV2_AVAILABLE:
            debug_frame = self._render_hud(
                frame, mask, primary_cx, primary_cy, center_x, lat_error,
                heading_deg, confidence, is_intersection, near_pt, far_pt,
                near_y_start, far_y_end
            )

        return PerceptionResult(
            detected=True,
            lateral_error=lat_error,
            normalized_error=norm_error,
            heading_error_deg=heading_deg,
            confidence=confidence,
            is_intersection=is_intersection,
            near_centroid=near_pt,
            far_centroid=far_pt,
            binary_mask=mask,
            debug_frame=debug_frame
        )

    def _find_band_centroid(self, band: np.ndarray, y_offset: int) -> Optional[Tuple[int, int]]:
        """Compute centroid of the largest contour within an image band."""
        if not CV2_AVAILABLE or band is None or band.size == 0:
            return None

        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid = [c for c in contours if cv2.contourArea(c) >= (self.detector.min_contour_area * 0.2)]
        if not valid:
            return None

        c = max(valid, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"]) + y_offset
        return cx, cy

    def _compute_confidence(self,
                            det: LineDetectionResult,
                            near_pt: Optional[Tuple[int, int]],
                            far_pt: Optional[Tuple[int, int]],
                            is_intersection: bool) -> float:
        """Calculate continuous confidence score [0.0, 1.0]."""
        score = 0.5  # Base confidence for valid contour

        # Area bonus: robust line coverage
        if det.contour_area > (self.detector.min_contour_area * 3):
            score += 0.2

        # Dual-band bonus: continuous track across near and far
        if near_pt is not None and far_pt is not None:
            score += 0.3
        elif near_pt is not None:
            score += 0.1

        # Intersection penalty (ambiguity in track direction)
        if is_intersection:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _empty_result(self, frame: Optional[np.ndarray] = None) -> PerceptionResult:
        """Safe fallback result when no line is detected."""
        debug_frame = None
        if frame is not None and CV2_AVAILABLE:
            debug_frame = frame.copy()
            cv2.putText(debug_frame, "NO LINE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return PerceptionResult(
            detected=False,
            lateral_error=None,
            normalized_error=None,
            heading_error_deg=None,
            confidence=0.0,
            is_intersection=False,
            near_centroid=None,
            far_centroid=None,
            binary_mask=None,
            debug_frame=debug_frame
        )

    def _render_hud(self,
                    frame: np.ndarray,
                    mask: np.ndarray,
                    cx: int,
                    cy: int,
                    center_x: int,
                    lat_error: int,
                    heading_deg: Optional[float],
                    confidence: float,
                    is_intersection: bool,
                    near_pt: Optional[Tuple[int, int]],
                    far_pt: Optional[Tuple[int, int]],
                    near_y_start: int,
                    far_y_end: int) -> np.ndarray:
        """Render informational HUD over frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw ROI boundary lines (subtle dashed/dotted or cyan)
        cv2.line(out, (0, far_y_end), (w, far_y_end), (255, 255, 0), 1)
        cv2.line(out, (0, near_y_start), (w, near_y_start), (255, 255, 0), 1)

        # Center reference vertical line (Green)
        cv2.line(out, (center_x, 0), (center_x, h), (0, 255, 0), 1)

        # Primary Centroid (Red)
        cv2.circle(out, (cx, cy), 8, (0, 0, 255), -1)

        # Near and Far points
        if near_pt is not None:
            cv2.circle(out, near_pt, 6, (0, 255, 255), -1)  # Yellow
        if far_pt is not None:
            cv2.circle(out, far_pt, 6, (255, 0, 255), -1)  # Magenta

        # Lookahead vector line
        if near_pt is not None and far_pt is not None:
            cv2.line(out, near_pt, far_pt, (0, 255, 255), 2)

        # Lateral error line
        cv2.line(out, (center_x, cy), (cx, cy), (255, 0, 0), 2)

        # Telemetry Text
        cv2.putText(out, f"Error: {lat_error:+d}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        heading_str = f"{heading_deg:+.1f} deg" if heading_deg is not None else "N/A"
        cv2.putText(out, f"Heading: {heading_str}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(out, f"Conf: {confidence:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if confidence > 0.5 else (0, 0, 255), 1)

        if is_intersection:
            cv2.putText(out, "INTERSECTION", (w - 180, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return out
