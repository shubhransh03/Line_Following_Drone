"""
camera_source.py — Modular Camera Abstraction for Raspberry Pi Autonomous Drone Guidance.

Provides:
  - CameraSourceBase: Abstract interface for frame acquisition and health monitoring.
  - OpenCVCameraSource: Real hardware camera adapter (V4L2, CSI libcamera GStreamer, USB).
  - SyntheticCameraSource: Injectable deterministic synthetic frame generator for offline tests & simulation.
  - CameraDiagnostics: Telemetry and statistics tracking (FPS, drops, latency).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


@dataclass
class CameraDiagnostics:
    """Operational metrics for camera health and throughput."""
    frames_read: int = 0
    frames_dropped: int = 0
    consecutive_drops: int = 0
    last_frame_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    failure_duration: float = 0.0
    effective_fps: float = 0.0
    is_opened: bool = False
    source_identifier: str = "unknown"


class CameraSourceBase(ABC):
    """Abstract interface defining the camera contract."""

    @abstractmethod
    def start(self) -> bool:
        """Initialize and open the camera stream."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read the next frame.

        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray], timestamp: float).
            If success is False, frame is None.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release camera resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera device/stream is actively opened."""
        pass

    @abstractmethod
    def get_diagnostics(self) -> CameraDiagnostics:
        """Retrieve operational statistics and health metrics."""
        pass


class OpenCVCameraSource(CameraSourceBase):
    """
    OpenCV-based camera capture supporting V4L2 USB cameras, CSI Camera Modules
    via GStreamer libcamerasrc pipelines, or offline test video files.
    """

    def __init__(self,
                 source: Union[int, str] = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 retry_sleep_sec: float = 0.05,
                 clock_fn: Callable[[], float] = time.monotonic):
        """
        Initialize OpenCV camera source.

        Args:
            source: Device index (e.g. 0), device path ('/dev/video0'),
                    GStreamer pipeline string, or video file path.
            width: Target frame width in pixels.
            height: Target frame height in pixels.
            fps: Target capture frame rate.
            retry_sleep_sec: Small sleep duration on read failure.
            clock_fn: Monotonic clock callable (injectable for testing).
        """
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.retry_sleep_sec = float(retry_sleep_sec)
        self.clock_fn = clock_fn

        self.cap: Optional[object] = None
        self._diagnostics = CameraDiagnostics(source_identifier=str(source))
        self._fps_window: List[float] = []

    def start(self) -> bool:
        """Open OpenCV VideoCapture with configured parameters."""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) is not available on this system.")

        if self.cap is not None and self.cap.isOpened():
            self._diagnostics.is_opened = True
            return True

        # Check if source is GStreamer pipeline string
        if isinstance(self.source, str) and ("!" in self.source or "libcamerasrc" in self.source):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self._diagnostics.is_opened = False
            return False

        # Apply properties (for standard USB / V4L2 devices)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._diagnostics.is_opened = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Acquire a frame from VideoCapture with diagnostic accounting.
        """
        now = self.clock_fn()

        if self.cap is None or not self.cap.isOpened():
            self._record_failure(now)
            return False, None, now

        ret, frame = self.cap.read()

        if not ret or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self._record_failure(now)
            if self.retry_sleep_sec > 0:
                time.sleep(self.retry_sleep_sec)
            return False, None, now

        # Successful read
        self._diagnostics.frames_read += 1
        self._diagnostics.consecutive_drops = 0
        self._diagnostics.failure_duration = 0.0
        self._diagnostics.last_failure_time = None
        self._diagnostics.last_frame_time = now

        # Update rolling FPS estimation
        self._fps_window.append(now)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        if len(self._fps_window) >= 2:
            elapsed = self._fps_window[-1] - self._fps_window[0]
            if elapsed > 0:
                self._diagnostics.effective_fps = (len(self._fps_window) - 1) / elapsed

        return True, frame, now

    def _record_failure(self, now: float) -> None:
        """Internal helper to update failure metrics."""
        self._diagnostics.frames_dropped += 1
        self._diagnostics.consecutive_drops += 1
        if self._diagnostics.last_failure_time is None:
            self._diagnostics.last_failure_time = now
        self._diagnostics.failure_duration = now - self._diagnostics.last_failure_time

    def reconnect(self) -> bool:
        """Attempt to re-initialize stalled camera device."""
        self.stop()
        return self.start()

    def stop(self) -> None:
        """Release the capture device."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._diagnostics.is_opened = False

    def is_opened(self) -> bool:
        """Return device open status."""
        return self.cap is not None and self.cap.isOpened()

    def get_diagnostics(self) -> CameraDiagnostics:
        """Return diagnostic metrics."""
        return self._diagnostics


class SyntheticCameraSource(CameraSourceBase):
    """
    Deterministic synthetic frame generator for offline tests, CI, and simulation.
    Generates synthetic downward track views without hardware cameras or video files.
    """

    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 line_color: Tuple[int, int, int] = (0, 0, 0),        # Dark line (BGR)
                 bg_color: Tuple[int, int, int] = (255, 255, 255),    # Light background (BGR)
                 line_thickness: int = 20,
                 clock_fn: Callable[[], float] = time.monotonic):
        """
        Initialize synthetic camera source.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            line_color: Line color in BGR.
            bg_color: Background color in BGR.
            line_thickness: Line stroke width in pixels.
            clock_fn: Injectable monotonic clock callable.
        """
        self.width = width
        self.height = height
        self.line_color = line_color
        self.bg_color = bg_color
        self.line_thickness = line_thickness
        self.clock_fn = clock_fn

        self._opened = False
        self._diagnostics = CameraDiagnostics(source_identifier="synthetic")
        self._pattern: str = "center"
        self._offset_px: int = 0
        self._custom_frame: Optional[np.ndarray] = None
        self._fail_next_n: int = 0
        self._frame_sequence: List[Tuple[bool, Optional[np.ndarray]]] = []

    def start(self) -> bool:
        """Open synthetic camera."""
        self._opened = True
        self._diagnostics.is_opened = True
        return True

    def stop(self) -> None:
        """Stop synthetic camera."""
        self._opened = False
        self._diagnostics.is_opened = False

    def is_opened(self) -> bool:
        return self._opened

    def get_diagnostics(self) -> CameraDiagnostics:
        return self._diagnostics

    # -------------------------------------------------------------------------
    # Test Pattern Configuration
    # -------------------------------------------------------------------------

    def set_pattern(self, pattern: str, offset_px: int = 0) -> None:
        """
        Set synthetic pattern type.
        Patterns: 'center', 'offset', 'curved', 'intersection', 'blank', 'dropped'.
        """
        self._pattern = pattern
        self._offset_px = offset_px
        self._custom_frame = None

    def set_custom_frame(self, frame: np.ndarray) -> None:
        """Inject an explicit NumPy frame."""
        self._custom_frame = frame

    def queue_failures(self, count: int) -> None:
        """Queue N consecutive failed reads (ret=False, frame=None)."""
        self._fail_next_n = count

    def queue_frame_sequence(self, sequence: List[Tuple[bool, Optional[np.ndarray]]]) -> None:
        """Queue a sequence of (success, frame) tuples to return consecutively."""
        self._frame_sequence = list(sequence)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Generate or dequeue next synthetic frame."""
        now = self.clock_fn()

        if not self._opened:
            self._record_failure(now)
            return False, None, now

        # 1. Check queued explicit sequence
        if self._frame_sequence:
            success, frame = self._frame_sequence.pop(0)
            if not success or frame is None:
                self._record_failure(now)
                return False, None, now
            self._record_success(now)
            return True, frame, now

        # 2. Check queued failure count
        if self._fail_next_n > 0:
            self._fail_next_n -= 1
            self._record_failure(now)
            return False, None, now

        # 3. Check pattern
        if self._pattern == "dropped":
            self._record_failure(now)
            return False, None, now

        # 4. Generate frame based on pattern
        frame = self._generate_pattern_frame()
        self._record_success(now)
        return True, frame, now

    def _record_success(self, now: float) -> None:
        self._diagnostics.frames_read += 1
        self._diagnostics.consecutive_drops = 0
        self._diagnostics.failure_duration = 0.0
        self._diagnostics.last_failure_time = None
        self._diagnostics.last_frame_time = now

    def _record_failure(self, now: float) -> None:
        self._diagnostics.frames_dropped += 1
        self._diagnostics.consecutive_drops += 1
        if self._diagnostics.last_failure_time is None:
            self._diagnostics.last_failure_time = now
        self._diagnostics.failure_duration = now - self._diagnostics.last_failure_time

    def _generate_pattern_frame(self) -> np.ndarray:
        """Render pattern onto a fresh frame."""
        if self._custom_frame is not None:
            return self._custom_frame.copy()

        frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        center_x = self.width // 2

        if self._pattern == "blank":
            return frame

        if not CV2_AVAILABLE:
            # Fallback for systems without cv2 drawing
            x = center_x + self._offset_px
            x1 = max(0, x - self.line_thickness // 2)
            x2 = min(self.width, x + self.line_thickness // 2)
            frame[:, x1:x2] = self.line_color
            return frame

        if self._pattern in ("center", "offset"):
            x = center_x + self._offset_px
            cv2.line(frame, (x, 0), (x, self.height), self.line_color, self.line_thickness)
        elif self._pattern == "curved":
            # Angled diagonal line from bottom to top
            x_bottom = center_x + self._offset_px
            x_top = center_x + self._offset_px + 80
            cv2.line(frame, (x_bottom, self.height), (x_top, 0), self.line_color, self.line_thickness)
        elif self._pattern == "intersection":
            # T-junction or cross
            cv2.line(frame, (center_x, 0), (center_x, self.height), self.line_color, self.line_thickness)
            mid_y = self.height // 2
            cv2.line(frame, (0, mid_y), (self.width, mid_y), self.line_color, self.line_thickness)

        return frame
