"""
Line Detection Test Script for the Line Following Drone.

Tests the computer vision pipeline for detecting a dark line on a light
background. Uses the GStreamer camera pipeline from config.py (fixes the
original bug of using VideoCapture(0) which fails on Raspberry Pi).

Usage:
    python line_detection.py

Press ESC to exit.
"""

import cv2
import numpy as np
import sys

import config
from line_detector import LineDetector


def main():
    print("[LINE_DET] Opening camera...")
    cap = cv2.VideoCapture(config.GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        # Fallback to default camera (useful for desktop testing)
        print("[LINE_DET] GStreamer pipeline failed, trying default camera...")
        cap = cv2.VideoCapture(0)
        # Force resolution to match config
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[LINE_DET] ERROR: Cannot open camera!")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[LINE_DET] Camera opened: {actual_w}x{actual_h}")
    print("[LINE_DET] Press ESC to exit.")

    detector = LineDetector(
        threshold_value=config.THRESHOLD_VALUE,
        blur_kernel_size=config.BLUR_KERNEL_SIZE,
        min_contour_area=config.MIN_CONTOUR_AREA,
        invert_threshold=True
    )

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[LINE_DET] WARNING: Frame not received, retrying...")
                continue

            # Delegate to unified LineDetector
            result = detector.detect(frame, draw_debug=True)

            if result.detected:
                print(f"[LINE_DET] Error: {result.error:+4d}px | "
                      f"Centroid: ({result.cx}, {result.cy}) | "
                      f"Area: {result.contour_area:.0f}px²")
            else:
                print("[LINE_DET] No line detected")

            # --- Display ---
            display_frame = result.debug_frame if result.debug_frame is not None else frame
            display_thresh = result.binary_mask if result.binary_mask is not None else frame
            cv2.imshow("Line Detection - Camera", display_frame)
            cv2.imshow("Line Detection - Threshold", display_thresh)

            if cv2.waitKey(1) == 27:  # ESC key
                print("[LINE_DET] ESC pressed — exiting.")
                break

    except KeyboardInterrupt:
        print("\n[LINE_DET] Interrupted (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[LINE_DET] Cleanup complete.")


if __name__ == "__main__":
    main()