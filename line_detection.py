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

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[LINE_DET] WARNING: Frame not received, retrying...")
                continue

            # --- Vision Pipeline ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,
                                    (config.BLUR_KERNEL_SIZE,
                                     config.BLUR_KERNEL_SIZE), 0)

            _, thresh = cv2.threshold(blur, config.THRESHOLD_VALUE, 255,
                                      cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)

            # Filter contours by minimum area to ignore noise
            if len(contours) > 0:
                valid_contours = [c for c in contours
                                  if cv2.contourArea(c) >= config.MIN_CONTOUR_AREA]

                if valid_contours:
                    c = max(valid_contours, key=cv2.contourArea)
                    M = cv2.moments(c)

                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # Draw centroid
                        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

                        # Draw center reference line
                        center_x = frame.shape[1] // 2
                        cv2.line(frame, (center_x, 0),
                                 (center_x, frame.shape[0]),
                                 (0, 255, 0), 1)

                        # Draw error line
                        cv2.line(frame, (center_x, cy), (cx, cy),
                                 (255, 0, 0), 2)

                        error = cx - center_x
                        cv2.putText(frame, f"Error: {error:+d}px", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 255), 2)

                        print(f"[LINE_DET] Error: {error:+4d}px | "
                              f"Centroid: ({cx}, {cy}) | "
                              f"Area: {cv2.contourArea(c):.0f}px²")
                else:
                    cv2.putText(frame, "No valid contour (too small)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No line detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

            # --- Display ---
            cv2.imshow("Line Detection - Camera", frame)
            cv2.imshow("Line Detection - Threshold", thresh)

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