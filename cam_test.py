"""
Camera Test Script for the Line Following Drone.

Tests the Raspberry Pi Camera Module feed using GStreamer + libcamera.
Uses the centralized pipeline from config.py.

Usage:
    python cam_test.py

Press ESC to exit.
"""

import cv2
import sys

import config


def main():
    print("[CAM_TEST] Opening camera...")
    print(f"[CAM_TEST] Pipeline: {config.GSTREAMER_PIPELINE}")

    cap = cv2.VideoCapture(config.GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        # Fallback to default camera (useful for desktop testing)
        print("[CAM_TEST] GStreamer pipeline failed, trying default camera...")
        cap = cv2.VideoCapture(0)
        # Force resolution to match config
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[CAM_TEST] ERROR: Cannot open camera!")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM_TEST] Camera opened: {actual_w}x{actual_h} "
          f"@ {config.CAMERA_FPS}fps")
    print("[CAM_TEST] Press ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[CAM_TEST] WARNING: Frame not received, retrying...")
                continue

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) == 27:  # ESC key
                print("[CAM_TEST] ESC pressed — exiting.")
                break

    except KeyboardInterrupt:
        print("\n[CAM_TEST] Interrupted (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[CAM_TEST] Cleanup complete.")


if __name__ == "__main__":
    main()