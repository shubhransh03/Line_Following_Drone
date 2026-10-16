"""
Main script for the Line Following Drone.

This is the integrated autonomous flight script that combines:
  1. Camera capture (GStreamer + libcamera)
  2. Line detection (OpenCV)
  3. PID control (error → velocity correction)
  4. Flight control (DroneKit velocity commands)
  5. Safety monitoring (battery, RC override, line-lost timeout)

Usage:
    python main.py              # Normal autonomous flight
    python main.py --no-fly     # Vision-only mode (no drone, for testing)
"""

import argparse
import cv2
import numpy as np
import time
import sys
import signal

import config
from pid_controller import PIDController
from flight_controller import FlightController


# ==============================================================================
# Global state for clean shutdown
# ==============================================================================

_shutdown_requested = False


def signal_handler(sig, frame):
    """Handle Ctrl+C for clean shutdown."""
    global _shutdown_requested
    print("\n[MAIN] Shutdown requested (Ctrl+C)...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)


# ==============================================================================
# Vision Processing
# ==============================================================================

def detect_line(frame):
    """
    Detect the line in a camera frame and compute lateral error.

    Args:
        frame: BGR image from camera

    Returns:
        Tuple of (error, cx, cy, thresh) where:
          - error: lateral pixel offset from center (positive = line is right)
          - cx, cy: centroid coordinates of detected line
          - thresh: thresholded binary image for debug display
        Returns (None, None, None, thresh) if no line is detected.
    """
    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,
                            (config.BLUR_KERNEL_SIZE, config.BLUR_KERNEL_SIZE),
                            0)

    # Binary threshold — dark line becomes white on black background
    _, thresh = cv2.threshold(blur, config.THRESHOLD_VALUE, 255,
                              cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        # Filter by minimum area to reject noise
        valid_contours = [c for c in contours
                          if cv2.contourArea(c) >= config.MIN_CONTOUR_AREA]

        if valid_contours:
            # Take the largest valid contour (assumed to be the line)
            c = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(c)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Error = how far the line centroid is from frame center
                frame_center_x = frame.shape[1] // 2
                error = cx - frame_center_x

                return error, cx, cy, thresh

    return None, None, None, thresh


def draw_debug_overlay(frame, thresh, error, cx, cy, pid_output, line_detected):
    """
    Draw debug information on the frame for visualization.

    Args:
        frame: BGR image to draw on (modified in place)
        thresh: Thresholded image to display
        error: Current lateral error (or None)
        cx, cy: Centroid coordinates (or None)
        pid_output: Current PID output value
        line_detected: Whether the line is currently detected
    """
    h, w = frame.shape[:2]
    center_x = w // 2

    # Draw center reference line
    cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 0), 1)

    if line_detected and cx is not None and cy is not None:
        # Draw centroid
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

        # Draw error line from center to centroid
        cv2.line(frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)

        # Text overlay
        cv2.putText(frame, f"Error: {error:+d}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"PID: {pid_output:+.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "LINE LOST", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Status bar
    status = "TRACKING" if line_detected else "SEARCHING"
    color = (0, 255, 0) if line_detected else (0, 0, 255)
    cv2.putText(frame, status, (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ==============================================================================
# Main Flight Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Line Following Drone")
    parser.add_argument('--no-fly', action='store_true',
                        help='Vision-only mode (no drone connection)')
    args = parser.parse_args()

    fly_mode = not args.no_fly

    print("=" * 60)
    print("  LINE FOLLOWING DRONE")
    print(f"  Mode: {'AUTONOMOUS FLIGHT' if fly_mode else 'VISION ONLY (no-fly)'}")
    print("=" * 60)

    # --- Initialize Camera ---
    print("[MAIN] Initializing camera...")
    cap = cv2.VideoCapture(config.GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        # Fallback to default camera (for testing on desktop)
        print("[MAIN] GStreamer pipeline failed, trying default camera...")
        cap = cv2.VideoCapture(0)
        # Force resolution to match config
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[MAIN] ERROR: Cannot open camera!")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[MAIN] Camera opened: {actual_w}x{actual_h} "
          f"@ {config.CAMERA_FPS}fps")

    # --- Initialize PID Controller ---
    pid = PIDController(
        kp=config.PID_KP,
        ki=config.PID_KI,
        kd=config.PID_KD,
        integral_limit=config.PID_INTEGRAL_LIMIT,
        output_limit=config.MAX_LATERAL_SPEED
    )
    print(f"[MAIN] PID initialized: {pid}")

    # --- Initialize Flight Controller ---
    fc = None
    if fly_mode:
        fc = FlightController()
        if not fc.connect():
            print("[MAIN] ERROR: Cannot connect to Pixhawk!")
            cap.release()
            sys.exit(1)

        print("[MAIN] Starting arm and takeoff sequence...")
        if not fc.arm_and_takeoff():
            print("[MAIN] ERROR: Takeoff failed!")
            fc.close()
            cap.release()
            sys.exit(1)

        # Brief hover to stabilize before starting line following
        print("[MAIN] Hovering for 2 seconds to stabilize...")
        time.sleep(2)

    # --- Main Control Loop ---
    print("[MAIN] Starting line following loop...")
    print("[MAIN] Press ESC to stop (or Ctrl+C)")

    line_lost_time = None
    loop_count = 0

    try:
        while not _shutdown_requested:
            ret, frame = cap.read()

            if not ret:
                print("[MAIN] WARNING: Frame not received, retrying...")
                continue

            # --- Vision Processing ---
            error, cx, cy, thresh = detect_line(frame)

            line_detected = error is not None
            pid_output = 0.0

            if line_detected:
                # Line found — reset lost timer
                line_lost_time = None

                # Normalize error to roughly [-1, 1] range
                # Use actual frame width (not config) to handle resolution differences
                frame_half_width = frame.shape[1] / 2
                normalized_error = error / frame_half_width

                # Compute PID correction
                pid_output = pid.compute(normalized_error)

                # Send velocity command to drone
                if fly_mode and fc:
                    fc.send_body_velocity(
                        vx=config.FORWARD_SPEED,   # Move forward
                        vy=pid_output,              # Lateral correction
                        vz=0                        # Maintain altitude
                    )

            else:
                # Line lost
                if line_lost_time is None:
                    line_lost_time = time.time()
                    pid.reset()
                    print("[MAIN] Line lost! Hovering...")

                lost_duration = time.time() - line_lost_time

                if fly_mode and fc:
                    # Hover in place while searching
                    fc.send_body_velocity(vx=0, vy=0, vz=0)

                # Check if line has been lost too long
                if lost_duration > config.LINE_LOST_TIMEOUT:
                    print(f"[MAIN] Line lost for {lost_duration:.1f}s — "
                          f"exceeds {config.LINE_LOST_TIMEOUT}s timeout!")
                    break

            # --- Safety Checks (every 10th frame to reduce overhead) ---
            if fly_mode and fc and loop_count % 10 == 0:
                if not fc.check_battery():
                    print("[MAIN] CRITICAL: Low battery! Emergency landing!")
                    break

                if fc.check_rc_override() or fc.is_mode_changed():
                    print("[MAIN] Pilot override detected! Releasing control.")
                    # Don't land automatically — pilot has control
                    fc.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

            # --- Debug Display ---
            if config.SHOW_DEBUG_FRAMES:
                draw_debug_overlay(frame, thresh, error, cx, cy,
                                   pid_output, line_detected)
                cv2.imshow("Line Following - Camera", frame)
                cv2.imshow("Line Following - Threshold", thresh)

                if cv2.waitKey(1) == 27:  # ESC key
                    print("[MAIN] ESC pressed — stopping.")
                    break

            loop_count += 1

    except Exception as e:
        print(f"[MAIN] ERROR in main loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Clean Shutdown ---
        print("[MAIN] Shutting down...")

        if fly_mode and fc:
            print("[MAIN] Initiating landing...")
            try:
                fc.land()
            except Exception as e:
                print(f"[MAIN] Landing error: {e}")
                print("[MAIN] Attempting RTL...")
                try:
                    fc.return_to_launch()
                except Exception:
                    pass
            fc.close()

        cap.release()
        cv2.destroyAllWindows()
        print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()
