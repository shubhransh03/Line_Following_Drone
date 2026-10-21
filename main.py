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
from line_detector import LineDetector


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

# Global detector instance configured from centralized parameters
_line_detector = LineDetector(
    threshold_value=config.THRESHOLD_VALUE,
    blur_kernel_size=config.BLUR_KERNEL_SIZE,
    min_contour_area=config.MIN_CONTOUR_AREA,
    invert_threshold=True
)


def detect_line(frame):
    """
    Detect the line in a camera frame and compute lateral error.
    Delegates to the unified LineDetector implementation.

    Args:
        frame: BGR image from camera

    Returns:
        Tuple of (error, cx, cy, thresh) where:
          - error: lateral pixel offset from center (positive = line is right)
          - cx, cy: centroid coordinates of detected line
          - thresh: thresholded binary image for debug display
        Returns (None, None, None, thresh) if no line is detected.
    """
    return _line_detector.detect(frame).as_tuple()


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


def _get_battery_failure_cause(fc):
    """Determine specific battery failure cause and log message from FlightController."""
    batt_mon = getattr(fc, 'battery_monitor', None)
    state = getattr(batt_mon, 'state', None)
    if state == "STALE":
        return "STALE_BATTERY", "[MAIN] CRITICAL: Battery telemetry stale / link lost! Emergency landing!"
    elif state == "UNKNOWN":
        return "UNKNOWN_BATTERY", "[MAIN] CRITICAL: Battery telemetry uninitialized! Emergency landing!"
    return "LOW_BATTERY", "[MAIN] CRITICAL: Low battery! Emergency landing!"


# ==============================================================================
# Flight Control Loop Execution
# ==============================================================================

def run_flight_loop(cap, detector, pid, fc=None, fly_mode=False,
                    clock_fn=time.monotonic, sleep_fn=time.sleep,
                    max_iterations=None, exit_on_override=True):
    """
    Core line-following control loop with bounded camera-failure handling.

    Args:
        cap: Video capture object or mock supporting .read().
        detector: LineDetector instance.
        pid: PIDController instance.
        fc: FlightController instance (or None/mock).
        fly_mode: Whether live velocity commands and flight safety checks are active.
        clock_fn: Callable returning monotonic time in seconds (injectable for testing).
        sleep_fn: Callable for sleep/pacing (injectable for testing).
        max_iterations: Bounded iteration count for tests (None = run until shutdown).
        exit_on_override: If True, calls sys.exit(0) on pilot override; if False, returns.

    Returns:
        Dict summarizing loop execution (loop_count, consecutive_frame_failures, shutdown_cause).
    """
    line_lost_time = None
    last_valid_frame_time = clock_fn()
    consecutive_frame_failures = 0
    loop_count = 0
    shutdown_cause = "NORMAL"

    print("[MAIN] Starting line following loop...")
    print("[MAIN] Press ESC to stop (or Ctrl+C)")

    while not _shutdown_requested:
        if max_iterations is not None and loop_count >= max_iterations:
            shutdown_cause = "MAX_ITERATIONS"
            break

        current_time = clock_fn()

        # 1. Capture frame
        ret, frame = cap.read()

        # Handle frame drop / camera read failure
        if not ret or frame is None:
            consecutive_frame_failures += 1
            elapsed_no_frame = current_time - last_valid_frame_time

            if consecutive_frame_failures == 1 or consecutive_frame_failures % config.CAMERA_MAX_CONSECUTIVE_FAILURES == 0:
                print(f"[MAIN] WARNING: Camera frame read failed (consecutive: {consecutive_frame_failures}, "
                      f"elapsed: {elapsed_no_frame:.2f}s)")

            # Do not run line detection on invalid frame!
            # Reset PID controller to discard any stale derivative/integral state
            pid.reset()

            # Prevent 100% CPU spinning during camera dropout
            if config.CAMERA_RETRY_SLEEP > 0:
                sleep_fn(config.CAMERA_RETRY_SLEEP)

            # Stage 1 Camera Failure: Bounded hover response once threshold is reached
            if elapsed_no_frame >= config.CAMERA_FRAME_TIMEOUT:
                if fly_mode and fc:
                    fc.send_body_velocity(vx=0, vy=0, vz=0)

            # Stage 2 Camera Failure: Persistent fault timeout -> trigger landing
            if elapsed_no_frame >= config.CAMERA_FAULT_TIMEOUT:
                print(f"[MAIN] CRITICAL: Camera feed lost for {elapsed_no_frame:.2f}s "
                      f"(exceeds {config.CAMERA_FAULT_TIMEOUT}s timeout). Aborting flight!")
                shutdown_cause = "CAMERA_TIMEOUT"
                break

            # Crucial: Run safety checks on every failed frame!
            if fly_mode and fc:
                if not fc.check_battery():
                    shutdown_cause, msg = _get_battery_failure_cause(fc)
                    print(f"{msg} (during camera failure)")
                    break

                if fc.check_rc_override() or fc.is_mode_changed():
                    print("[MAIN] Pilot override detected during camera failure! Releasing control.")
                    shutdown_cause = "RC_OVERRIDE"
                    if exit_on_override:
                        fc.close()
                        cap.release()
                        if config.SHOW_DEBUG_FRAMES and cv2:
                            cv2.destroyAllWindows()
                        sys.exit(0)
                    else:
                        break

            loop_count += 1
            continue

        # Valid frame arrived -> Reset camera failure counters
        consecutive_frame_failures = 0
        last_valid_frame_time = current_time

        # 2. Vision processing with LineDetector
        result = detector.detect(frame, draw_debug=config.SHOW_DEBUG_FRAMES)
        line_detected = result.detected
        pid_output = 0.0

        if line_detected:
            line_lost_time = None
            pid_output = pid.compute(result.normalized_error)

            if fly_mode and fc:
                fc.send_body_velocity(
                    vx=config.FORWARD_SPEED,
                    vy=pid_output,
                    vz=0
                )
        else:
            # Line lost on valid frame
            if line_lost_time is None:
                line_lost_time = current_time
                pid.reset()
                print("[MAIN] Line lost! Hovering...")

            lost_duration = current_time - line_lost_time

            if fly_mode and fc:
                fc.send_body_velocity(vx=0, vy=0, vz=0)

            if lost_duration > config.LINE_LOST_TIMEOUT:
                print(f"[MAIN] Line lost for {lost_duration:.1f}s — "
                      f"exceeds {config.LINE_LOST_TIMEOUT}s timeout!")
                shutdown_cause = "LINE_LOST_TIMEOUT"
                break

        # 3. Safety checks (battery, RC override, mode changed)
        if fly_mode and fc:
            if loop_count % 5 == 0:
                if not fc.check_battery():
                    shutdown_cause, msg = _get_battery_failure_cause(fc)
                    print(msg)
                    break

            if fc.check_rc_override() or fc.is_mode_changed():
                print("[MAIN] Pilot override detected! Releasing control.")
                shutdown_cause = "RC_OVERRIDE"
                if exit_on_override:
                    fc.close()
                    cap.release()
                    if config.SHOW_DEBUG_FRAMES and cv2:
                        cv2.destroyAllWindows()
                    sys.exit(0)
                else:
                    break

        # 4. Debug display
        if config.SHOW_DEBUG_FRAMES and cv2 and result.debug_frame is not None:
            cv2.imshow("Line Following - Camera", result.debug_frame)
            if result.binary_mask is not None:
                cv2.imshow("Line Following - Threshold", result.binary_mask)

            if cv2.waitKey(1) == 27:
                print("[MAIN] ESC pressed — stopping.")
                shutdown_cause = "USER_ESC"
                break

        loop_count += 1

    return {
        "loop_count": loop_count,
        "consecutive_frame_failures": consecutive_frame_failures,
        "shutdown_cause": shutdown_cause
    }


# ==============================================================================
# Main CLI Entry Point
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
            status = getattr(fc, 'last_takeoff_status', 'UNKNOWN')
            print(f"[MAIN] ERROR: Takeoff failed with status: {status}!")
            if status == "TIMEOUT":
                print("[MAIN] CRITICAL WARNING: Takeoff timed out after arming! Vehicle may be airborne in GUIDED mode.")
                print("[MAIN] Telemetry connection is closing; vehicle safety depends on physical RC takeover or ArduPilot failsafe.")
            fc.close()
            cap.release()
            sys.exit(1)

        # Brief hover to stabilize before starting line following
        print("[MAIN] Hovering for 2 seconds to stabilize...")
        time.sleep(2)

    try:
        run_flight_loop(
            cap=cap,
            detector=_line_detector,
            pid=pid,
            fc=fc,
            fly_mode=fly_mode,
            exit_on_override=True
        )
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
                landed = fc.land()
                if not landed:
                    print("[MAIN] WARNING: Landing did not confirm disarm within timeout!")
            except Exception as e:
                print(f"[MAIN] Landing error: {e}")
                print("[MAIN] Attempting RTL...")
                try:
                    fc.return_to_launch()
                except Exception:
                    pass
            fc.close()

        if cap is not None:
            cap.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
        print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()
