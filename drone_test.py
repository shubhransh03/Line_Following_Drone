"""
Drone Connection Test Script for the Line Following Drone.

Tests the serial connection to the Pixhawk flight controller via DroneKit
and prints extended telemetry. Uses centralized settings from config.py.

Usage:
    python drone_test.py
"""

import sys

from dronekit import connect

import config


def main():
    vehicle = None

    try:
        print("[DRONE_TEST] Connecting to Pixhawk...")
        print(f"[DRONE_TEST] Port: {config.SERIAL_PORT}")
        print(f"[DRONE_TEST] Baud: {config.BAUD_RATE}")

        vehicle = connect(config.SERIAL_PORT, baud=config.BAUD_RATE,
                          wait_ready=True, heartbeat_timeout=config.CONNECT_TIMEOUT)

        print("[DRONE_TEST] Connected successfully!")
        print()
        print("=" * 40)
        print("  VEHICLE TELEMETRY")
        print("=" * 40)
        print(f"  Battery  : {vehicle.battery}")
        print(f"  GPS      : {vehicle.gps_0}")
        print(f"  Mode     : {vehicle.mode.name}")
        print(f"  Armed    : {vehicle.armed}")
        print(f"  Heading  : {vehicle.heading}°")

        if vehicle.location.global_relative_frame:
            alt = vehicle.location.global_relative_frame.alt
            print(f"  Altitude : {alt:.1f} m")

        print(f"  EKF OK   : {vehicle.ekf_ok}")
        print(f"  System   : {vehicle.system_status.state}")
        print(f"  Firmware : {vehicle.version}")
        print("=" * 40)

    except KeyboardInterrupt:
        print("\n[DRONE_TEST] Interrupted (Ctrl+C).")

    except Exception as e:
        print(f"[DRONE_TEST] ERROR: {e}")
        sys.exit(1)

    finally:
        if vehicle is not None:
            vehicle.close()
            print("[DRONE_TEST] Connection closed.")


if __name__ == "__main__":
    main()