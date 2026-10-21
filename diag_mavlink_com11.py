"""
diag_mavlink_com11.py — Strictly Passive Read-Only MAVLink Telemetry Diagnostic

Safety Guarantees:
  1. Serial port (COM11) is pre-configured with dtr=False and rts=False BEFORE opening.
     This ensures neither DTR nor RTS control lines are asserted by pyserial on Windows.
  2. ZERO bytes are transmitted. The serial handle only executes read(). No write/send methods exist.
  3. Uses pymavlink MAVLink dialect parser to rigorously validate packet framing and CRC checksums.
  4. Bounded execution window (6.0 seconds), cleanly closing the serial port in finally block.
  5. Strictly targets COM11. Never touches COM9.
  6. Checks arming state: if armed, halts immediately without sending any commands.
"""

import sys
import time
import serial
from pymavlink.dialects.v20 import common as mavlink2


def main():
    port_name = "COM11"
    baud_rate = 57600
    listen_timeout = 6.0

    print("=" * 65)
    print("  PHASE 1B: FORMAL READ-ONLY MAVLINK VALIDATION")
    print(f"  Target Port    : {port_name}")
    print(f"  Baud Rate      : {baud_rate}")
    print(f"  Control Lines  : DTR=Disabled, RTS=Disabled (pre-open DCB config)")
    print(f"  Data Direction : STRICTLY READ-ONLY (Zero bytes transmitted)")
    print(f"  Duration       : {listen_timeout:.1f}s bounded listening window")
    print("=" * 65)

    # 1. Pre-configure serial port BEFORE opening to prevent DTR/RTS assertions
    ser = serial.Serial()
    ser.port = port_name
    ser.baudrate = baud_rate
    ser.timeout = 0.5
    ser.dsrdtr = False
    ser.rtscts = False
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
    except Exception as e:
        print(f"\n[FAIL] Unable to open {port_name}: {e}")
        sys.exit(1)

    print(f"[STATUS] {port_name} successfully opened. dtr={ser.dtr}, rts={ser.rts}")
    print(f"[STATUS] Passively capturing incoming MAVLink 2.0 frames for {listen_timeout}s...\n")

    # 2. Instantiate pymavlink dialect parser (does not own or write to any stream)
    parser = mavlink2.MAVLink(None)
    parser.robust_parsing = True

    heartbeats = []
    battery_voltages = []
    flight_modes = set()
    total_raw_bytes = 0
    total_valid_packets = 0
    bad_data_count = 0

    start_time = time.time()
    try:
        while time.time() - start_time < listen_timeout:
            # Read whatever bytes are available in the OS buffer
            waiting = ser.in_waiting
            chunk_size = max(1, waiting)
            raw_chunk = ser.read(chunk_size)

            if not raw_chunk:
                continue

            total_raw_bytes += len(raw_chunk)

            # Parse through pymavlink parser (validates CRC16-MCRF4XX & header)
            messages = parser.parse_buffer(raw_chunk)
            if messages:
                for msg in messages:
                    msg_type = msg.get_type()

                    if msg_type == 'BAD_DATA':
                        bad_data_count += 1
                        continue

                    total_valid_packets += 1

                    # 1. HEARTBEAT Message (Msg ID 0)
                    if msg_type == 'HEARTBEAT':
                        heartbeats.append(msg)
                        custom_mode = getattr(msg, 'custom_mode', None)
                        if custom_mode is not None:
                            flight_modes.add(custom_mode)

                        # Check arming flag immediately
                        is_armed = bool(msg.base_mode & mavlink2.MAV_MODE_FLAG_SAFETY_ARMED)
                        if is_armed:
                            print("\n[CRITICAL WARNING] DRONE REPORTS ARMED STATUS ON BENCH!")
                            print("Halting diagnostic immediately. Remove power or disarm via RC transmitter.")
                            return

                    # 2. SYS_STATUS Message (Msg ID 1) for battery telemetry
                    elif msg_type == 'SYS_STATUS':
                        v_mv = getattr(msg, 'voltage_battery', None)
                        if v_mv is not None and v_mv > 0:
                            battery_voltages.append(v_mv / 1000.0)

    except KeyboardInterrupt:
        print("\n[INFO] Diagnostic interrupted by user.")
    finally:
        ser.close()
        print(f"[STATUS] {port_name} cleanly closed and resources released.")

    # 3. Report Results
    print("\n" + "=" * 65)
    print("  PHASE 1B DIAGNOSTIC RESULTS")
    print("=" * 65)
    print(f"  Total Bytes Received        : {total_raw_bytes}")
    print(f"  Valid MAVLink Packets       : {total_valid_packets}")
    print(f"  Corrupted / Bad Packets     : {bad_data_count}")
    print(f"  Validated Heartbeats Count  : {len(heartbeats)}")

    if not heartbeats:
        print("\n[RESULT] FAIL: No validated MAVLink HEARTBEAT received.")
        print("Interpretation: Link lost, baud rate mismatch, or flight controller silent.")
        return

    # Decode details from verified heartbeats
    hb = heartbeats[-1]  # latest heartbeat

    autopilot_enum = mavlink2.enums['MAV_AUTOPILOT'].get(hb.autopilot, None)
    autopilot_str = autopilot_enum.name if autopilot_enum else f"Unknown ({hb.autopilot})"

    vehicle_enum = mavlink2.enums['MAV_TYPE'].get(hb.type, None)
    vehicle_str = vehicle_enum.name if vehicle_enum else f"Unknown ({hb.type})"

    is_armed = bool(hb.base_mode & mavlink2.MAV_MODE_FLAG_SAFETY_ARMED)
    status_enum = mavlink2.enums['MAV_STATE'].get(hb.system_status, None)
    status_str = status_enum.name if status_enum else f"Unknown ({hb.system_status})"

    print(f"\n  [VERIFIED FLIGHT CONTROLLER TELEMETRY]")
    print(f"  • Autopilot Type       : {autopilot_str} (ID: {hb.autopilot})")
    print(f"  • Vehicle Type         : {vehicle_str} (ID: {hb.type})")
    print(f"  • System ID            : {hb.get_srcSystem()}")
    print(f"  • Component ID         : {hb.get_srcComponent()}")
    print(f"  • Armed Status         : {'ARMED (DANGER)' if is_armed else 'DISARMED (SAFE)'}")
    print(f"  • System State         : {status_str}")
    print(f"  • Base Mode Flags      : 0x{hb.base_mode:02X}")
    print(f"  • Custom Flight Mode ID: {list(flight_modes)}")

    if battery_voltages:
        avg_v = sum(battery_voltages) / len(battery_voltages)
        print(f"  • Battery Voltage      : {avg_v:.2f} V ({len(battery_voltages)} samples from SYS_STATUS)")
    else:
        print(f"  • Battery Voltage      : No SYS_STATUS message received during window")

    print("=" * 65)


if __name__ == '__main__':
    main()
