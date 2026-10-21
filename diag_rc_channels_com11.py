"""
diag_rc_channels_com11.py — Strictly Passive Read-Only RC Channel Diagnostics

Safety Guarantees:
  1. Serial port COM11 pre-configured with dtr=False and rts=False BEFORE opening.
     Guarantees no DTR/RTS assertion on Windows.
  2. ZERO bytes are transmitted over the serial interface.
     No MAVLink messages, stream requests, or parameter queries are sent.
  3. Targets COM11 exclusively. Does not access COM9 or any other port.
  4. Bounded duration: 30.0 seconds maximum.
  5. Serial cleanup: Every operation after ser.open() is enclosed in a try/finally
     block, guaranteeing ser.close() executes on normal termination, exceptions,
     and KeyboardInterrupt (Ctrl+C).
  6. Arming guard: If any HEARTBEAT reports ARMED status, halts immediately without
     sending any commands or attempting an automatic disarm.
  7. Passive packet parsing using pymavlink dialect parser with CRC validation.
"""

import sys
import time
import serial
from pymavlink.dialects.v20 import common as mavlink2


def main():
    port_name = "COM11"
    baud_rate = 57600
    listen_timeout = 30.0

    print("=" * 70)
    print("  PHASE 2A: PASSIVE RC TRANSMITTER & CHANNEL MAPPING DIAGNOSTIC")
    print(f"  Target Port    : {port_name}")
    print(f"  Baud Rate      : {baud_rate}")
    print(f"  Control Lines  : DTR=Disabled, RTS=Disabled (pre-open DCB config)")
    print(f"  Data Direction : STRICTLY READ-ONLY (Zero bytes transmitted)")
    print(f"  Max Duration   : {listen_timeout:.1f}s bounded window (Ctrl+C to stop early)")
    print("  PRECONDITION   : ALL PROPELLERS MUST BE REMOVED BEFORE TESTING.")
    print("=" * 70)

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

    # 2. Complete try/finally wrapping every operation after successful open
    heartbeat_count = 0
    bad_data_count = 0
    total_raw_bytes = 0
    all_message_counts = {}
    is_armed_detected = False

    rc_channels_received = False
    rc_source_msg = None
    channels_data = {
        ch: {
            'latest': None,
            'min': None,
            'max': None,
            'samples': 0
        } for ch in range(1, 19)
    }

    radio_status_messages = []
    start_time = time.time()
    elapsed = 0.0

    try:
        print(f"[STATUS] {port_name} successfully opened. dtr={ser.dtr}, rts={ser.rts}")
        print(f"[STATUS] Passively listening for up to {listen_timeout:.1f}s without sending data...")
        print("[NOTE] Moving transmitter sticks/switches will test if channel updates stream passively.\n")

        parser = mavlink2.MAVLink(None)
        parser.robust_parsing = True

        while time.time() - start_time < listen_timeout:
            # Read available bytes from OS buffer
            waiting = ser.in_waiting
            chunk_size = max(1, waiting)
            raw_chunk = ser.read(chunk_size)

            if not raw_chunk:
                continue

            total_raw_bytes += len(raw_chunk)

            # Parse through pymavlink parser (validates CRC16 & framing)
            messages = parser.parse_buffer(raw_chunk)
            if not messages:
                continue

            for msg in messages:
                msg_type = msg.get_type()
                all_message_counts[msg_type] = all_message_counts.get(msg_type, 0) + 1

                if msg_type == 'BAD_DATA':
                    bad_data_count += 1
                    continue

                # 1. HEARTBEAT Message (Msg ID 0)
                if msg_type == 'HEARTBEAT':
                    heartbeat_count += 1
                    is_armed = bool(msg.base_mode & mavlink2.MAV_MODE_FLAG_SAFETY_ARMED)
                    if is_armed:
                        is_armed_detected = True
                        print("\n[CRITICAL WARNING] DRONE REPORTS ARMED STATUS ON BENCH!")
                        print("[ACTION] Terminating diagnostic immediately. No commands will be sent.")
                        break

                # 2. RC_CHANNELS Message (Msg ID 65)
                elif msg_type == 'RC_CHANNELS':
                    rc_channels_received = True
                    rc_source_msg = 'RC_CHANNELS'
                    chan_count = getattr(msg, 'chancount', 18)
                    for ch in range(1, min(chan_count, 18) + 1):
                        val = getattr(msg, f'chan{ch}_raw', 0)
                        if val > 0:
                            cd = channels_data[ch]
                            cd['latest'] = val
                            cd['samples'] += 1
                            if cd['min'] is None or val < cd['min']:
                                cd['min'] = val
                            if cd['max'] is None or val > cd['max']:
                                cd['max'] = val

                # 3. RC_CHANNELS_RAW Message (Msg ID 35)
                elif msg_type == 'RC_CHANNELS_RAW':
                    rc_channels_received = True
                    if rc_source_msg is None:
                        rc_source_msg = 'RC_CHANNELS_RAW'
                    for ch in range(1, 9):
                        val = getattr(msg, f'chan{ch}_raw', 0)
                        if val > 0:
                            cd = channels_data[ch]
                            cd['latest'] = val
                            cd['samples'] += 1
                            if cd['min'] is None or val < cd['min']:
                                cd['min'] = val
                            if cd['max'] is None or val > cd['max']:
                                cd['max'] = val

                # 4. RADIO_STATUS Message (Msg ID 109)
                elif msg_type == 'RADIO_STATUS':
                    radio_status_messages.append({
                        'rssi': getattr(msg, 'rssi', 0),
                        'remrssi': getattr(msg, 'remrssi', 0),
                        'txbuf': getattr(msg, 'txbuf', 0),
                        'noise': getattr(msg, 'noise', 0),
                        'remnoise': getattr(msg, 'remnoise', 0),
                        'rxerrors': getattr(msg, 'rxerrors', 0),
                        'fixed': getattr(msg, 'fixed', 0)
                    })

            if is_armed_detected:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Diagnostic stopped early by user (Ctrl+C).")
    finally:
        elapsed = time.time() - start_time
        if ser.is_open:
            ser.close()
        print(f"[STATUS] {port_name} cleanly closed and resources released.")

    # 3. Comprehensive Diagnostic Reporting
    print("\n" + "=" * 70)
    print(f"  PHASE 2A DIAGNOSTIC RESULTS (Observed over {elapsed:.1f}s)")
    print("=" * 70)
    print(f"  Total Raw Bytes Received  : {total_raw_bytes}")
    print(f"  Heartbeats Received       : {heartbeat_count}")
    print(f"  Corrupted Packets (Bad)   : {bad_data_count}")
    print(f"  All Message Types Seen    : {dict(sorted(all_message_counts.items()))}")

    if is_armed_detected:
        print("\n  [SAFETY INTERLOCK TRIGGERED]")
        print("  The diagnostic was halted early because an ARMED heartbeat was detected.")
        print("  Motors may be active. Do not touch the aircraft.")
        return

    # Heartbeat Reception Rate
    if heartbeat_count > 0:
        rate = heartbeat_count / max(1.0, elapsed)
        print(f"  Heartbeat Rate (Approx)   : ~{rate:.1f} Hz ({heartbeat_count} messages over {elapsed:.1f}s)")
    else:
        print("  Heartbeat Rate (Approx)   : 0 Hz (No heartbeats received)")

    # Radio Status Report
    print("\n  --- TELEMETRY RADIO LINK (RADIO_STATUS) ---")
    if radio_status_messages:
        latest_radio = radio_status_messages[-1]
        print(f"  RADIO_STATUS Packets Seen : {len(radio_status_messages)}")
        print(f"  Local RSSI / Remote RSSI  : {latest_radio['rssi']} / {latest_radio['remrssi']}")
        print(f"  Local Noise / Remote Noise: {latest_radio['noise']} / {latest_radio['remnoise']}")
        print(f"  Rx Errors / Fixed Packets : {latest_radio['rxerrors']} / {latest_radio['fixed']}")
        print(f"  Tx Buffer Free            : {latest_radio['txbuf']}%")
    else:
        print("  RADIO_STATUS Status       : Not observed during passive listening")

    # RC Channel Report
    print("\n  --- RC CHANNELS TELEMETRY ---")
    if not rc_channels_received:
        print("  RC Channels Status        : NOT OBSERVED during passive listening")
        print("  Details                   : Neither RC_CHANNELS (ID 65) nor RC_CHANNELS_RAW (ID 35)")
        print(f"                              were received in this {elapsed:.1f}s window.")
        print("  Note                      : This diagnostic was strictly passive and did not send")
        print("                              stream requests (REQUEST_DATA_STREAM / SET_MESSAGE_INTERVAL).")
    else:
        print(f"  Source Message Type       : {rc_source_msg}")
        print("  CH  | Latest PWM | Min PWM | Max PWM | Delta | Samples | Status")
        print("  ----+------------+---------+---------+-------+---------+------------------")
        active_channels = 0
        for ch in range(1, 19):
            cd = channels_data[ch]
            if cd['samples'] > 0:
                active_channels += 1
                delta = cd['max'] - cd['min']
                has_moved = "MOVED / ACTIVE" if delta > 20 else "STATIC"
                print(f"  {ch:2d}  |   {cd['latest']:4d} µs  | {cd['min']:4d} µs  | {cd['max']:4d} µs  |  {delta:4d} |  {cd['samples']:5d}  | {has_moved}")
        if active_channels == 0:
            print("  All channels reported zero or inactive.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
