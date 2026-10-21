"""
battery_monitor.py — Robust Battery Telemetry Monitoring & State Machine

Provides health evaluation, rolling noise filtering, invalid/stale telemetry rejection,
and hysteresis recovery for UAV LiPo battery monitoring.
"""

import math
import time
import collections
import config


class BatteryState:
    """Possible health states of the vehicle battery telemetry."""
    UNKNOWN = "UNKNOWN"  # No valid readings received yet
    NORMAL = "NORMAL"    # Voltage within safe operating range
    LOW = "LOW"          # Sustained voltage below safe threshold
    STALE = "STALE"      # Telemetry timed out / no fresh updates received


class BatteryMonitor:
    """
    Robust battery telemetry monitor.

    Features:
      - State machine (UNKNOWN, NORMAL, LOW, STALE).
      - Rejection of non-finite (NaN/Inf), negative, disconnected, or out-of-physical-range voltages.
      - Consecutive low reading filtering (prevents transient motor sag from triggering abort).
      - Hysteresis recovery margin (prevents oscillation at the threshold boundary).
      - Rolling window average calculation.
      - Injectable clock for deterministic, fast offline testing.
    """

    def __init__(self,
                 low_threshold=None,
                 consecutive_low_required=None,
                 stale_timeout=None,
                 min_valid_voltage=None,
                 max_valid_voltage=None,
                 hysteresis=None,
                 recovery_count=None,
                 window_size=5,
                 clock_fn=time.monotonic):
        """
        Initialize the battery monitor.

        Args:
            low_threshold: Voltage below which battery is considered low (default: config.LOW_BATTERY_VOLTAGE)
            consecutive_low_required: Consecutive low samples needed to trigger LOW state (default: config.BATTERY_LOW_CONSECUTIVE_COUNT)
            stale_timeout: Maximum elapsed seconds without update before marked STALE (default: config.BATTERY_STALE_TIMEOUT)
            min_valid_voltage: Lower sanity bound in volts (default: config.BATTERY_MIN_VALID_VOLTAGE)
            max_valid_voltage: Upper sanity bound in volts (default: config.BATTERY_MAX_VALID_VOLTAGE)
            hysteresis: Extra voltage required above low_threshold to recover to NORMAL (default: config.BATTERY_HYSTERESIS_VOLTAGE)
            recovery_count: Consecutive healthy samples required to clear LOW state (default: config.BATTERY_RECOVERY_COUNT)
            window_size: Sample count for rolling window average (default: 5)
            clock_fn: Monotonic time callable
        """
        self.low_threshold = (low_threshold if low_threshold is not None
                              else config.LOW_BATTERY_VOLTAGE)
        self.consecutive_low_required = (consecutive_low_required if consecutive_low_required is not None
                                         else config.BATTERY_LOW_CONSECUTIVE_COUNT)
        self.stale_timeout = (stale_timeout if stale_timeout is not None
                              else config.BATTERY_STALE_TIMEOUT)
        self.min_valid_voltage = (min_valid_voltage if min_valid_voltage is not None
                                  else config.BATTERY_MIN_VALID_VOLTAGE)
        self.max_valid_voltage = (max_valid_voltage if max_valid_voltage is not None
                                  else config.BATTERY_MAX_VALID_VOLTAGE)
        self.hysteresis = (hysteresis if hysteresis is not None
                           else config.BATTERY_HYSTERESIS_VOLTAGE)
        self.recovery_count = (recovery_count if recovery_count is not None
                               else config.BATTERY_RECOVERY_COUNT)
        self.window_size = window_size
        self.clock_fn = clock_fn

        self._state = BatteryState.UNKNOWN
        self._last_voltage = None
        self._last_valid_time = None
        self._consecutive_low = 0
        self._consecutive_recovery = 0
        self._history = collections.deque(maxlen=window_size)
        self.rejected_readings_count = 0
        self.total_readings_count = 0
        self.last_rejection_reason = None

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_voltage(self, voltage):
        """
        Check if a raw voltage reading is physically plausible and finite.

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        if voltage is None:
            return False, "Voltage is None"

        if not isinstance(voltage, (int, float)):
            return False, f"Non-numeric voltage type: {type(voltage).__name__}"

        if math.isnan(voltage):
            return False, "Voltage is NaN"

        if math.isinf(voltage):
            return False, "Voltage is infinite"

        if voltage < self.min_valid_voltage:
            return False, (f"Voltage {voltage:.2f}V is below physical minimum "
                           f"{self.min_valid_voltage:.1f}V (disconnected/ground fault)")

        if voltage > self.max_valid_voltage:
            return False, (f"Voltage {voltage:.2f}V is above physical maximum "
                           f"{self.max_valid_voltage:.1f}V (sensor spike/misconfigured)")

        return True, ""

    # -------------------------------------------------------------------------
    # State Machine Update
    # -------------------------------------------------------------------------

    def update(self, voltage, timestamp=None):
        """
        Process a new voltage reading and update monitor state.

        Args:
            voltage: Raw voltage reading from telemetry
            timestamp: Monotonic timestamp (uses clock_fn if None)

        Returns:
            Current BatteryState
        """
        now = timestamp if timestamp is not None else self.clock_fn()
        self.total_readings_count += 1

        is_valid, reason = self.validate_voltage(voltage)
        if not is_valid:
            self.rejected_readings_count += 1
            self.last_rejection_reason = reason
            return self.get_state(now)

        # Valid reading arrived
        self._last_valid_time = now
        self._last_voltage = float(voltage)
        self._history.append(float(voltage))

        if self._last_voltage < self.low_threshold:
            self._consecutive_low += 1
            self._consecutive_recovery = 0
            if self._consecutive_low >= self.consecutive_low_required:
                self._state = BatteryState.LOW
        else:
            # Voltage >= low_threshold
            if self._state == BatteryState.LOW:
                # Require voltage >= low_threshold + hysteresis to recover
                if self._last_voltage >= (self.low_threshold + self.hysteresis):
                    self._consecutive_recovery += 1
                    if self._consecutive_recovery >= self.recovery_count:
                        self._state = BatteryState.NORMAL
                        self._consecutive_low = 0
                        self._consecutive_recovery = 0
                else:
                    self._consecutive_recovery = 0
            else:
                self._consecutive_low = 0
                self._consecutive_recovery = 0
                self._state = BatteryState.NORMAL

        return self._state

    # -------------------------------------------------------------------------
    # Query & Accessors
    # -------------------------------------------------------------------------

    def get_state(self, timestamp=None):
        """
        Query current state, accounting for telemetry staleness.

        Args:
            timestamp: Monotonic timestamp (uses clock_fn if None)

        Returns:
            BatteryState enum string
        """
        now = timestamp if timestamp is not None else self.clock_fn()

        if self._last_valid_time is None:
            return BatteryState.UNKNOWN

        if (now - self._last_valid_time) > self.stale_timeout:
            return BatteryState.STALE

        return self._state

    @property
    def state(self):
        """Current state evaluated at the current time."""
        return self.get_state()

    @property
    def last_voltage(self):
        """Most recent valid voltage reading, or None."""
        return self._last_voltage

    @property
    def last_valid_time(self):
        """Monotonic timestamp of most recent valid reading, or None."""
        return self._last_valid_time

    @property
    def consecutive_low_count(self):
        """Number of consecutive low readings observed."""
        return self._consecutive_low

    @property
    def rolling_average(self):
        """Rolling average of recent valid voltage readings, or None if empty."""
        if not self._history:
            return None
        return sum(self._history) / len(self._history)

    def is_ok(self, timestamp=None, allow_unknown=False):
        """
        Check if battery status is safe for flight.

        Args:
            timestamp: Monotonic timestamp (uses clock_fn if None)
            allow_unknown: If True, UNKNOWN state is treated as OK (e.g. before connection)

        Returns:
            True if state is NORMAL (or UNKNOWN when allowed), False if LOW or STALE
        """
        st = self.get_state(timestamp)
        if st == BatteryState.NORMAL:
            return True
        if st == BatteryState.UNKNOWN:
            return allow_unknown
        return False

    def reset(self):
        """Reset history, counters, and state to UNKNOWN."""
        self._state = BatteryState.UNKNOWN
        self._last_voltage = None
        self._last_valid_time = None
        self._consecutive_low = 0
        self._consecutive_recovery = 0
        self._history.clear()
        self.rejected_readings_count = 0
        self.total_readings_count = 0
        self.last_rejection_reason = None
