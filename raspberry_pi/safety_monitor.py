"""
safety_monitor.py — Integrated Multi-Subsystem Safety Monitor for Autonomous Drone Guidance.

Monitors:
  - Camera freshness & persistent frame-drop detection.
  - Autopilot telemetry link & heartbeat staleness.
  - Battery health (hysteresis, rolling filtering, and disconnect detection).
  - Pilot RC transmitter manual override.
  - Autopilot flight mode divergence.
  - Line loss duration watchdog.

Provides deterministic SafetyDecision objects with clear action codes:
  - 'CONTINUE': All subsystems nominal.
  - 'HOVER': Transient fault requiring immediate velocity arrest (e.g. short frame drop or line search).
  - 'ABORT': Critical unrecoverable fault requiring autonomous landing.
  - 'RELEASE_CONTROL': Pilot has taken manual control via RC switch or mode change.
"""

from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import time
from typing import Callable, List, Optional

# Ensure project root is available on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from battery_monitor import BatteryMonitor, BatteryState
from raspberry_pi.mavlink_adapter import MAVLinkAdapterBase


class SafetyAction(str, Enum):
    """Actions mandated by the safety monitor."""
    CONTINUE = "CONTINUE"
    HOVER = "HOVER"
    ABORT = "ABORT"
    RELEASE_CONTROL = "RELEASE_CONTROL"


@dataclass
class SafetyDecision:
    """Decision evaluated by SafetyMonitor on each control cycle."""
    action: SafetyAction
    safe_to_proceed: bool
    fault_code: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


class SafetyMonitor:
    """
    Central safety monitor evaluating all critical flight conditions.
    """

    def __init__(self,
                 battery_monitor: Optional[BatteryMonitor] = None,
                 camera_frame_timeout: float = 1.0,
                 camera_fault_timeout: float = 3.0,
                 telemetry_timeout: float = 3.0,
                 line_lost_timeout: float = 3.0,
                 rc_override_channel: Optional[int] = None,
                 rc_override_threshold: int = 1800,
                 rc_override_enabled: bool = False,
                 clock_fn: Callable[[], float] = time.monotonic):
        """
        Initialize safety monitor.

        Args:
            battery_monitor: Instance of BatteryMonitor from battery_monitor.py.
            camera_frame_timeout: Seconds without frame before ordering HOVER.
            camera_fault_timeout: Seconds without frame before ordering ABORT.
            telemetry_timeout: Seconds without heartbeat before ordering ABORT.
            line_lost_timeout: Seconds without line before ordering ABORT.
            rc_override_channel: RC channel to monitor for manual pilot takeover.
            rc_override_threshold: PWM threshold for override.
            rc_override_enabled: Whether RC override check is active.
            clock_fn: Injectable monotonic clock callable.
        """
        self.clock_fn = clock_fn
        self.camera_frame_timeout = float(camera_frame_timeout)
        self.camera_fault_timeout = float(camera_fault_timeout)
        self.telemetry_timeout = float(telemetry_timeout)
        self.line_lost_timeout = float(line_lost_timeout)
        self.rc_override_channel = rc_override_channel
        self.rc_override_threshold = int(rc_override_threshold)
        self.rc_override_enabled = bool(rc_override_enabled)

        self.battery_monitor = battery_monitor if battery_monitor is not None else BatteryMonitor(clock_fn=clock_fn)

        # State tracking timestamps
        self.last_valid_frame_time: Optional[float] = None
        self.last_line_detected_time: Optional[float] = None
        self.consecutive_frame_failures: int = 0
        self.is_monitoring: bool = False

    def start_monitoring(self) -> None:
        """Mark start of active autonomous flight monitoring."""
        now = self.clock_fn()
        self.last_valid_frame_time = now
        self.last_line_detected_time = now
        self.consecutive_frame_failures = 0
        self.is_monitoring = True

    def record_frame_success(self, now: Optional[float] = None) -> None:
        """Record a successful camera frame read."""
        t = now if now is not None else self.clock_fn()
        self.last_valid_frame_time = t
        self.consecutive_frame_failures = 0

    def record_frame_failure(self, now: Optional[float] = None) -> None:
        """Record a failed camera frame read."""
        self.consecutive_frame_failures += 1

    def record_line_detected(self, detected: bool, now: Optional[float] = None) -> None:
        """Record whether tracking line was detected in the current frame."""
        t = now if now is not None else self.clock_fn()
        if detected:
            self.last_line_detected_time = t

    def update_battery_voltage(self, voltage: Optional[float]) -> BatteryState:
        """Feed latest voltage into the battery state machine."""
        return self.battery_monitor.update(voltage)

    def evaluate(self,
                 mavlink: MAVLinkAdapterBase,
                 line_detected: bool = True,
                 now: Optional[float] = None) -> SafetyDecision:
        """
        Perform comprehensive safety assessment across all flight sub-systems.

        Order of precedence:
          1. Pilot RC Override (highest priority — pilot authority is absolute)
          2. Flight Mode Divergence (pilot changed mode on RC)
          3. Telemetry Link Loss (stale heartbeat)
          4. Battery Criticality (LOW or STALE)
          5. Camera Feed Loss (persistent fault -> ABORT, transient -> HOVER)
          6. Line Lost Duration (timeout -> ABORT, transient -> HOVER)

        Returns:
            SafetyDecision specifying the mandated action and diagnostic reasons.
        """
        t = now if now is not None else self.clock_fn()
        reasons: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Pilot Manual RC Override Check
        # ---------------------------------------------------------------------
        if self.rc_override_enabled and self.rc_override_channel is not None:
            if mavlink.check_rc_override(self.rc_override_channel, self.rc_override_threshold):
                return SafetyDecision(
                    action=SafetyAction.RELEASE_CONTROL,
                    safe_to_proceed=False,
                    fault_code="PILOT_OVERRIDE",
                    reasons=[f"Pilot RC override detected on channel {self.rc_override_channel}!"]
                )

        # ---------------------------------------------------------------------
        # 2. Flight Mode Divergence Check
        # ---------------------------------------------------------------------
        if mavlink.is_connected() and mavlink.is_mode_diverged(expected_mode="GUIDED"):
            telem = mavlink.get_telemetry()
            return SafetyDecision(
                action=SafetyAction.RELEASE_CONTROL,
                safe_to_proceed=False,
                fault_code="MODE_DIVERGED",
                reasons=[f"Autopilot mode diverged to '{telem.mode}'. Autonomous control surrendered."]
            )

        # ---------------------------------------------------------------------
        # 3. Telemetry Link & Heartbeat Freshness Check
        # ---------------------------------------------------------------------
        if not mavlink.is_connected():
            return SafetyDecision(
                action=SafetyAction.ABORT,
                safe_to_proceed=False,
                fault_code="TELEMETRY_DISCONNECTED",
                reasons=["Autopilot connection lost!"]
            )

        if not mavlink.is_heartbeat_fresh(self.telemetry_timeout):
            telem = mavlink.get_telemetry()
            return SafetyDecision(
                action=SafetyAction.ABORT,
                safe_to_proceed=False,
                fault_code="TELEMETRY_STALE",
                reasons=[f"Autopilot heartbeat stale ({telem.heartbeat_age_sec:.1f}s > {self.telemetry_timeout}s timeout)!"]
            )

        # ---------------------------------------------------------------------
        # 4. Battery Health Evaluation
        # ---------------------------------------------------------------------
        telem = mavlink.get_telemetry()
        # Feed voltage update if telemetry heartbeat is fresh
        batt_state = self.update_battery_voltage(telem.voltage)
        if batt_state == BatteryState.LOW:
            v_val = self.battery_monitor.last_voltage
            v_str = f"{v_val:.2f}V" if v_val is not None else "N/A"
            return SafetyDecision(
                action=SafetyAction.ABORT,
                safe_to_proceed=False,
                fault_code="LOW_BATTERY",
                reasons=[f"Battery critically low ({v_str})!"]
            )
        elif batt_state == BatteryState.STALE:
            return SafetyDecision(
                action=SafetyAction.ABORT,
                safe_to_proceed=False,
                fault_code="STALE_BATTERY",
                reasons=["Battery telemetry stale / unverified!"]
            )

        # ---------------------------------------------------------------------
        # 5. Camera Feed Freshness Check
        # ---------------------------------------------------------------------
        if self.last_valid_frame_time is not None:
            camera_drop_elapsed = t - self.last_valid_frame_time
            if camera_drop_elapsed >= self.camera_fault_timeout:
                return SafetyDecision(
                    action=SafetyAction.ABORT,
                    safe_to_proceed=False,
                    fault_code="CAMERA_FAULT",
                    reasons=[f"Camera feed lost for {camera_drop_elapsed:.1f}s (exceeds {self.camera_fault_timeout}s timeout)!"]
                )
            elif camera_drop_elapsed >= self.camera_frame_timeout:
                reasons.append(f"Camera frame drop for {camera_drop_elapsed:.1f}s. Zero-velocity hover commanded.")
                return SafetyDecision(
                    action=SafetyAction.HOVER,
                    safe_to_proceed=True,
                    fault_code="CAMERA_DROP_HOVER",
                    reasons=reasons
                )

        # ---------------------------------------------------------------------
        # 6. Line Loss Watchdog Check
        # ---------------------------------------------------------------------
        self.record_line_detected(line_detected, now=t)
        if self.last_line_detected_time is not None and not line_detected:
            line_loss_elapsed = t - self.last_line_detected_time
            if line_loss_elapsed >= self.line_lost_timeout:
                return SafetyDecision(
                    action=SafetyAction.ABORT,
                    safe_to_proceed=False,
                    fault_code="LINE_LOST_TIMEOUT",
                    reasons=[f"Line lost for {line_loss_elapsed:.1f}s (exceeds {self.line_lost_timeout}s timeout)!"]
                )
            else:
                reasons.append(f"Line lost for {line_loss_elapsed:.1f}s. Search hover commanded.")
                return SafetyDecision(
                    action=SafetyAction.HOVER,
                    safe_to_proceed=True,
                    fault_code="LINE_SEARCH_HOVER",
                    reasons=reasons
                )

        # Nominal flight condition
        return SafetyDecision(
            action=SafetyAction.CONTINUE,
            safe_to_proceed=True,
            fault_code=None,
            reasons=[]
        )
