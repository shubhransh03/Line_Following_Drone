"""
PID Controller for the Line Following Drone.

A general-purpose PID controller with:
  - Anti-windup clamping on the integral term
  - Output clamping to prevent excessive corrections
  - Reset functionality for re-initialization
"""

import time


class PIDController:
    """
    Proportional-Integral-Derivative controller.

    Usage:
        pid = PIDController(Kp=0.3, Ki=0.0, Kd=0.1, integral_limit=50.0)
        output = pid.compute(error, dt)
    """

    def __init__(self, kp, ki, kd, integral_limit=50.0, output_limit=None):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Maximum absolute value of integral term (anti-windup)
            output_limit: Maximum absolute value of output (None = no limit)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = None

    def compute(self, error, dt=None):
        """
        Compute PID output for given error.

        Args:
            error: Current error value (setpoint - measurement)
            dt: Time delta in seconds. If None, computed automatically
                from wall clock (first call returns proportional-only output).

        Returns:
            Control output value
        """
        current_time = time.time()

        if dt is None:
            if self._last_time is None:
                # First call — no derivative or integral yet
                self._last_time = current_time
                self._previous_error = error
                output = self.kp * error
                return self._clamp_output(output)
            dt = current_time - self._last_time

        # Guard against zero or negative dt
        if dt <= 0:
            dt = 0.001

        self._last_time = current_time

        # --- Proportional ---
        p_term = self.kp * error

        # --- Integral (with anti-windup) ---
        self._integral += error * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        # --- Derivative ---
        derivative = (error - self._previous_error) / dt
        d_term = self.kd * derivative

        self._previous_error = error

        output = p_term + i_term + d_term
        return self._clamp_output(output)

    def reset(self):
        """Reset the PID controller state."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = None

    def _clamp_output(self, output):
        """Clamp output to configured limits."""
        if self.output_limit is not None:
            return max(-self.output_limit, min(self.output_limit, output))
        return output

    def __repr__(self):
        return (f"PIDController(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
                f"integral_limit={self.integral_limit}, "
                f"output_limit={self.output_limit})")
