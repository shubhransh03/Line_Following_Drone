"""
pid_controller.py — PID Controller for Raspberry Pi Deployment.

Reuses the verified PIDController from the project root to guarantee
zero code duplication and identical mathematical behavior.
"""

import sys
from pathlib import Path

# Ensure project root is available on sys.path for direct module import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pid_controller import PIDController

__all__ = ["PIDController"]
