"""
Alarm interface.

Current implementation: terminal bell (\a).
Actual GPIO buzzer integration is handled on the ClockApp C++ side by reading SHM.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def play_alarm() -> None:
    """Sound the alarm once when drowsiness is detected (terminal bell)."""
    print("\a", end="", flush=True)
    logger.info("Alarm triggered")
