"""
알람 인터페이스.

현재 구현: 터미널 벨(\\a) 출력.
실제 GPIO Buzzer 연동은 ClockApp C++ 측에서 SHM 읽어 처리합니다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def play_alarm() -> None:
    """졸음 감지 시 한 번 알람을 울립니다 (터미널 벨)."""
    print("\a", end="", flush=True)
    logger.info("알람 발생")
