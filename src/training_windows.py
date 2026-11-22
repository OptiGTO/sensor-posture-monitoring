"""Shared validation for user-selectable fixed training windows."""

from __future__ import annotations

from collections.abc import Iterable


MIN_WINDOW_SECONDS = 1
MAX_WINDOW_SECONDS = 60
QUICK_WINDOW_SECONDS = (1, 2, 3, 5, 10, 15, 20, 30)


def validate_training_windows(windows: Iterable[int | str]) -> tuple[int, ...]:
    """Return unique, sorted whole-second windows within the supported range."""
    selected: list[int] = []
    for value in windows:
        if isinstance(value, bool):
            raise ValueError("윈도우 길이는 정수여야 합니다.")
        try:
            numeric = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("윈도우 길이는 정수여야 합니다.") from error
        if not isinstance(value, str) and numeric != value:
            raise ValueError("윈도우 길이는 정수여야 합니다.")
        selected.append(numeric)
    result = tuple(sorted(set(selected)))
    if not result:
        raise ValueError("학습할 윈도우 길이를 하나 이상 선택해야 합니다.")
    if any(value < MIN_WINDOW_SECONDS or value > MAX_WINDOW_SECONDS for value in result):
        raise ValueError(
            f"윈도우 길이는 {MIN_WINDOW_SECONDS}초부터 {MAX_WINDOW_SECONDS}초 사이여야 합니다."
        )
    return result


def parse_window_text(value: str) -> tuple[int, ...]:
    """Parse comma/space separated whole-second values entered in the GUI."""
    tokens = value.replace(",", " ").split()
    return validate_training_windows(tokens)


def merge_window_selection(quick_windows: Iterable[int], custom_text: str) -> tuple[int, ...]:
    """Combine checked quick choices with optional free-form GUI input."""
    custom = parse_window_text(custom_text) if custom_text.strip() else ()
    return validate_training_windows((*quick_windows, *custom))
