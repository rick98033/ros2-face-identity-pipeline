"""Freshness-bounded state wrapper (SI-5.3)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class StaleStateError(Exception):
    """Raised when FreshValue.get_or_raise() detects stale state."""

    def __init__(self, age_sec: float, max_age_sec: float):
        self.age_sec = age_sec
        self.max_age_sec = max_age_sec
        super().__init__(
            f"Stale state: age={age_sec:.3f}s, max_age={max_age_sec}s"
        )


@dataclass(frozen=True)
class FreshValue(Generic[T]):
    """Value paired with acquisition timestamp and freshness bound."""

    value: T
    timestamp: float       # time.monotonic() at acquisition
    max_age_sec: float     # declared freshness bound

    @classmethod
    def now(cls, value: T, max_age_sec: float) -> FreshValue[T]:
        """Construct with current monotonic time."""
        return cls(value=value, timestamp=time.monotonic(), max_age_sec=max_age_sec)

    @property
    def age_sec(self) -> float:
        return time.monotonic() - self.timestamp

    @property
    def is_fresh(self) -> bool:
        return self.age_sec < self.max_age_sec

    def get_or_stale(self, fallback: T) -> T:
        """Return value if fresh, fallback if stale."""
        return self.value if self.is_fresh else fallback

    def get_or_raise(
        self, exc_factory: Optional[Callable[[float, float], Exception]] = None,
    ) -> T:
        """Return value if fresh, raise if stale."""
        if self.is_fresh:
            return self.value
        age = self.age_sec
        if exc_factory is not None:
            raise exc_factory(age, self.max_age_sec)
        raise StaleStateError(age, self.max_age_sec)

    def as_stale_info(self) -> dict:
        """Structured context for logging and health responses."""
        return {
            "value_age_sec": round(self.age_sec, 3),
            "max_age_sec": self.max_age_sec,
            "is_fresh": self.is_fresh,
        }
