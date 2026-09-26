"""In-process rate limit for the expensive HTTP routes.

``GET /health`` is not limited. ``POST /rollout``, ``POST /eval/predict``,
``POST /plan``, ``POST /eval/plan``, and ``POST /train`` share one sliding
window per client address. ``TestClient`` has no peer address other than
``testclient``, so those calls share one bucket.

``WORLDFORGE_RATE_LIMIT`` is the number of those POSTs allowed per window
(default 60). ``WORLDFORGE_RATE_WINDOW_SECONDS`` is the window length in
seconds (default 60). ``worldforge serve --rate-limit`` and
``--rate-window`` set the same variables for the server process.
"""

from __future__ import annotations

import math
import os
import threading
import time

DEFAULT_RATE_LIMIT = 60
DEFAULT_RATE_WINDOW_SECONDS = 60.0
RATE_LIMIT_ENV = "WORLDFORGE_RATE_LIMIT"
RATE_WINDOW_ENV = "WORLDFORGE_RATE_WINDOW_SECONDS"

# Mutating and expensive routes. Health is absent on purpose.
LIMITED_PATHS = frozenset(
    {
        "/rollout",
        "/eval/predict",
        "/plan",
        "/eval/plan",
        "/train",
    }
)


class RateLimitConfigError(ValueError):
    """``WORLDFORGE_RATE_LIMIT`` or the window is not a usable number."""


class RateLimiter:
    """Sliding window of hit timestamps, keyed by client address."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def configure(self, limit: int, window_seconds: float) -> None:
        """Apply new settings. A change drops recorded hits."""

        with self._lock:
            if self.limit == limit and self.window_seconds == window_seconds:
                return
            self.limit = limit
            self.window_seconds = window_seconds
            self._hits.clear()

    def reset(self) -> None:
        """Drop every recorded hit. Tests use this between cases."""

        with self._lock:
            self._hits.clear()

    def check(self, key: str, *, now: float | None = None) -> int | None:
        """Record one hit, or return ``Retry-After`` seconds when limited.

        ``now`` is a monotonic timestamp. Production calls omit it.
        """

        moment = time.monotonic() if now is None else now
        with self._lock:
            window = self.window_seconds
            hits = [stamp for stamp in self._hits.get(key, []) if moment - stamp < window]
            if len(hits) >= self.limit:
                retry_after = window - (moment - hits[0])
                self._hits[key] = hits
                return max(1, math.ceil(retry_after - 1e-9))
            hits.append(moment)
            self._hits[key] = hits
            return None


_limiter = RateLimiter(DEFAULT_RATE_LIMIT, DEFAULT_RATE_WINDOW_SECONDS)


def read_rate_settings() -> tuple[int, float]:
    """Read the limit and window from the environment."""

    raw_limit = os.environ.get(RATE_LIMIT_ENV, str(DEFAULT_RATE_LIMIT))
    raw_window = os.environ.get(RATE_WINDOW_ENV, str(DEFAULT_RATE_WINDOW_SECONDS))
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise RateLimitConfigError(
            f"{RATE_LIMIT_ENV} must be an integer >= 1, got {raw_limit!r}"
        ) from exc
    try:
        window = float(raw_window)
    except (TypeError, ValueError) as exc:
        raise RateLimitConfigError(
            f"{RATE_WINDOW_ENV} must be a positive number of seconds, got {raw_window!r}"
        ) from exc
    if not math.isfinite(window) or window <= 0:
        raise RateLimitConfigError(
            f"{RATE_WINDOW_ENV} must be a positive number of seconds, got {raw_window!r}"
        )
    if limit < 1:
        raise RateLimitConfigError(f"{RATE_LIMIT_ENV} must be an integer >= 1, got {raw_limit!r}")
    return limit, window


def get_limiter() -> RateLimiter:
    """Return the process limiter, updated from the environment."""

    limit, window = read_rate_settings()
    _limiter.configure(limit, window)
    return _limiter


def reset_rate_limiter() -> None:
    """Clear recorded hits on the process limiter."""

    _limiter.reset()
