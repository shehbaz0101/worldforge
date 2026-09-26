"""Shared fixtures. The rate limiter is process-wide, so each test starts clean."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _reset_worldforge_rate_limit() -> Iterator[None]:
    from worldforge.ratelimit import reset_rate_limiter

    reset_rate_limiter()
    yield
    reset_rate_limiter()
