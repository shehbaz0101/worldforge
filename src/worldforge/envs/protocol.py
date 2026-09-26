"""Minimal dynamics-env protocol.

The step signature matches Gymnasium's five-tuple, without depending on
Gymnasium. Day 1 has one implementation. Later environments can satisfy the
same methods and still land in the trajectory schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

InfoValue = float | int | bool | str


@runtime_checkable
class DynamicsEnv(Protocol):
    """Deterministic controlled dynamical system."""

    env_id: str
    observation_dim: int
    action_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    seed: int | None
    t: float

    def reset(
        self,
        seed: int | None = None,
        *,
        state: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """Return the initial observation. The same seed repeats it."""

    def step(
        self,
        action: Sequence[float],
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, InfoValue]]:
        """Advance one step. Returns obs, reward, terminated, truncated, info."""

    def render_ascii(self) -> str:
        """Return a one-line state summary for logs and the Day 1 CLI."""
