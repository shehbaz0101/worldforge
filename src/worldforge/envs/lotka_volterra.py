"""Controlled Lotka-Volterra predator-prey kinetics.

This is the default WorldForge environment. The state is two populations,
prey ``x`` and predator ``y``:

    dx/dt = x * (alpha - beta * y) + u_x
    dy/dt = y * (delta * x - gamma) + u_y

Day 1 uses ``alpha = beta = delta = gamma = 1``, so the coexistence
equilibrium is ``(1, 1)`` and the vector field is

    dx/dt = x * (1 - y) + u_x
    dy/dt = y * (x - 1) + u_y

``u`` is a dose rate clipped to ``[-0.25, 0.25]`` on each species. A positive
component stocks that species. A negative component harvests it. The dose is
held constant across the Runge-Kutta substeps of one environment step
(zero-order hold).

A Gray-Scott reaction-diffusion plate was the other candidate. It is not
the default: even a small grid makes the observation tens or hundreds of
floats, and Day 1 tests need a tiny deterministic system. The trajectory
schema is not specific to this environment, so a grid environment can be
registered later.

Given a seed, ``reset`` draws the initial populations with ``random.Random``.
Passing ``state`` uses that population instead and still records the seed.
The same seed and the same actions reproduce the same states. This module
does not use the network.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

ENV_ID = "lotka_volterra"
ALPHA = 1.0
BETA = 1.0
DELTA = 1.0
GAMMA = 1.0
EQUILIBRIUM = (GAMMA / DELTA, ALPHA / BETA)
STEP_DT = 0.2
SUBSTEPS = 4
ACTION_LIMIT = 0.25
INIT_LOW = 0.5
INIT_HIGH = 1.8
EXTINCT = 1e-8
MAX_POPULATION = 50.0
DEFAULT_HORIZON = 32
_BAR_WIDTH = 10
_BAR_SCALE = 2.0


def lotka_volterra_field(
    prey: float,
    predator: float,
    *,
    prey_dose: float = 0.0,
    predator_dose: float = 0.0,
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA,
    gamma: float = GAMMA,
) -> tuple[float, float]:
    """Return ``(dx/dt, dy/dt)`` for the controlled Lotka-Volterra field."""

    d_prey = prey * (alpha - beta * predator) + prey_dose
    d_predator = predator * (delta * prey - gamma) + predator_dose
    return d_prey, d_predator


class LotkaVolterraEnv:
    """Two-species kinetic sandbox with a dose action.

    ``reset(seed)`` samples the initial population. ``step(action)`` integrates
    ``STEP_DT`` time units and returns the Gymnasium-style tuple
    ``(obs, reward, terminated, truncated, info)``. The reward is the negative
    L1 distance of the next population from the coexistence equilibrium, a
    placeholder regulation objective for later control evals.

    The episode truncates at ``horizon`` steps. It terminates if either
    population dies out or exceeds ``MAX_POPULATION``.
    """

    env_id = ENV_ID
    observation_dim = 2
    action_dim = 2

    def __init__(self, horizon: int = DEFAULT_HORIZON) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be an int >= 1")
        self.horizon = horizon
        self.step_dt = STEP_DT
        self.substeps = SUBSTEPS
        self.action_low = (-ACTION_LIMIT, -ACTION_LIMIT)
        self.action_high = (ACTION_LIMIT, ACTION_LIMIT)
        self.seed: int | None = None
        self.t = 0.0
        self._prey = 0.0
        self._predator = 0.0
        self._step_index = 0
        self._ready = False
        self._done = False

    def reset(
        self,
        seed: int | None = None,
        *,
        state: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        resolved = _resolve_seed(seed)
        if state is None:
            rng = random.Random(resolved)
            prey = rng.uniform(INIT_LOW, INIT_HIGH)
            predator = rng.uniform(INIT_LOW, INIT_HIGH)
        else:
            prey, predator = _parse_state(state)
        self.seed = resolved
        self._prey = prey
        self._predator = predator
        self.t = 0.0
        self._step_index = 0
        self._ready = True
        self._done = False
        return (self._prey, self._predator)

    def step(
        self,
        action: Sequence[float],
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, float | int | bool | str]]:
        if not self._ready:
            raise RuntimeError("call reset before step")
        if self._done:
            raise RuntimeError("episode is finished; call reset")
        prey_dose, predator_dose, clipped = _clip_action(
            action,
            low=self.action_low,
            high=self.action_high,
        )
        self._prey, self._predator = _integrate(
            self._prey,
            self._predator,
            prey_dose,
            predator_dose,
            step_dt=self.step_dt,
            substeps=self.substeps,
        )
        self._step_index += 1
        self.t = self._step_index * self.step_dt
        terminated = _out_of_bounds(self._prey, self._predator)
        truncated = self._step_index >= self.horizon
        self._done = terminated or truncated
        reward = _regulation_reward(self._prey, self._predator)
        info: dict[str, float | int | bool | str] = {
            "t": self.t,
            "step": self._step_index,
            "action_clipped": clipped,
            "prey_dose": prey_dose,
            "predator_dose": predator_dose,
        }
        return (self._prey, self._predator), reward, terminated, truncated, info

    def render_ascii(self) -> str:
        """One-line population summary. Bars saturate at ``_BAR_SCALE``."""

        if not self._ready:
            raise RuntimeError("call reset before render_ascii")
        return (
            f"t={self.t:6.2f}  prey [{_bar(self._prey)}] {self._prey:7.4f}  "
            f"predator [{_bar(self._predator)}] {self._predator:7.4f}"
        )

    def __repr__(self) -> str:
        return f"LotkaVolterraEnv(horizon={self.horizon}, env_id={self.env_id!r})"


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.SystemRandom().randrange(0, 2**31)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    return seed


def _parse_state(state: Sequence[float]) -> tuple[float, float]:
    if len(state) != 2:
        raise ValueError(f"state must have length 2, got {len(state)}")
    prey = _as_float(state[0], label="prey")
    predator = _as_float(state[1], label="predator")
    if prey < 0.0 or predator < 0.0:
        raise ValueError("populations must be >= 0")
    return prey, predator


def _as_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _clip_action(
    action: Sequence[float],
    *,
    low: tuple[float, ...],
    high: tuple[float, ...],
) -> tuple[float, float, bool]:
    if len(action) != 2:
        raise ValueError(f"action must have length 2, got {len(action)}")
    prey_dose = _as_float(action[0], label="prey_dose")
    predator_dose = _as_float(action[1], label="predator_dose")
    clipped_prey = min(max(prey_dose, low[0]), high[0])
    clipped_predator = min(max(predator_dose, low[1]), high[1])
    clipped = clipped_prey != prey_dose or clipped_predator != predator_dose
    return clipped_prey, clipped_predator, clipped


def _integrate(
    prey: float,
    predator: float,
    prey_dose: float,
    predator_dose: float,
    *,
    step_dt: float,
    substeps: int,
) -> tuple[float, float]:
    """Classical RK4 with the dose held fixed, then a non-negativity projection."""

    h = step_dt / substeps
    x, y = prey, predator
    for _ in range(substeps):
        k1x, k1y = lotka_volterra_field(x, y, prey_dose=prey_dose, predator_dose=predator_dose)
        k2x, k2y = lotka_volterra_field(
            x + 0.5 * h * k1x,
            y + 0.5 * h * k1y,
            prey_dose=prey_dose,
            predator_dose=predator_dose,
        )
        k3x, k3y = lotka_volterra_field(
            x + 0.5 * h * k2x,
            y + 0.5 * h * k2y,
            prey_dose=prey_dose,
            predator_dose=predator_dose,
        )
        k4x, k4y = lotka_volterra_field(
            x + h * k3x,
            y + h * k3y,
            prey_dose=prey_dose,
            predator_dose=predator_dose,
        )
        x = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y = y + (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        x = max(0.0, x)
        y = max(0.0, y)
    return x, y


def _out_of_bounds(prey: float, predator: float) -> bool:
    if prey <= EXTINCT or predator <= EXTINCT:
        return True
    if prey >= MAX_POPULATION or predator >= MAX_POPULATION:
        return True
    return False


def _regulation_reward(prey: float, predator: float) -> float:
    eq_prey, eq_predator = EQUILIBRIUM
    cost = abs(prey - eq_prey) + abs(predator - eq_predator)
    if cost == 0.0:
        return 0.0
    return -cost


def _bar(value: float, *, width: int = _BAR_WIDTH, scale: float = _BAR_SCALE) -> str:
    fraction = 0.0 if scale <= 0.0 else max(0.0, min(value / scale, 1.0))
    filled = int(round(fraction * width))
    return "#" * filled + "-" * (width - filled)
