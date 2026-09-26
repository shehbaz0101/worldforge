"""Cross-entropy method planner on a latent world model.

The search variable is a short action sequence. Candidates are rolled
open-loop with :func:`worldforge.eval.rollout.rollout_latent` and scored
by :data:`~worldforge.plan.objective.OBJECTIVE_NAME`. The environment is
not stepped here. Actions are clipped to the configured bounds before
they are scored and before they are returned.

The population is an axis-aligned Gaussian. Its mean starts at the zero
dose, which is also scored, so the returned sequence is at least as good
as doing nothing under the model. Each iteration draws ``n_samples``
sequences, clips them, keeps the top elite fraction, and replaces the
mean and standard deviation with that elite's statistics. The sequence
handed back is the best candidate scored during the search, including
each iteration's clipped mean.

Defaults are small on purpose: a few samples, a short horizon, and a
couple of iterations. Closed-loop control replans at every environment
step and executes only the first action.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from worldforge.envs.lotka_volterra import ACTION_LIMIT
from worldforge.eval.rollout import rollout_latent
from worldforge.models.tensors import module_device, prepare_feature_tensor
from worldforge.models.world import WorldModel
from worldforge.plan.objective import default_target, regulation_reward

DEFAULT_HORIZON = 3
DEFAULT_N_SAMPLES = 8
DEFAULT_N_ITERATIONS = 2
DEFAULT_ELITE_FRACTION = 0.25
DEFAULT_MIN_STD = 0.01
DEFAULT_ACTION_LOW = (-ACTION_LIMIT, -ACTION_LIMIT)
DEFAULT_ACTION_HIGH = (ACTION_LIMIT, ACTION_LIMIT)


@dataclass(frozen=True)
class CEMConfig:
    """Search settings for one CEM plan.

    ``seed`` draws the Gaussian samples from a private CPU generator. It
    does not initialize the network. ``init_std`` is the starting standard
    deviation on every action component and defaults to the Lotka-Volterra
    dose limit. ``min_std`` keeps later iterations from collapsing to a
    single point. ``action_low`` and ``action_high`` are inclusive clip
    bounds, one value per action component.
    """

    horizon: int = DEFAULT_HORIZON
    n_samples: int = DEFAULT_N_SAMPLES
    n_iterations: int = DEFAULT_N_ITERATIONS
    elite_fraction: float = DEFAULT_ELITE_FRACTION
    seed: int = 0
    init_std: float = ACTION_LIMIT
    min_std: float = DEFAULT_MIN_STD
    action_low: tuple[float, ...] = DEFAULT_ACTION_LOW
    action_high: tuple[float, ...] = DEFAULT_ACTION_HIGH

    def __post_init__(self) -> None:
        horizon = _require_int(self.horizon, label="horizon", minimum=1)
        n_samples = _require_int(self.n_samples, label="n_samples", minimum=1)
        n_iterations = _require_int(self.n_iterations, label="n_iterations", minimum=1)
        seed = _require_int(self.seed, label="seed", minimum=0)
        elite = _require_fraction(self.elite_fraction, label="elite_fraction")
        init_std = _require_positive(self.init_std, label="init_std")
        min_std = _require_positive(self.min_std, label="min_std")
        if min_std > init_std:
            raise ValueError("min_std must be <= init_std")
        low = _require_bounds(self.action_low, label="action_low")
        high = _require_bounds(self.action_high, label="action_high")
        if len(low) != len(high):
            raise ValueError("action_low and action_high must have the same length")
        if any(lower > upper for lower, upper in zip(low, high, strict=True)):
            raise ValueError("action_low must be <= action_high on every component")
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "n_iterations", n_iterations)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "elite_fraction", elite)
        object.__setattr__(self, "init_std", init_std)
        object.__setattr__(self, "min_std", min_std)
        object.__setattr__(self, "action_low", low)
        object.__setattr__(self, "action_high", high)

    @property
    def n_elite(self) -> int:
        """How many samples survive each iteration. At least one."""

        count = math.ceil(self.elite_fraction * self.n_samples)
        return max(1, min(self.n_samples, count))


@dataclass(frozen=True)
class ActionPlan:
    """Best clipped action sequence found by :func:`plan_actions`.

    ``actions`` has shape ``(horizon, action_dim)``, float32, on CPU.
    ``predicted_return`` is the undiscounted sum of regulation rewards of
    the decoded states. Higher is better. The initial observation is not
    included.
    """

    actions: torch.Tensor
    predicted_return: float


def score_action_sequence(
    model: WorldModel,
    observation: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Undiscounted regulation return of an open-loop latent rollout.

    ``observation`` has shape ``(..., obs_dim)``. ``actions`` has shape
    ``(..., horizon, action_dim)``. The result has shape ``(...)``. This
    function does not clip actions and does not change the module's
    train/eval flag. :func:`plan_actions` clips before it calls this.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    rolled = rollout_latent(model, observation, actions)
    target = _target_tensor(
        model.config.obs_dim,
        device=rolled.predicted_observations.device,
        dtype=rolled.predicted_observations.dtype,
    )
    rewards = regulation_reward(rolled.predicted_observations, target)
    return rewards.sum(dim=-1)


def plan_actions(
    model: WorldModel,
    observation: torch.Tensor,
    config: CEMConfig | None = None,
) -> ActionPlan:
    """Search one clipped action sequence from ``observation``.

    ``observation`` has shape ``(obs_dim,)``. The model is set to eval for
    the search and the previous mode is restored. Sampling uses a private
    generator. The caller's CPU generator state is restored, and the intra-op
    thread count is pinned to 1 for the search so a seed repeats.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    if config is None:
        config = CEMConfig()
    if not isinstance(config, CEMConfig):
        raise TypeError("config must be a CEMConfig")
    device = module_device(model)
    if device.type != "cpu":
        raise ValueError("planner runs on cpu")
    obs = prepare_feature_tensor(
        observation,
        name="observation",
        features=model.config.obs_dim,
        device=device,
    )
    if obs.ndim != 1:
        raise ValueError(
            f"observation must have shape ({model.config.obs_dim},), got {tuple(obs.shape)}"
        )
    _require_action_bounds(config, model.config.action_dim)
    obs = obs.detach()

    low = torch.tensor(config.action_low, dtype=torch.float32)
    high = torch.tensor(config.action_high, dtype=torch.float32)
    horizon = config.horizon
    action_dim = model.config.action_dim
    mu = torch.zeros(horizon, action_dim, dtype=torch.float32)
    std = torch.full((horizon, action_dim), config.init_std, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    was_training = model.training
    model.eval()
    try:
        with _cpu_plan_context(), torch.no_grad():
            zero_return = _finite_return(score_action_sequence(model, obs, mu), label="return")
            best_actions = mu.clone()
            best_return = zero_return
            for _iteration in range(config.n_iterations):
                samples = mu + std * torch.randn(
                    (config.n_samples, horizon, action_dim),
                    generator=generator,
                    dtype=torch.float32,
                )
                samples = torch.maximum(torch.minimum(samples, high), low)
                returns = score_action_sequence(
                    model,
                    obs.unsqueeze(0).expand(config.n_samples, -1),
                    samples,
                )
                returns = _finite_returns(returns)
                elite_index = torch.topk(returns, config.n_elite).indices
                elites = samples[elite_index]
                elite_returns = returns[elite_index]
                local_best = int(torch.argmax(elite_returns).item())
                local_return = float(elite_returns[local_best].item())
                if local_return > best_return:
                    best_return = local_return
                    best_actions = elites[local_best].detach().clone()
                mu = elites.mean(dim=0)
                if config.n_elite == 1:
                    std = torch.full_like(mu, config.min_std)
                else:
                    std = elites.std(dim=0, unbiased=False).clamp_min(config.min_std)
                mu = torch.maximum(torch.minimum(mu, high), low)
                mean_return = _finite_return(
                    score_action_sequence(model, obs, mu),
                    label="return",
                )
                if mean_return > best_return:
                    best_return = mean_return
                    best_actions = mu.detach().clone()
            confirmed = _finite_return(
                score_action_sequence(model, obs, best_actions),
                label="return",
            )
    finally:
        model.train(was_training)

    return ActionPlan(
        actions=best_actions.detach().to(dtype=torch.float32, device="cpu"),
        predicted_return=confirmed,
    )


def _target_tensor(obs_dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = default_target(obs_dim)
    return torch.tensor(values, dtype=dtype, device=device)


def _require_action_bounds(config: CEMConfig, action_dim: int) -> None:
    if len(config.action_low) != action_dim or len(config.action_high) != action_dim:
        raise ValueError(
            "action bounds must match action_dim, "
            f"got low {len(config.action_low)}, high {len(config.action_high)}, action_dim {action_dim}"
        )


def _finite_return(value: torch.Tensor, *, label: str) -> float:
    if value.ndim != 0:
        raise ValueError(f"{label} must be a scalar")
    number = float(value.detach().item())
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _finite_returns(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"returns must have shape (n_samples,), got {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise ValueError("predicted returns are not finite")
    return values


def _require_int(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_fraction(value: object, *, label: str) -> float:
    number = _require_finite(value, label=label)
    if number <= 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in (0, 1]")
    return number


def _require_positive(value: object, *, label: str) -> float:
    number = _require_finite(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0")
    return number


def _require_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_bounds(value: object, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of floats")
    if len(value) < 1:
        raise ValueError(f"{label} must be non-empty")
    return tuple(_require_finite(item, label=label) for item in value)


@contextmanager
def _cpu_plan_context() -> Iterator[None]:
    """Pin CPU kernels for one search, then restore threads, MKLDNN, and RNG."""

    threads = torch.get_num_threads()
    rng_state = torch.get_rng_state().clone()
    mkldnn = _mkldnn_enabled()
    try:
        torch.set_num_threads(1)
        if mkldnn is not None:
            _set_mkldnn(False)
        yield
    finally:
        if mkldnn is not None:
            _set_mkldnn(mkldnn)
        torch.set_rng_state(rng_state)
        torch.set_num_threads(threads)


def _mkldnn_enabled() -> bool | None:
    backend = getattr(torch.backends, "mkldnn", None)
    if backend is None or not backend.is_available():
        return None
    return bool(backend.enabled)


def _set_mkldnn(enabled: bool) -> None:
    backend = getattr(torch.backends, "mkldnn", None)
    if backend is not None and backend.is_available():
        backend.enabled = enabled
