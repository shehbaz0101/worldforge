"""Open-loop multi-step latent rollout.

Given an initial observation and a sequence of actions, the Day 3 model
is stepped in latent space and decoded at each horizon. Intermediate
predictions are not encoded again. Horizon 1 is
:meth:`~worldforge.models.world.WorldModel.predict`. This module does not
update parameters and it does not search for actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldforge.models.tensors import module_device, prepare_feature_tensor
from worldforge.models.world import WorldModel
from worldforge.schemas import Trajectory


@dataclass(frozen=True)
class LatentRollout:
    """Decoded open-loop trajectory.

    ``predicted_observations`` has shape ``(..., horizon, obs_dim)`` and
    index ``h`` is the prediction at horizon ``h + 1``. ``latents`` has
    shape ``(..., horizon, latent_dim)`` and holds ``z_1 .. z_H``.
    ``initial_latent`` is ``z_0 = encode(observation)``.
    """

    predicted_observations: torch.Tensor
    latents: torch.Tensor
    initial_latent: torch.Tensor


@dataclass(frozen=True)
class TrajectoryWindow:
    """One episode prefix used as an open-loop question.

    ``observation`` is the first stored state, shape ``(obs_dim,)``.
    ``actions`` has shape ``(horizon, action_dim)``. ``targets`` has shape
    ``(horizon, obs_dim)`` and is the stored next observation at each step.
    """

    observation: torch.Tensor
    actions: torch.Tensor
    targets: torch.Tensor


def rollout_latent(
    model: WorldModel,
    observation: torch.Tensor,
    actions: torch.Tensor,
) -> LatentRollout:
    """Roll ``model`` open-loop from ``observation`` under ``actions``.

    ``observation`` has shape ``(..., obs_dim)``. ``actions`` has shape
    ``(..., horizon, action_dim)`` with ``horizon >= 1``. The leading
    dimensions must match. Each step is ``z <- step(z, action)`` followed
    by ``decode(z)``. The encoder runs once, on the initial observation.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    device = module_device(model)
    obs = prepare_feature_tensor(
        observation,
        name="observation",
        features=model.config.obs_dim,
        device=device,
    )
    acts = prepare_feature_tensor(
        actions,
        name="actions",
        features=model.config.action_dim,
        device=device,
    )
    horizon = _horizon_length(obs, acts, action_dim=model.config.action_dim)
    initial_latent = model.encode(obs)
    latent = initial_latent
    predicted: list[torch.Tensor] = []
    stepped: list[torch.Tensor] = []
    for index in range(horizon):
        latent = model.step(latent, acts[..., index, :])
        stepped.append(latent)
        predicted.append(model.decode(latent))
    return LatentRollout(
        predicted_observations=torch.stack(predicted, dim=-2),
        latents=torch.stack(stepped, dim=-2),
        initial_latent=initial_latent,
    )


def trajectory_window(
    model: WorldModel,
    trajectory: Trajectory,
    *,
    horizon: int | None = None,
) -> TrajectoryWindow:
    """Read an initial observation, actions, and targets from ``trajectory``.

    ``horizon`` left as ``None`` uses every stored transition. A shorter
    prefix stops early. The observation chain must be consistent: each
    step's observation equals the previous step's next observation.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be a Trajectory")
    limit = resolve_horizon(trajectory, horizon)
    _require_chain(trajectory, limit)
    _require_widths(model, trajectory, limit)
    steps = trajectory.transitions
    observation = torch.tensor(steps[0].observation.values, dtype=torch.float32)
    action_rows = [steps[index].action.values for index in range(limit)]
    target_rows = [steps[index].next_observation.values for index in range(limit)]
    actions = torch.tensor(action_rows, dtype=torch.float32)
    targets = torch.tensor(target_rows, dtype=torch.float32)
    return TrajectoryWindow(observation=observation, actions=actions, targets=targets)


def rollout_trajectory(
    model: WorldModel,
    trajectory: Trajectory,
    *,
    horizon: int | None = None,
) -> LatentRollout:
    """Open-loop rollout on one stored episode.

    Actions come from the trajectory. Predicted observations are not written
    back into the latent state.
    """

    window = trajectory_window(model, trajectory, horizon=horizon)
    return rollout_latent(model, window.observation, window.actions)


def resolve_horizon(trajectory: Trajectory, horizon: int | None) -> int:
    """Number of steps to score on ``trajectory``.

    ``None`` means the full transition list. An explicit horizon must be at
    least 1 and cannot pass the end of the episode.
    """

    length = len(trajectory.transitions)
    if length < 1:
        raise ValueError(
            "trajectory has no transitions "
            f"(env_id={trajectory.env_id}, seed={trajectory.seed})"
        )
    if horizon is None:
        return length
    limit = require_int(horizon, label="horizon", minimum=1)
    if limit > length:
        raise ValueError(
            f"horizon {limit} exceeds {length} transitions "
            f"(env_id={trajectory.env_id}, seed={trajectory.seed})"
        )
    return limit


def require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    """Reject bools and non-integers. Optionally enforce a lower bound."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _horizon_length(observation: torch.Tensor, actions: torch.Tensor, *, action_dim: int) -> int:
    if actions.ndim < 2:
        raise ValueError(
            f"actions must have shape (..., horizon, {action_dim}), got {tuple(actions.shape)}"
        )
    if observation.shape[:-1] != actions.shape[:-2]:
        raise ValueError(
            "observation and actions leading shapes must match, "
            f"got {tuple(observation.shape[:-1])} and {tuple(actions.shape[:-2])}"
        )
    horizon = int(actions.shape[-2])
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return horizon


def _require_chain(trajectory: Trajectory, horizon: int) -> None:
    steps = trajectory.transitions
    for index in range(1, horizon):
        previous = steps[index - 1].next_observation.values
        current = steps[index].observation.values
        if previous != current:
            raise ValueError(
                f"trajectory observations are not a single chain at step {index} "
                f"(env_id={trajectory.env_id}, seed={trajectory.seed})"
            )


def _require_widths(model: WorldModel, trajectory: Trajectory, horizon: int) -> None:
    obs_dim = model.config.obs_dim
    action_dim = model.config.action_dim
    for index, step in enumerate(trajectory.transitions[:horizon]):
        observation = step.observation.values
        nxt = step.next_observation.values
        action = step.action.values
        if len(observation) != obs_dim or len(nxt) != obs_dim:
            raise ValueError(
                f"observation length {len(observation)} does not match obs_dim {obs_dim} "
                f"at step {index}"
            )
        if len(action) != action_dim:
            raise ValueError(
                f"action length {len(action)} does not match action_dim {action_dim} "
                f"at step {index}"
            )
