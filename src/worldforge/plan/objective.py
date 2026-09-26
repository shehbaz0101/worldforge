"""Regulation objective for model-based planning.

The Day 1 environment scores a state by the negative L1 distance from the
Lotka-Volterra coexistence equilibrium. The planner maximizes the
undiscounted sum of that same score on decoded observations. It does not
learn a reward, and it does not step the environment while scoring a
candidate.

``regulation_l1`` is the only objective in this revision. When the
observation width is 2, the target is ``(1, 1)``. For any other width the
target is a vector of ones, using the same negative-L1 reduction. The
closed-loop eval runs the default environment, so it always uses the
equilibrium target.
"""

from __future__ import annotations

import torch

from worldforge.envs.lotka_volterra import EQUILIBRIUM

OBJECTIVE_NAME = "regulation_l1"


def default_target(obs_dim: int) -> tuple[float, ...]:
    """Target observation for :func:`regulation_reward`.

    Width 2 uses the coexistence equilibrium. Any other width uses ones.
    """

    if isinstance(obs_dim, bool) or not isinstance(obs_dim, int) or obs_dim < 1:
        raise ValueError("obs_dim must be an int >= 1")
    if obs_dim == len(EQUILIBRIUM):
        return tuple(float(value) for value in EQUILIBRIUM)
    return tuple(1.0 for _ in range(obs_dim))


def regulation_reward(observations: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative L1 distance from ``target``.

    ``observations`` has shape ``(..., obs_dim)``. ``target`` has shape
    ``(obs_dim,)``. The result has shape ``(...)`` and matches
    ``LotkaVolterraEnv`` when both vectors are a prey/predator pair and
    ``target`` is the equilibrium: ``-(|x-1| + |y-1|)``.
    """

    if not isinstance(observations, torch.Tensor):
        raise TypeError("observations must be a torch.Tensor")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if observations.ndim < 1 or target.ndim != 1:
        raise ValueError("observations must be (..., obs_dim) and target must be (obs_dim,)")
    if observations.shape[-1] != target.shape[-1]:
        raise ValueError(
            "target length must match the observation width, "
            f"got {target.shape[-1]} and {observations.shape[-1]}"
        )
    if not observations.is_floating_point() or not target.is_floating_point():
        raise TypeError("observations and target must be floating-point tensors")
    aligned = target.to(dtype=observations.dtype, device=observations.device)
    return -(observations - aligned).abs().sum(dim=-1)
