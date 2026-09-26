"""One-step latent dynamics.

Day 3 baseline
--------------
``MLPDynamics`` is a deterministic residual MLP::

    z_{t+1} = z_t + mlp(concat(z_t, a_t))

The residual is part of the baseline, not a learned gate. The same
``z`` and the same action always produce the same next latent. There
is no hidden recurrent state beyond ``z`` itself.

Extension point
---------------
``build_dynamics`` is the only place that chooses a variant.
``ModelConfig.dynamics == "rssm"`` is reserved for a later stochastic
recurrent state-space model (a noisy latent, typically with a separate
deterministic recurrent state). That class is not implemented.
A future variant should keep ``forward(latent, action) -> next_latent``
so :meth:`worldforge.models.world.WorldModel.step` stays the call site
used by training.
"""

from __future__ import annotations

import torch
from torch import nn

from worldforge.models.config import ModelConfig
from worldforge.models.layers import describe_mlp, mlp
from worldforge.models.tensors import module_device, prepare_feature_tensor, require_same_leading

RSSM_NOT_IMPLEMENTED = (
    "dynamics 'rssm' is reserved for a later stochastic recurrent state-space "
    "model and is not implemented. Day 3 ships the deterministic residual MLP "
    "baseline (dynamics 'mlp')."
)


class MLPDynamics(nn.Module):
    """Deterministic residual MLP, ``(z_t, a_t) -> z_{t+1}``."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.net = mlp(
            latent_dim + action_dim,
            latent_dim,
            hidden_dim,
            hidden_layers,
            generator,
        )

    def describe(self) -> str:
        width = self.latent_dim + self.action_dim
        sizes = [width, *([self.hidden_dim] * self.hidden_layers), self.latent_dim]
        return describe_mlp(sizes, residual=True)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = prepare_feature_tensor(
            latent,
            name="latent",
            features=self.latent_dim,
            device=module_device(self),
        )
        action = prepare_feature_tensor(
            action,
            name="action",
            features=self.action_dim,
            device=module_device(self),
        )
        require_same_leading(latent, action, left_name="latent", right_name="action")
        delta = self.net(torch.cat((latent, action), dim=-1))
        return latent + delta


def build_dynamics(config: ModelConfig, generator: torch.Generator) -> MLPDynamics:
    """Construct the Day 3 dynamics module named by ``config.dynamics``.

    ``mlp`` returns :class:`MLPDynamics`. ``rssm`` raises
    :class:`NotImplementedError`. Any other name is rejected by
    :class:`~worldforge.models.config.ModelConfig` before this function
    runs; the final branch is a guard.
    """

    if config.dynamics == "mlp":
        return MLPDynamics(
            config.latent_dim,
            config.action_dim,
            config.hidden_dim,
            config.dynamics_hidden_layers,
            generator,
        )
    if config.dynamics == "rssm":
        raise NotImplementedError(RSSM_NOT_IMPLEMENTED)
    raise ValueError(f"unknown dynamics {config.dynamics!r}")
