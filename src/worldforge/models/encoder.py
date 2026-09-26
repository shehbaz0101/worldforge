"""Deterministic encoder from an observation vector to a latent code.

For the default environment the observation is ``(prey, predator)``.
The map is a small ReLU MLP. There is no posterior and no noise: a
stochastic encoder would belong to the reserved RSSM variant.
"""

from __future__ import annotations

import torch
from torch import nn

from worldforge.models.layers import describe_mlp, mlp
from worldforge.models.tensors import module_device, prepare_feature_tensor


class Encoder(nn.Module):
    """Map ``(..., obs_dim)`` to ``(..., latent_dim)``."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.net = mlp(obs_dim, latent_dim, hidden_dim, hidden_layers, generator)

    def describe(self) -> str:
        sizes = [self.obs_dim, *([self.hidden_dim] * self.hidden_layers), self.latent_dim]
        return describe_mlp(sizes)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = prepare_feature_tensor(
            observation,
            name="observation",
            features=self.obs_dim,
            device=module_device(self),
        )
        return self.net(features)
