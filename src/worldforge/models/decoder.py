"""Deterministic decoder from a latent code back to an observation.

Day 3 keeps this smaller than the encoder: one hidden width, fewer
layers by default. Prediction MSE later compares ``decode(z)`` with a
stored observation. This module does not compute that loss.
"""

from __future__ import annotations

import torch
from torch import nn

from worldforge.models.layers import describe_mlp, mlp
from worldforge.models.tensors import module_device, prepare_feature_tensor


class Decoder(nn.Module):
    """Map ``(..., latent_dim)`` to ``(..., obs_dim)``."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.net = mlp(latent_dim, obs_dim, hidden_dim, hidden_layers, generator)

    def describe(self) -> str:
        sizes = [self.latent_dim, *([self.hidden_dim] * self.hidden_layers), self.obs_dim]
        return describe_mlp(sizes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = prepare_feature_tensor(
            latent,
            name="latent",
            features=self.latent_dim,
            device=module_device(self),
        )
        return self.net(features)
