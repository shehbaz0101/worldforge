"""World model wrapper: encode, step, decode, and one-step predict.

``forward`` is a pure tensor map. It does not sample a batch, zero a
gradient, or call an optimizer. Day 4 can score
``reconstructed`` against the current observation and
``predicted_observation`` against the next observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from worldforge.models.config import ModelConfig
from worldforge.models.decoder import Decoder
from worldforge.models.dynamics import MLPDynamics, build_dynamics
from worldforge.models.encoder import Encoder


def _generator(seed: int) -> torch.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@dataclass(frozen=True)
class StepPrediction:
    """Tensors from one encode / step / decode pass.

    ``latent`` is ``z_t``. ``next_latent`` is ``z_{t+1}``.
    ``reconstructed`` is ``decode(z_t)``. ``predicted_observation`` is
    ``decode(z_{t+1})``, the one-step observation prediction.
    """

    latent: torch.Tensor
    next_latent: torch.Tensor
    reconstructed: torch.Tensor
    predicted_observation: torch.Tensor


class WorldModel(nn.Module):
    """Encoder, baseline dynamics, and decoder for one :class:`ModelConfig`.

    Parameters are created on CPU in float32. ``seed`` initializes every
    linear map from a private generator, in order: encoder, dynamics,
    decoder. The same config and seed rebuild the same weights.
    ``train()`` and ``eval()`` match: there is no dropout and no
    normalization.
    """

    def __init__(self, config: ModelConfig | None = None, *, seed: int = 0) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config
        generator = _generator(seed)
        self.encoder = Encoder(
            config.obs_dim,
            config.latent_dim,
            config.hidden_dim,
            config.encoder_hidden_layers,
            generator,
        )
        self.dynamics: MLPDynamics = build_dynamics(config, generator)
        self.decoder = Decoder(
            config.latent_dim,
            config.obs_dim,
            config.hidden_dim,
            config.decoder_hidden_layers,
            generator,
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Map an observation to ``z_t``."""

        return self.encoder(observation)

    def step(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Map ``(z_t, a_t)`` to ``z_{t+1}`` with the Day 3 baseline."""

        return self.dynamics(latent, action)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Map a latent code to an observation vector."""

        return self.decoder(latent)

    def predict(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One-step observation prediction, ``decode(step(encode(obs), action))``."""

        return self.decode(self.step(self.encode(observation), action))

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> StepPrediction:
        """Encode, step, and decode the current and next latent codes."""

        latent = self.encode(observation)
        next_latent = self.step(latent, action)
        return StepPrediction(
            latent=latent,
            next_latent=next_latent,
            reconstructed=self.decode(latent),
            predicted_observation=self.decode(next_latent),
        )
