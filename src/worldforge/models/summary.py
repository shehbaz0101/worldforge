"""Plain-text reports for ``model-info`` and ``forward-smoke``.

These strings describe a forward pass. They are not a training log.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from worldforge.models.config import MODEL_REPORT_FORMAT
from worldforge.models.world import StepPrediction, WorldModel


def count_parameters(module: torch.nn.Module) -> int:
    """Number of trainable and frozen elements. Day 3 freezes nothing."""

    return sum(parameter.numel() for parameter in module.parameters())


def format_vector(values: Sequence[float]) -> str:
    """Space-separated decimals for CLI smoke output."""

    return " ".join(format(float(value), ".8g") for value in values)


def format_model_info(model: WorldModel) -> str:
    """Architecture sizes and parameter counts. Independent of the init seed."""

    config = model.config
    encoder_n = count_parameters(model.encoder)
    dynamics_n = count_parameters(model.dynamics)
    decoder_n = count_parameters(model.decoder)
    total = count_parameters(model)
    lines = [
        f"format: {MODEL_REPORT_FORMAT}",
        f"dynamics: {config.dynamics}",
        "baseline: deterministic residual MLP",
        f"obs_dim: {config.obs_dim}",
        f"action_dim: {config.action_dim}",
        f"latent_dim: {config.latent_dim}",
        f"hidden_dim: {config.hidden_dim}",
        f"encoder_hidden_layers: {config.encoder_hidden_layers}",
        f"dynamics_hidden_layers: {config.dynamics_hidden_layers}",
        f"decoder_hidden_layers: {config.decoder_hidden_layers}",
        f"encoder: {model.encoder.describe()}",
        f"dynamics_net: {model.dynamics.describe()}",
        f"decoder: {model.decoder.describe()}",
        f"encoder_parameters: {encoder_n}",
        f"dynamics_parameters: {dynamics_n}",
        f"decoder_parameters: {decoder_n}",
        f"parameters: {total}",
    ]
    return "\n".join(lines)


def format_forward_smoke(
    *,
    source: str,
    seed: int,
    episode: int | None,
    transition: int | None,
    model: WorldModel,
    observation: Sequence[float],
    action: Sequence[float],
    next_observation: Sequence[float] | None,
    prediction: StepPrediction,
) -> str:
    """Shapes and vectors from one encode / step / decode, without a loss."""

    finite = _finite(prediction)
    lines = [
        f"source: {source}",
        f"seed: {seed}",
        f"dynamics: {model.config.dynamics}",
        f"latent_dim: {model.config.latent_dim}",
    ]
    if episode is not None:
        lines.append(f"episode: {episode}")
    if transition is not None:
        lines.append(f"transition: {transition}")
    lines.extend(
        [
            f"latent_shape: {tuple(prediction.latent.shape)}",
            f"next_latent_shape: {tuple(prediction.next_latent.shape)}",
            f"reconstructed_shape: {tuple(prediction.reconstructed.shape)}",
            f"predicted_shape: {tuple(prediction.predicted_observation.shape)}",
            f"finite: {str(finite).lower()}",
            f"observation: {format_vector(observation)}",
            f"action: {format_vector(action)}",
        ]
    )
    if next_observation is not None:
        lines.append(f"next_observation: {format_vector(next_observation)}")
    lines.extend(
        [
            f"reconstructed: {format_vector(prediction.reconstructed.detach().cpu().tolist())}",
            f"predicted: {format_vector(prediction.predicted_observation.detach().cpu().tolist())}",
        ]
    )
    return "\n".join(lines)


def _finite(prediction: StepPrediction) -> bool:
    tensors = (
        prediction.latent,
        prediction.next_latent,
        prediction.reconstructed,
        prediction.predicted_observation,
    )
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
