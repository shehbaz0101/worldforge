"""Small float32 linear layers for the Day 3 MLPs.

Weights are drawn from a caller-owned :class:`torch.Generator`. ``nn.Linear``
also samples the global generator during construction; those values are
overwritten and the global CPU generator is restored. Building a model
therefore depends on ``seed`` and does not move the caller's RNG.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class Affine(nn.Linear):
    """``nn.Linear`` initialized from ``generator`` on CPU in float32.

    The initialization matches PyTorch's default linear layer: Kaiming
    uniform weights with ``a = sqrt(5)`` and a uniform bias of width
    ``1 / sqrt(in_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        generator: torch.Generator,
    ) -> None:
        state = torch.get_rng_state()
        try:
            super().__init__(in_features, out_features, device="cpu", dtype=torch.float32)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
            if self.bias is None:
                raise RuntimeError("Affine expects a bias parameter")
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound, generator=generator)
        finally:
            torch.set_rng_state(state)


def mlp(
    in_features: int,
    out_features: int,
    hidden_dim: int,
    hidden_layers: int,
    generator: torch.Generator,
) -> nn.Sequential:
    """ReLU MLP. The last map is linear, with no activation."""

    if hidden_layers < 1:
        raise ValueError("hidden_layers must be >= 1")
    sizes = [in_features, *([hidden_dim] * hidden_layers), out_features]
    modules: list[nn.Module] = []
    pairs = _consecutive(sizes)
    last = len(pairs) - 1
    for index, (fan_in, fan_out) in enumerate(pairs):
        modules.append(Affine(fan_in, fan_out, generator))
        if index < last:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def describe_mlp(sizes: Sequence[int], *, residual: bool = False) -> str:
    """One-line shape string for ``model-info``."""

    pairs = _consecutive(sizes)
    parts: list[str] = []
    last = len(pairs) - 1
    for index, (fan_in, fan_out) in enumerate(pairs):
        parts.append(f"Linear({fan_in}, {fan_out})")
        if index < last:
            parts.append("ReLU")
    text = " -> ".join(parts)
    if residual:
        return f"residual {text}"
    return text


def _consecutive(sizes: Sequence[int]) -> list[tuple[int, int]]:
    return [(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)]
