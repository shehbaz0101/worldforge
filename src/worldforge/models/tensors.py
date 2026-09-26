"""Tensor checks shared by the encoder, dynamics, and decoder.

Inputs are floating tensors whose last dimension is the feature width.
Leading dimensions are a batch, possibly empty. Values are cast to
float32 on the module device. This is the Day 3 forward path; it does
not sample noise and it does not step an optimizer.
"""

from __future__ import annotations

import torch
from torch import nn


def module_device(module: nn.Module) -> torch.device:
    try:
        parameter = next(module.parameters())
    except StopIteration as exc:
        raise RuntimeError(f"{type(module).__name__} has no parameters") from exc
    return parameter.device


def prepare_feature_tensor(
    value: object,
    *,
    name: str,
    features: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``value`` as a float32 tensor with last dimension ``features``."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 1:
        raise ValueError(f"{name} must have shape (..., {features})")
    if value.shape[-1] != features:
        raise ValueError(
            f"{name} feature dimension must be {features}, got shape {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    return value.to(dtype=torch.float32, device=device)


def require_same_leading(left: torch.Tensor, right: torch.Tensor, *, left_name: str, right_name: str) -> None:
    """Raise if ``left`` and ``right`` disagree on every axis but the last."""

    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError(
            f"{left_name} and {right_name} leading shapes must match, "
            f"got {tuple(left.shape[:-1])} and {tuple(right.shape[:-1])}"
        )
