"""Latent world model for Day 3.

:class:`~worldforge.models.config.ModelConfig` does not need PyTorch.
Importing the classes below loads the optional ``ml`` extra. The package
initializer stays importable without that extra so ``import worldforge``
and the dataset commands do not require torch. Attribute access below is
lazy for the same reason.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "Decoder",
    "Encoder",
    "MLPDynamics",
    "ModelConfig",
    "StepPrediction",
    "WorldModel",
    "build_dynamics",
    "format_forward_smoke",
    "format_model_info",
    "load_checkpoint",
    "save_checkpoint",
]

_LAZY: dict[str, tuple[str, str]] = {
    "ModelConfig": ("worldforge.models.config", "ModelConfig"),
    "Encoder": ("worldforge.models.encoder", "Encoder"),
    "Decoder": ("worldforge.models.decoder", "Decoder"),
    "MLPDynamics": ("worldforge.models.dynamics", "MLPDynamics"),
    "build_dynamics": ("worldforge.models.dynamics", "build_dynamics"),
    "StepPrediction": ("worldforge.models.world", "StepPrediction"),
    "WorldModel": ("worldforge.models.world", "WorldModel"),
    "load_checkpoint": ("worldforge.models.checkpoint", "load_checkpoint"),
    "save_checkpoint": ("worldforge.models.checkpoint", "save_checkpoint"),
    "format_model_info": ("worldforge.models.summary", "format_model_info"),
    "format_forward_smoke": ("worldforge.models.summary", "format_forward_smoke"),
}

if TYPE_CHECKING:
    from worldforge.models.checkpoint import load_checkpoint, save_checkpoint
    from worldforge.models.config import ModelConfig
    from worldforge.models.decoder import Decoder
    from worldforge.models.dynamics import MLPDynamics, build_dynamics
    from worldforge.models.encoder import Encoder
    from worldforge.models.summary import format_forward_smoke, format_model_info
    from worldforge.models.world import StepPrediction, WorldModel


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
