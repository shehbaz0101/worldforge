"""Architecture config for the Day 3 latent world model.

This module does not import PyTorch. The command line can read the
defaults without the optional ``ml`` extra. Building a module from the
config does need that extra.

Defaults match the Lotka-Volterra sandbox: a 2-vector observation and a
2-vector dose. ``latent_dim`` is the width of ``z``. ``dynamics="mlp"``
is the deterministic residual baseline. ``dynamics="rssm"`` is accepted
here so a checkpoint can name it, and rejected when the network is built.
That variant is a later stochastic model, not part of Day 3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_OBS_DIM = 2
DEFAULT_ACTION_DIM = 2
DEFAULT_LATENT_DIM = 8
DEFAULT_HIDDEN_DIM = 64
DEFAULT_ENCODER_HIDDEN_LAYERS = 2
DEFAULT_DYNAMICS_HIDDEN_LAYERS = 2
DEFAULT_DECODER_HIDDEN_LAYERS = 1
DEFAULT_DYNAMICS = "mlp"

DynamicsName = Literal["mlp", "rssm"]
IMPLEMENTED_DYNAMICS: frozenset[DynamicsName] = frozenset({"mlp"})
RESERVED_DYNAMICS: frozenset[DynamicsName] = frozenset({"rssm"})
KNOWN_DYNAMICS: tuple[DynamicsName, ...] = ("mlp", "rssm")

CHECKPOINT_FORMAT = "worldforge.checkpoint.v1"
MODEL_REPORT_FORMAT = "worldforge.model.v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_StrictModel):
    """Sizes and the dynamics variant for one :class:`WorldModel`.

    Hidden-layer counts are part of the checkpoint. They fix the MLP depth
    so a saved config rebuilds the same modules.
    """

    obs_dim: int = Field(default=DEFAULT_OBS_DIM, ge=1)
    action_dim: int = Field(default=DEFAULT_ACTION_DIM, ge=1)
    latent_dim: int = Field(default=DEFAULT_LATENT_DIM, ge=1)
    hidden_dim: int = Field(default=DEFAULT_HIDDEN_DIM, ge=1)
    encoder_hidden_layers: int = Field(default=DEFAULT_ENCODER_HIDDEN_LAYERS, ge=1)
    dynamics_hidden_layers: int = Field(default=DEFAULT_DYNAMICS_HIDDEN_LAYERS, ge=1)
    decoder_hidden_layers: int = Field(default=DEFAULT_DECODER_HIDDEN_LAYERS, ge=1)
    dynamics: DynamicsName = DEFAULT_DYNAMICS
