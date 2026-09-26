"""Request and response models for the HTTP API.

These models do not import PyTorch. A checkpoint path and a model spec
are separate: when ``checkpoint`` is set, the architecture fields are
ignored and the directory's ``config.json`` wins. Inline trajectories
use the same :class:`~worldforge.schemas.trajectory.Trajectory` document
as an episode JSON file.

``POST /train`` is a short run. The CLI trainer is not capped.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from worldforge.models.config import (
    DEFAULT_ACTION_DIM,
    DEFAULT_DECODER_HIDDEN_LAYERS,
    DEFAULT_DYNAMICS,
    DEFAULT_DYNAMICS_HIDDEN_LAYERS,
    DEFAULT_ENCODER_HIDDEN_LAYERS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_OBS_DIM,
    DynamicsName,
    ModelConfig,
)
from worldforge.schemas import Trajectory

HTTP_MAX_EPOCHS = 5
HTTP_MAX_STEPS_PER_EPOCH = 8


def _strip_path(value: str | None, info: ValidationInfo) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        label = info.field_name or "path"
        raise ValueError(f"{label} must not be empty")
    return stripped


def _finite_values(values: list[float], *, label: str) -> list[float]:
    if not values:
        raise ValueError(f"{label} must be non-empty")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{label} values must be finite")
    return values


class HealthResponse(BaseModel):
    """Liveness payload. No model load and no file read."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    version: str


class ModelSpec(BaseModel):
    """Architecture used when ``checkpoint`` is omitted.

    Defaults match the Lotka-Volterra sandbox. Hidden-layer counts are
    the same fields a checkpoint stores. They are ignored when a
    checkpoint directory is loaded.
    """

    model_config = ConfigDict(extra="forbid")

    obs_dim: int = Field(default=DEFAULT_OBS_DIM, ge=1)
    action_dim: int = Field(default=DEFAULT_ACTION_DIM, ge=1)
    latent_dim: int = Field(default=DEFAULT_LATENT_DIM, ge=1)
    hidden_dim: int = Field(default=DEFAULT_HIDDEN_DIM, ge=1)
    encoder_hidden_layers: int = Field(default=DEFAULT_ENCODER_HIDDEN_LAYERS, ge=1)
    dynamics_hidden_layers: int = Field(default=DEFAULT_DYNAMICS_HIDDEN_LAYERS, ge=1)
    decoder_hidden_layers: int = Field(default=DEFAULT_DECODER_HIDDEN_LAYERS, ge=1)
    dynamics: DynamicsName = DEFAULT_DYNAMICS

    def to_config(self) -> ModelConfig:
        """Build the Day 3 config from these fields."""

        return ModelConfig(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            encoder_hidden_layers=self.encoder_hidden_layers,
            dynamics_hidden_layers=self.dynamics_hidden_layers,
            decoder_hidden_layers=self.decoder_hidden_layers,
            dynamics=self.dynamics,
        )


class PredictRequest(ModelSpec):
    """Open-loop score. One corpus path or a small inline episode list."""

    data: str | None = None
    trajectories: list[Trajectory] | None = None
    checkpoint: str | None = None
    seed: int = Field(default=0, ge=0)
    horizon: int | None = Field(default=None, ge=1)

    @field_validator("data", "checkpoint")
    @classmethod
    def _paths(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _strip_path(value, info)

    @field_validator("trajectories")
    @classmethod
    def _episodes(cls, value: list[Trajectory] | None) -> list[Trajectory] | None:
        if value is not None and len(value) < 1:
            raise ValueError("trajectories must be non-empty")
        return value

    @model_validator(mode="after")
    def _one_source(self) -> PredictRequest:
        if self.data is not None and self.trajectories is not None:
            raise ValueError("provide a data path or inline trajectories, not both")
        if self.data is None and self.trajectories is None:
            raise ValueError("provide a data path or inline trajectories")
        return self


class PredictResponse(BaseModel):
    """``worldforge.predict.v1``. The same object the eval commands write."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["worldforge.predict.v1"]
    horizons: list[int]
    mse_by_h: list[float]
    mae_by_h: list[float]
    residual_std_by_h: list[float]
    n_episodes: int
    n_episodes_by_h: list[int]
    checkpoint: str | None
    data: str | None


class PlanRequest(ModelSpec):
    """One CEM action sequence from an observation.

    ``observation`` and ``data`` are optional and mutually exclusive.
    With neither, the start is the regulation target (``(1, 1)`` for the
    default width). ``data`` uses the first observation of ``episode``.
    ``seed`` draws the CEM samples and, when ``checkpoint`` is omitted,
    initializes the weights.
    """

    observation: list[float] | None = None
    data: str | None = None
    episode: int = Field(default=0, ge=0)
    checkpoint: str | None = None
    seed: int = Field(default=0, ge=0)
    horizon: int = Field(default=3, ge=1)
    n_samples: int = Field(default=8, ge=1)
    n_iterations: int = Field(default=2, ge=1)
    elite_fraction: float = Field(default=0.25, gt=0, le=1)

    @field_validator("observation")
    @classmethod
    def _observation(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        return _finite_values(value, label="observation")

    @field_validator("data", "checkpoint")
    @classmethod
    def _paths(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _strip_path(value, info)

    @model_validator(mode="after")
    def _one_start(self) -> PlanRequest:
        if self.observation is not None and self.data is not None:
            raise ValueError("provide an observation or a data path, not both")
        return self


class ActionSequenceResponse(BaseModel):
    """``worldforge.action_sequence.v1``. One clipped sequence."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["worldforge.action_sequence.v1"]
    objective: Literal["regulation_l1"]
    observation: list[float]
    actions: list[list[float]]
    predicted_return: float
    horizon: int
    n_samples: int
    n_iterations: int
    elite_fraction: float
    seed: int
    checkpoint: str | None


class EvalPlanRequest(ModelSpec):
    """Closed-loop regret. ``n_steps`` is the environment horizon.

    ``seed`` draws CEM samples and, without a checkpoint, the weights.
    ``env_seed`` resets the environment and seeds the random baseline.
    Defaults match ``worldforge eval-plan`` and stay short.
    """

    checkpoint: str | None = None
    seed: int = Field(default=0, ge=0)
    env_seed: int = 0
    n_steps: int = Field(default=4, ge=1)
    horizon: int = Field(default=3, ge=1)
    n_samples: int = Field(default=8, ge=1)
    n_iterations: int = Field(default=2, ge=1)
    elite_fraction: float = Field(default=0.25, gt=0, le=1)

    @field_validator("checkpoint")
    @classmethod
    def _checkpoint(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _strip_path(value, info)

    @field_validator("env_seed")
    @classmethod
    def _env_seed(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("env_seed must be an int")
        return value


class PlanReportResponse(BaseModel):
    """``worldforge.plan.v1``. Closed-loop returns and regret."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["worldforge.plan.v1"]
    objective: Literal["regulation_l1"]
    regret_definition: Literal["baseline_best - planner_return"]
    env_id: str
    seed: int
    cem_seed: int
    n_steps: int
    planner_steps: int
    horizon: int
    n_samples: int
    n_iterations: int
    elite_fraction: float
    initial_observation: list[float]
    planner_return: float
    zero_return: float
    random_return: float
    baseline_best: float
    regret: float
    checkpoint: str | None
    actions: list[list[float]]


class TrainRequest(ModelSpec):
    """A short CPU fit. Defaults are one step, not the CLI coverage pass.

    ``epochs`` is at most :data:`HTTP_MAX_EPOCHS`. ``steps_per_epoch`` is
    at most :data:`HTTP_MAX_STEPS_PER_EPOCH`. ``out`` omitted writes a new
    temporary directory and returns that path. ``device`` is ``cpu`` only.
    """

    data: str | None = None
    trajectories: list[Trajectory] | None = None
    out: str | None = None
    epochs: int = Field(default=1, ge=1, le=HTTP_MAX_EPOCHS)
    batch_size: int = Field(default=1, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    seed: int = Field(default=0, ge=0)
    device: Literal["cpu"] = "cpu"
    reconstruction_weight: float = Field(default=1.0, ge=0)
    steps_per_epoch: int = Field(default=1, ge=1, le=HTTP_MAX_STEPS_PER_EPOCH)
    optimizer: Literal["adam", "sgd"] = "adam"

    @field_validator("data", "out")
    @classmethod
    def _paths(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _strip_path(value, info)

    @field_validator("trajectories")
    @classmethod
    def _episodes(cls, value: list[Trajectory] | None) -> list[Trajectory] | None:
        if value is not None and len(value) < 1:
            raise ValueError("trajectories must be non-empty")
        return value

    @field_validator("lr", "reconstruction_weight")
    @classmethod
    def _finite_scalars(cls, value: float, info: ValidationInfo) -> float:
        if not math.isfinite(value):
            label = info.field_name or "value"
            raise ValueError(f"{label} must be finite")
        return value

    @model_validator(mode="after")
    def _one_source(self) -> TrainRequest:
        if self.data is not None and self.trajectories is not None:
            raise ValueError("provide a data path or inline trajectories, not both")
        if self.data is None and self.trajectories is None:
            raise ValueError("provide a data path or inline trajectories")
        return self


class TrainResponse(BaseModel):
    """Final loss and the checkpoint directory from a short run."""

    model_config = ConfigDict(extra="forbid")

    final_train_loss: float
    checkpoint: str
    n_episodes: int
    n_transitions: int
    epochs: int
    steps_per_epoch: int
