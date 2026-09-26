"""One-step training for the Day 3 world model.

The loop draws stored transitions with :func:`worldforge.data.sample_transition_batch`
and updates :class:`~worldforge.models.world.WorldModel` with Adam or SGD.
The objective is the MSE of the one-step observation prediction, plus an
optional reconstruction term. It writes the Day 3 checkpoint
(``config.json``, ``weights.pt``) and a JSONL metrics log. It does not roll
the latent state past one step, and it does not plan.

Training runs on CPU. For the duration of :func:`train_world_model` the
intra-op thread count is pinned to 1 and MKLDNN is disabled when it is
present, then both are restored. The same model seed and the same
:class:`TrainConfig` seed reproduce the same weights.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from worldforge.data import sample_transition_batch
from worldforge.data.dataset import TransitionBatch
from worldforge.models.checkpoint import save_checkpoint
from worldforge.models.world import WorldModel
from worldforge.schemas import Trajectory

TRAIN_RUN_FORMAT = "worldforge.train.v1"
METRICS_NAME = "metrics.jsonl"
RUN_NAME = "train.json"
OPTIMIZERS: tuple[str, ...] = ("adam", "sgd")
OptimizerName = Literal["adam", "sgd"]
_EPOCH_STRIDE = 1_000_003
_METRIC_KEYS = frozenset(
    {"epoch", "loss", "prediction_loss", "reconstruction_loss", "steps"}
)
_RUN_KEYS = frozenset(
    {
        "format",
        "optimizer",
        "epochs",
        "batch_size",
        "steps_per_epoch",
        "lr",
        "seed",
        "device",
        "reconstruction_weight",
        "n_episodes",
        "n_transitions",
        "final_train_loss",
        "data",
    }
)


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer and replay settings for one training run.

    ``seed`` selects replay batches. It does not initialize the network;
    pass the same integer to :class:`~worldforge.models.world.WorldModel`
    when a run should also fix the starting weights. ``steps_per_epoch``
    left as ``None`` means one coverage pass: ``ceil(transitions / batch_size)``.
    ``device`` is ``cpu`` only.
    """

    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    reconstruction_weight: float = 1.0
    steps_per_epoch: int | None = None
    optimizer: OptimizerName = "adam"

    def __post_init__(self) -> None:
        epochs = _require_int(self.epochs, label="epochs", minimum=1)
        batch_size = _require_int(self.batch_size, label="batch_size", minimum=1)
        seed = _require_int(self.seed, label="seed", minimum=0)
        lr = _require_finite(self.lr, label="lr")
        if lr <= 0:
            raise ValueError("lr must be > 0")
        weight = _require_finite(self.reconstruction_weight, label="reconstruction_weight")
        if weight < 0:
            raise ValueError("reconstruction_weight must be >= 0")
        if not isinstance(self.device, str) or self.device != "cpu":
            raise ValueError("device must be 'cpu'")
        if self.optimizer not in OPTIMIZERS:
            names = ", ".join(OPTIMIZERS)
            raise ValueError(f"optimizer must be one of: {names}")
        steps = self.steps_per_epoch
        if steps is not None:
            steps = _require_int(steps, label="steps_per_epoch", minimum=1)
            if steps > _EPOCH_STRIDE:
                raise ValueError(f"steps_per_epoch must be <= {_EPOCH_STRIDE}")
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "reconstruction_weight", weight)
        object.__setattr__(self, "steps_per_epoch", steps)


@dataclass(frozen=True)
class EpochLog:
    """Mean losses for one epoch. ``epoch`` is 1-based."""

    epoch: int
    loss: float
    prediction_loss: float
    reconstruction_loss: float
    steps: int


@dataclass(frozen=True)
class TrainResult:
    """A finished run: epoch logs plus the checkpoint directory."""

    config: TrainConfig
    steps_per_epoch: int
    n_episodes: int
    n_transitions: int
    epochs: tuple[EpochLog, ...]
    final_loss: float
    checkpoint: Path
    metrics_path: Path
    run_path: Path
    data_path: str | None

    @property
    def final_train_loss(self) -> float:
        return self.final_loss


def replay_batch_seed(seed: int, epoch: int, step: int) -> int:
    """Seed for the replay batch at zero-based ``epoch`` and ``step``.

    :func:`~worldforge.data.sample_transition_batch` consumes this value.
    Steps in one epoch stay below ``1_000_003``, so the next epoch does not
    reuse a batch seed.
    """

    seed = _require_int(seed, label="seed", minimum=0)
    epoch = _require_int(epoch, label="epoch", minimum=0)
    step = _require_int(step, label="step", minimum=0)
    if step >= _EPOCH_STRIDE:
        raise ValueError(f"step must be < {_EPOCH_STRIDE}")
    return seed + epoch * _EPOCH_STRIDE + step


def one_step_loss(
    model: WorldModel,
    observation: torch.Tensor,
    action: torch.Tensor,
    next_observation: torch.Tensor,
    *,
    reconstruction_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, prediction, and reconstruction losses.

    The prediction term is the MSE between ``predicted_observation`` and
    the stored next observation. The reconstruction term is the MSE between
    ``reconstructed`` and the current observation. ``reconstruction_weight``
    scales that second term. A weight of zero leaves it out of the backward
    graph. Both returned component tensors are the unweighted MSEs.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    weight = _require_finite(reconstruction_weight, label="reconstruction_weight")
    if weight < 0:
        raise ValueError("reconstruction_weight must be >= 0")
    prediction = model(observation, action)
    prediction_loss = F.mse_loss(prediction.predicted_observation, next_observation)
    reconstruction_loss = F.mse_loss(prediction.reconstructed, observation)
    if weight == 0.0:
        total = prediction_loss
    else:
        total = prediction_loss + weight * reconstruction_loss
    return total, prediction_loss, reconstruction_loss


def train_world_model(
    model: WorldModel,
    trajectories: Sequence[Trajectory],
    config: TrainConfig | None = None,
    *,
    out: str | Path,
    data_path: str | Path | None = None,
) -> TrainResult:
    """Fit ``model`` on stored transitions and write a checkpoint directory.

    ``out`` receives ``config.json`` and ``weights.pt`` from
    :func:`~worldforge.models.checkpoint.save_checkpoint`, plus
    ``metrics.jsonl`` (one object per epoch) and ``train.json`` (the run
    settings and ``final_train_loss``). Optimizer state is not stored.
    ``final_loss`` is the last epoch's mean step loss. Each step loss is the
    batch-mean objective, taken before that step's optimizer update.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    if config is None:
        config = TrainConfig()
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    episodes = _require_trajectories(trajectories)
    n_transitions = _check_feature_widths(model, episodes)
    steps_per_epoch = _resolve_steps(n_transitions, config)
    directory = Path(out)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {directory}")
    recorded_data = None if data_path is None else str(data_path)

    device = torch.device(config.device)
    model.to(device)
    model.train()
    optimizer = _build_optimizer(model, config)
    logs: list[EpochLog] = []
    with _cpu_training_context(config.seed):
        for epoch in range(config.epochs):
            logs.append(
                _run_epoch(
                    model,
                    optimizer,
                    episodes,
                    config,
                    device=device,
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                )
            )

    result = TrainResult(
        config=config,
        steps_per_epoch=steps_per_epoch,
        n_episodes=len(episodes),
        n_transitions=n_transitions,
        epochs=tuple(logs),
        final_loss=logs[-1].loss,
        checkpoint=directory,
        metrics_path=directory / METRICS_NAME,
        run_path=directory / RUN_NAME,
        data_path=recorded_data,
    )
    directory.mkdir(parents=True, exist_ok=True)
    _write_metrics(result.metrics_path, result.epochs)
    _write_run(result.run_path, result)
    save_checkpoint(directory, model)
    return result


def format_train_report(result: TrainResult) -> str:
    """Plain-text summary for ``worldforge train``, including the final loss."""

    data = result.data_path if result.data_path is not None else "memory"
    lines = [
        f"data: {data}",
        f"episodes: {result.n_episodes}",
        f"transitions: {result.n_transitions}",
        f"optimizer: {result.config.optimizer}",
        f"epochs: {result.config.epochs}",
        f"batch_size: {result.config.batch_size}",
        f"steps_per_epoch: {result.steps_per_epoch}",
        f"lr: {_format_float(result.config.lr)}",
        f"seed: {result.config.seed}",
        f"device: {result.config.device}",
        f"reconstruction_weight: {_format_float(result.config.reconstruction_weight)}",
    ]
    for row in result.epochs:
        lines.append(
            f"epoch: {row.epoch} loss: {_format_float(row.loss)} "
            f"prediction_loss: {_format_float(row.prediction_loss)} "
            f"reconstruction_loss: {_format_float(row.reconstruction_loss)} "
            f"steps: {row.steps}"
        )
    lines.append(f"checkpoint: {result.checkpoint}")
    lines.append(f"metrics: {result.metrics_path}")
    lines.append(f"final_train_loss: {_format_float(result.final_loss)}")
    return "\n".join(lines)


def read_metrics(path: str | Path) -> tuple[EpochLog, ...]:
    """Read ``metrics.jsonl`` written by :func:`train_world_model`."""

    location = Path(path)
    text = location.read_text(encoding="utf-8")
    if text == "":
        raise ValueError(f"metrics file is empty: {location}")
    rows: list[EpochLog] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metrics line {line_number} is not valid JSON: {exc}") from exc
        rows.append(_epoch_from_payload(payload, line_number))
    if not rows:
        raise ValueError(f"metrics file has no epochs: {location}")
    for index, row in enumerate(rows, start=1):
        if row.epoch != index:
            raise ValueError(f"metrics epochs must be 1..{len(rows)} in order")
    return tuple(rows)


def _run_epoch(
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[Trajectory],
    config: TrainConfig,
    *,
    device: torch.device,
    epoch: int,
    steps_per_epoch: int,
) -> EpochLog:
    loss_sum = 0.0
    prediction_sum = 0.0
    reconstruction_sum = 0.0
    for step in range(steps_per_epoch):
        batch = sample_transition_batch(
            episodes,
            batch_size=config.batch_size,
            seed=replay_batch_seed(config.seed, epoch, step),
        )
        observation, action, nxt = _batch_tensors(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, prediction_loss, reconstruction_loss = one_step_loss(
            model,
            observation,
            action,
            nxt,
            reconstruction_weight=config.reconstruction_weight,
        )
        loss_value = _finite_float(loss, label=f"loss at epoch {epoch + 1} step {step + 1}")
        prediction_value = _finite_float(prediction_loss, label="prediction loss")
        reconstruction_value = _finite_float(reconstruction_loss, label="reconstruction loss")
        loss.backward()
        optimizer.step()
        loss_sum += loss_value
        prediction_sum += prediction_value
        reconstruction_sum += reconstruction_value
    count = float(steps_per_epoch)
    return EpochLog(
        epoch=epoch + 1,
        loss=loss_sum / count,
        prediction_loss=prediction_sum / count,
        reconstruction_loss=reconstruction_sum / count,
        steps=steps_per_epoch,
    )


def _batch_tensors(
    batch: TransitionBatch, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observation = torch.tensor(batch.observations, dtype=torch.float32, device=device)
    action = torch.tensor(batch.actions, dtype=torch.float32, device=device)
    nxt = torch.tensor(batch.next_observations, dtype=torch.float32, device=device)
    return observation, action, nxt


def _build_optimizer(model: WorldModel, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr)
    names = ", ".join(OPTIMIZERS)
    raise ValueError(f"optimizer must be one of: {names}")


def _resolve_steps(n_transitions: int, config: TrainConfig) -> int:
    if config.batch_size > n_transitions:
        raise ValueError(f"batch_size {config.batch_size} exceeds {n_transitions} transitions")
    if config.steps_per_epoch is None:
        return math.ceil(n_transitions / config.batch_size)
    return config.steps_per_epoch


def _require_trajectories(trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    if isinstance(trajectories, (str, bytes)) or not isinstance(trajectories, Sequence):
        raise TypeError("trajectories must be a sequence of Trajectory")
    episodes = list(trajectories)
    if len(episodes) < 1:
        raise ValueError("trajectories must be non-empty")
    for episode in episodes:
        if not isinstance(episode, Trajectory):
            raise TypeError("trajectories must contain Trajectory values")
    return episodes


def _check_feature_widths(model: WorldModel, episodes: Sequence[Trajectory]) -> int:
    obs_dim = model.config.obs_dim
    action_dim = model.config.action_dim
    n_transitions = 0
    for episode in episodes:
        for transition in episode.transitions:
            n_transitions += 1
            observation = transition.observation.values
            nxt = transition.next_observation.values
            action = transition.action.values
            if len(observation) != obs_dim:
                raise ValueError(
                    f"observation length {len(observation)} does not match obs_dim {obs_dim}"
                )
            if len(nxt) != obs_dim:
                raise ValueError(
                    f"next observation length {len(nxt)} does not match obs_dim {obs_dim}"
                )
            if len(action) != action_dim:
                raise ValueError(
                    f"action length {len(action)} does not match action_dim {action_dim}"
                )
    if n_transitions < 1:
        raise ValueError("trajectories have no transitions")
    return n_transitions


@contextmanager
def _cpu_training_context(seed: int) -> Iterator[None]:
    """Pin CPU kernels for one run, then restore threads, MKLDNN, and RNG."""

    threads = torch.get_num_threads()
    rng_state = torch.get_rng_state().clone()
    mkldnn = _mkldnn_enabled()
    try:
        torch.set_num_threads(1)
        if mkldnn is not None:
            _set_mkldnn(False)
        torch.manual_seed(seed)
        yield
    finally:
        if mkldnn is not None:
            _set_mkldnn(mkldnn)
        torch.set_rng_state(rng_state)
        torch.set_num_threads(threads)


def _mkldnn_enabled() -> bool | None:
    backend = getattr(torch.backends, "mkldnn", None)
    if backend is None or not backend.is_available():
        return None
    return bool(backend.enabled)


def _set_mkldnn(enabled: bool) -> None:
    backend = getattr(torch.backends, "mkldnn", None)
    if backend is not None and backend.is_available():
        backend.enabled = enabled


def _finite_float(value: torch.Tensor, *, label: str) -> float:
    number = float(value.detach().item())
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _write_metrics(path: Path, rows: Sequence[EpochLog]) -> None:
    lines = [json.dumps(_epoch_payload(row), allow_nan=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run(path: Path, result: TrainResult) -> None:
    payload = {
        "format": TRAIN_RUN_FORMAT,
        "optimizer": result.config.optimizer,
        "epochs": result.config.epochs,
        "batch_size": result.config.batch_size,
        "steps_per_epoch": result.steps_per_epoch,
        "lr": result.config.lr,
        "seed": result.config.seed,
        "device": result.config.device,
        "reconstruction_weight": result.config.reconstruction_weight,
        "n_episodes": result.n_episodes,
        "n_transitions": result.n_transitions,
        "final_train_loss": result.final_loss,
        "data": result.data_path,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _epoch_payload(row: EpochLog) -> dict[str, object]:
    return {
        "epoch": row.epoch,
        "loss": row.loss,
        "prediction_loss": row.prediction_loss,
        "reconstruction_loss": row.reconstruction_loss,
        "steps": row.steps,
    }


def _epoch_from_payload(payload: object, line_number: int) -> EpochLog:
    if not isinstance(payload, dict) or set(payload) != _METRIC_KEYS:
        raise ValueError(
            f"metrics line {line_number} must contain epoch, loss, prediction_loss, "
            "reconstruction_loss, and steps"
        )
    return EpochLog(
        epoch=_require_int(payload["epoch"], label="epoch", minimum=1),
        loss=_require_finite(payload["loss"], label="loss"),
        prediction_loss=_require_finite(payload["prediction_loss"], label="prediction_loss"),
        reconstruction_loss=_require_finite(
            payload["reconstruction_loss"], label="reconstruction_loss"
        ),
        steps=_require_int(payload["steps"], label="steps", minimum=1),
    )


def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def read_run(path: str | Path) -> dict[str, object]:
    """Read ``train.json`` written by :func:`train_world_model`."""

    location = Path(path)
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"train record is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _RUN_KEYS:
        raise ValueError("train.json must contain the Day 4 run fields")
    if payload["format"] != TRAIN_RUN_FORMAT:
        raise ValueError(
            f"unsupported train record format {payload['format']!r}; expected {TRAIN_RUN_FORMAT!r}"
        )
    _require_finite(payload["final_train_loss"], label="final_train_loss")
    _require_finite(payload["lr"], label="lr")
    _require_finite(payload["reconstruction_weight"], label="reconstruction_weight")
    return payload


def _format_float(value: float) -> str:
    return format(float(value), ".8g")
