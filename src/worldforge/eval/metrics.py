"""Open-loop prediction metrics.

Each episode is rolled from its first observation under the stored actions.
The score at horizon ``h`` compares the decoded latent state with the stored
observation ``h`` steps later. The aggregate is the mean over episodes that
reach that horizon, and over observation features. A per-horizon residual
standard deviation is included as a calibration stub. This module does not
plan and it does not train.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from worldforge.eval.rollout import require_int, rollout_latent, trajectory_window
from worldforge.models.world import WorldModel
from worldforge.schemas import Trajectory

PREDICT_REPORT_FORMAT = "worldforge.predict.v1"
_REPORT_KEYS = frozenset(
    {
        "format",
        "horizons",
        "mse_by_h",
        "mae_by_h",
        "residual_std_by_h",
        "n_episodes",
        "n_episodes_by_h",
        "checkpoint",
        "data",
    }
)


@dataclass(frozen=True)
class PredictReport:
    """Per-horizon open-loop scores.

    ``horizons`` is 1-based and contiguous. ``mse_by_h[i]`` is the mean
    squared error at ``horizons[i]``. ``n_episodes`` counts episodes that
    contributed at least one horizon. ``n_episodes_by_h`` counts how many
    of those reached each horizon. ``checkpoint`` and ``data`` are optional
    path notes for the CLI report.
    """

    horizons: tuple[int, ...]
    mse_by_h: tuple[float, ...]
    mae_by_h: tuple[float, ...]
    residual_std_by_h: tuple[float, ...]
    n_episodes: int
    n_episodes_by_h: tuple[int, ...]
    checkpoint: str | None = None
    data: str | None = None


def open_loop_metrics(
    model: WorldModel,
    trajectories: Sequence[Trajectory],
    *,
    horizon: int | None = None,
    checkpoint: str | None = None,
    data: str | None = None,
) -> PredictReport:
    """Score ``model`` on stored episodes and return a prediction report.

    ``horizon`` left as ``None`` scores each episode to its last transition.
    An explicit horizon scores ``1 .. horizon`` and raises when no episode
    is that long. Shorter episodes still contribute to the horizons they
    reach. The model is set to eval for the rollout and restored afterward.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    if isinstance(trajectories, (str, bytes)) or not isinstance(trajectories, Sequence):
        raise TypeError("trajectories must be a sequence of Trajectory")
    if len(trajectories) < 1:
        raise ValueError("trajectories must be non-empty")
    requested = None if horizon is None else require_int(horizon, label="horizon", minimum=1)
    checkpoint_note = _optional_path(checkpoint, label="checkpoint")
    data_note = _optional_path(data, label="data")

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for episode in trajectories:
                if not isinstance(episode, Trajectory):
                    raise TypeError("trajectories must contain Trajectory values")
                limit = _score_limit(episode, requested)
                window = trajectory_window(model, episode, horizon=limit)
                rolled = rollout_latent(model, window.observation, window.actions)
                predictions.append(rolled.predicted_observations)
                targets.append(window.targets)
    finally:
        model.train(was_training)

    longest = max(tensor.shape[0] for tensor in predictions)
    if requested is not None and longest < requested:
        raise ValueError(
            f"horizon {requested} exceeds every trajectory (longest scored length is {longest})"
        )
    report_h = requested if requested is not None else longest
    return _aggregate(
        predictions,
        targets,
        report_h=report_h,
        checkpoint=checkpoint_note,
        data=data_note,
    )


def write_predict_report(path: str | Path, report: PredictReport) -> None:
    """Write ``report`` as UTF-8 JSON. Parent directories are created."""

    if not isinstance(report, PredictReport):
        raise TypeError("report must be a PredictReport")
    location = Path(path)
    if location.exists() and location.is_dir():
        raise ValueError(f"report path is a directory: {location}")
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(_report_json(report), encoding="utf-8")


def read_predict_report(path: str | Path) -> PredictReport:
    """Read a report written by :func:`write_predict_report`."""

    location = Path(path)
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"prediction report is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _REPORT_KEYS:
        raise ValueError(
            "prediction report must contain format, horizons, mse_by_h, mae_by_h, "
            "residual_std_by_h, n_episodes, n_episodes_by_h, checkpoint, and data"
        )
    if payload["format"] != PREDICT_REPORT_FORMAT:
        raise ValueError(
            f"unsupported prediction report format {payload['format']!r}; "
            f"expected {PREDICT_REPORT_FORMAT!r}"
        )
    horizons = _int_list(payload["horizons"], label="horizons", minimum=1)
    if horizons != list(range(1, len(horizons) + 1)):
        raise ValueError("horizons must be 1 .. H")
    width = len(horizons)
    if width < 1:
        raise ValueError("horizons must be non-empty")
    mse = _float_list(payload["mse_by_h"], label="mse_by_h", width=width, minimum=0.0)
    mae = _float_list(payload["mae_by_h"], label="mae_by_h", width=width, minimum=0.0)
    residual = _float_list(
        payload["residual_std_by_h"],
        label="residual_std_by_h",
        width=width,
        minimum=0.0,
    )
    n_episodes = require_int(payload["n_episodes"], label="n_episodes", minimum=1)
    counts = _int_list(payload["n_episodes_by_h"], label="n_episodes_by_h", minimum=1)
    if len(counts) != width:
        raise ValueError("n_episodes_by_h length must match horizons")
    if any(count > n_episodes for count in counts):
        raise ValueError("n_episodes_by_h cannot exceed n_episodes")
    return PredictReport(
        horizons=tuple(horizons),
        mse_by_h=tuple(mse),
        mae_by_h=tuple(mae),
        residual_std_by_h=tuple(residual),
        n_episodes=n_episodes,
        n_episodes_by_h=tuple(counts),
        checkpoint=_optional_note(payload["checkpoint"], label="checkpoint"),
        data=_optional_note(payload["data"], label="data"),
    )


def format_predict_report(report: PredictReport) -> str:
    """Plain-text summary of a prediction report. The JSON file is the record."""

    if not isinstance(report, PredictReport):
        raise TypeError("report must be a PredictReport")
    lines = [
        f"format: {PREDICT_REPORT_FORMAT}",
        f"n_episodes: {report.n_episodes}",
        "horizons: " + " ".join(str(horizon) for horizon in report.horizons),
        "mse_by_h: " + " ".join(_format_float(value) for value in report.mse_by_h),
        "mae_by_h: " + " ".join(_format_float(value) for value in report.mae_by_h),
        "residual_std_by_h: "
        + " ".join(_format_float(value) for value in report.residual_std_by_h),
        "n_episodes_by_h: " + " ".join(str(count) for count in report.n_episodes_by_h),
    ]
    if report.checkpoint is not None:
        lines.append(f"checkpoint: {report.checkpoint}")
    if report.data is not None:
        lines.append(f"data: {report.data}")
    return "\n".join(lines)


def _score_limit(trajectory: Trajectory, horizon: int | None) -> int:
    length = len(trajectory.transitions)
    if length < 1:
        raise ValueError(
            "trajectory has no transitions "
            f"(env_id={trajectory.env_id}, seed={trajectory.seed})"
        )
    if horizon is None:
        return length
    return min(horizon, length)


def _aggregate(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    *,
    report_h: int,
    checkpoint: str | None,
    data: str | None,
) -> PredictReport:
    mse: list[float] = []
    mae: list[float] = []
    residual_std: list[float] = []
    counts: list[int] = []
    for index in range(report_h):
        rows: list[torch.Tensor] = []
        for predicted, target in zip(predictions, targets, strict=True):
            if predicted.shape[0] <= index:
                continue
            rows.append(
                predicted[index] - target[index].to(dtype=predicted.dtype, device=predicted.device)
            )
        if not rows:
            raise ValueError(f"no episodes reach horizon {index + 1}")
        residual = torch.stack(rows, dim=0)
        if not torch.isfinite(residual).all():
            raise ValueError(f"horizon {index + 1} residuals are not finite")
        flat = residual.reshape(-1)
        mse.append(float(flat.square().mean()))
        mae.append(float(flat.abs().mean()))
        residual_std.append(float(flat.std(unbiased=False)))
        counts.append(int(residual.shape[0]))
    return PredictReport(
        horizons=tuple(range(1, report_h + 1)),
        mse_by_h=tuple(mse),
        mae_by_h=tuple(mae),
        residual_std_by_h=tuple(residual_std),
        n_episodes=len(predictions),
        n_episodes_by_h=tuple(counts),
        checkpoint=checkpoint,
        data=data,
    )


def _report_json(report: PredictReport) -> str:
    payload = {
        "format": PREDICT_REPORT_FORMAT,
        "horizons": list(report.horizons),
        "mse_by_h": list(report.mse_by_h),
        "mae_by_h": list(report.mae_by_h),
        "residual_std_by_h": list(report.residual_std_by_h),
        "n_episodes": report.n_episodes,
        "n_episodes_by_h": list(report.n_episodes_by_h),
        "checkpoint": report.checkpoint,
        "data": report.data,
    }
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _optional_path(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string or None")
    return value


def _optional_note(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string or null")
    return value


def _int_list(value: object, *, label: str, minimum: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return [require_int(item, label=label, minimum=minimum) for item in value]


def _float_list(value: object, *, label: str, width: int, minimum: float) -> list[float]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{label} length must match horizons")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{label} must contain floats")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        if number < minimum:
            raise ValueError(f"{label} must be >= {minimum}")
        numbers.append(number)
    return numbers


def _format_float(value: float) -> str:
    return format(float(value), ".8g")
