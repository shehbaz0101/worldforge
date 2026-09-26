"""Closed-loop planning regret and the Day 6 JSON reports.

``planning_regret`` resets the default Lotka-Volterra environment, replans
with CEM at every step, and executes only the first action. The same
initial seed is then rolled with a zero dose and with the Day 1 random
dose. All three returns are sums of the environment reward, which is the
negative L1 distance from the coexistence equilibrium. Higher is better.

``regret`` is ``baseline_best - planner_return``, where ``baseline_best``
is the larger of the zero-dose return and the random-dose return. A
negative regret means the planner beat both baselines. A positive regret
means the better baseline scored higher. An untrained model can lose to
the zero dose; the report still records that gap.

``worldforge.plan.v1`` is the closed-loop report. ``worldforge.action_sequence.v1``
is the one-shot sequence from :func:`worldforge.plan.cem.plan_actions`.
Neither file stores weights or secrets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from worldforge.envs import DEFAULT_ENV_ID, make_env
from worldforge.models.world import WorldModel
from worldforge.plan.cem import ActionPlan, CEMConfig, plan_actions
from worldforge.plan.objective import OBJECTIVE_NAME
from worldforge.rollout import rollout
from worldforge.schemas import Trajectory

PLAN_REPORT_FORMAT = "worldforge.plan.v1"
ACTION_SEQUENCE_FORMAT = "worldforge.action_sequence.v1"
REGRET_DEFINITION = "baseline_best - planner_return"
DEFAULT_N_STEPS = 4

_PLAN_KEYS = frozenset(
    {
        "format",
        "objective",
        "regret_definition",
        "env_id",
        "seed",
        "cem_seed",
        "n_steps",
        "planner_steps",
        "horizon",
        "n_samples",
        "n_iterations",
        "elite_fraction",
        "initial_observation",
        "planner_return",
        "zero_return",
        "random_return",
        "baseline_best",
        "regret",
        "checkpoint",
        "actions",
    }
)
_SEQUENCE_KEYS = frozenset(
    {
        "format",
        "objective",
        "observation",
        "actions",
        "predicted_return",
        "horizon",
        "n_samples",
        "n_iterations",
        "elite_fraction",
        "seed",
        "checkpoint",
    }
)


@dataclass(frozen=True)
class PlanReport:
    """Closed-loop returns and regret for one seeded episode.

    ``actions`` are the doses actually applied, one per environment step,
    not the full planning horizon. ``horizon`` is the CEM horizon used at
    each replan. ``seed`` resets the environment and draws the random
    baseline. ``cem_seed`` only draws CEM candidates.
    """

    seed: int
    cem_seed: int
    n_steps: int
    planner_steps: int
    horizon: int
    n_samples: int
    n_iterations: int
    elite_fraction: float
    initial_observation: tuple[float, ...]
    planner_return: float
    zero_return: float
    random_return: float
    baseline_best: float
    regret: float
    actions: tuple[tuple[float, ...], ...]
    env_id: str = DEFAULT_ENV_ID
    checkpoint: str | None = None

    @property
    def objective(self) -> str:
        return OBJECTIVE_NAME


@dataclass(frozen=True)
class ActionSequenceReport:
    """One planned sequence and the model return it scored."""

    observation: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]
    predicted_return: float
    horizon: int
    n_samples: int
    n_iterations: int
    elite_fraction: float
    seed: int
    checkpoint: str | None = None

    @property
    def objective(self) -> str:
        return OBJECTIVE_NAME


def planning_regret(
    model: WorldModel,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    env_seed: int = 0,
    cem: CEMConfig | None = None,
    checkpoint: str | None = None,
) -> PlanReport:
    """Roll the planner and both baselines from ``env_seed``.

    The environment horizon is ``n_steps``. Each policy stops early if the
    episode terminates. CEM bounds must match the environment. The model
    observation and action widths must match it too.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    steps = _require_int(n_steps, label="n_steps", minimum=1)
    seed = _require_int(env_seed, label="env_seed")
    config = cem if cem is not None else CEMConfig()
    if not isinstance(config, CEMConfig):
        raise TypeError("cem must be a CEMConfig")
    note = _optional_path(checkpoint, label="checkpoint")
    env = make_env(horizon=steps)
    if model.config.obs_dim != env.observation_dim or model.config.action_dim != env.action_dim:
        raise ValueError(
            "model dimensions must match the environment, "
            f"got obs {model.config.obs_dim} action {model.config.action_dim}, "
            f"env obs {env.observation_dim} action {env.action_dim}"
        )
    if config.action_low != tuple(env.action_low) or config.action_high != tuple(env.action_high):
        raise ValueError("CEM action bounds must match the environment")

    observation = env.reset(seed)
    initial = _as_float_tuple(observation, label="initial_observation")
    executed: list[tuple[float, ...]] = []
    total = 0.0
    for _step in range(steps):
        current = torch.tensor(observation, dtype=torch.float32)
        plan = plan_actions(model, current, config)
        action = _as_float_tuple(plan.actions[0].tolist(), label="action")
        observation, reward, terminated, truncated, _info = env.step(action)
        executed.append(action)
        total += float(reward)
        if terminated or truncated:
            break
    planner_return = _clean_zero(total)
    zero_return = _episode_return(rollout(env, n_steps=steps, seed=seed, policy="zero"))
    random_return = _episode_return(rollout(env, n_steps=steps, seed=seed, policy="random"))
    baseline_best = max(zero_return, random_return)
    regret = baseline_best - planner_return
    return PlanReport(
        seed=seed,
        cem_seed=config.seed,
        n_steps=steps,
        planner_steps=len(executed),
        horizon=config.horizon,
        n_samples=config.n_samples,
        n_iterations=config.n_iterations,
        elite_fraction=config.elite_fraction,
        initial_observation=initial,
        planner_return=planner_return,
        zero_return=zero_return,
        random_return=random_return,
        baseline_best=baseline_best,
        regret=regret,
        actions=tuple(executed),
        env_id=env.env_id,
        checkpoint=note,
    )


def action_sequence_report(
    plan: ActionPlan,
    config: CEMConfig,
    observation: Sequence[float],
    *,
    checkpoint: str | None = None,
) -> ActionSequenceReport:
    """Pack a one-shot plan into the action-sequence report."""

    if not isinstance(plan, ActionPlan):
        raise TypeError("plan must be an ActionPlan")
    if not isinstance(config, CEMConfig):
        raise TypeError("config must be a CEMConfig")
    actions = _rows_from_tensor(plan.actions)
    if len(actions) != config.horizon:
        raise ValueError("plan length must match config.horizon")
    return ActionSequenceReport(
        observation=_as_float_tuple(observation, label="observation"),
        actions=actions,
        predicted_return=_require_finite(plan.predicted_return, label="predicted_return"),
        horizon=config.horizon,
        n_samples=config.n_samples,
        n_iterations=config.n_iterations,
        elite_fraction=config.elite_fraction,
        seed=config.seed,
        checkpoint=_optional_path(checkpoint, label="checkpoint"),
    )


def write_plan_report(path: str | Path, report: PlanReport) -> None:
    """Write ``report`` as UTF-8 JSON. Parent directories are created."""

    if not isinstance(report, PlanReport):
        raise TypeError("report must be a PlanReport")
    _write_json(path, _plan_payload(report))


def read_plan_report(path: str | Path) -> PlanReport:
    """Read a report written by :func:`write_plan_report`."""

    payload = _read_object(path, label="plan report")
    if set(payload) != _PLAN_KEYS:
        raise ValueError("plan report is missing required fields or has unexpected fields")
    if payload["format"] != PLAN_REPORT_FORMAT:
        raise ValueError(
            f"unsupported plan report format {payload['format']!r}; expected {PLAN_REPORT_FORMAT!r}"
        )
    if payload["objective"] != OBJECTIVE_NAME:
        raise ValueError(f"unsupported objective {payload['objective']!r}; expected {OBJECTIVE_NAME!r}")
    if payload["regret_definition"] != REGRET_DEFINITION:
        raise ValueError(
            f"unsupported regret_definition {payload['regret_definition']!r}; "
            f"expected {REGRET_DEFINITION!r}"
        )
    env_id = payload["env_id"]
    if not isinstance(env_id, str) or not env_id:
        raise ValueError("env_id must be a non-empty string")
    actions = _action_rows(payload["actions"], label="actions")
    n_steps = _require_int(payload["n_steps"], label="n_steps", minimum=1)
    planner_steps = _require_int(payload["planner_steps"], label="planner_steps", minimum=1)
    if planner_steps != len(actions):
        raise ValueError("planner_steps must match the action list")
    if planner_steps > n_steps:
        raise ValueError("planner_steps cannot exceed n_steps")
    planner_return = _require_finite(payload["planner_return"], label="planner_return")
    zero_return = _require_finite(payload["zero_return"], label="zero_return")
    random_return = _require_finite(payload["random_return"], label="random_return")
    baseline_best = _require_finite(payload["baseline_best"], label="baseline_best")
    regret = _require_finite(payload["regret"], label="regret")
    if abs(baseline_best - max(zero_return, random_return)) > 1e-9:
        raise ValueError("baseline_best must be the max of zero_return and random_return")
    if abs((baseline_best - planner_return) - regret) > 1e-9:
        raise ValueError("regret must equal baseline_best - planner_return")
    return PlanReport(
        seed=_require_int(payload["seed"], label="seed"),
        cem_seed=_require_int(payload["cem_seed"], label="cem_seed", minimum=0),
        n_steps=n_steps,
        planner_steps=planner_steps,
        horizon=_require_int(payload["horizon"], label="horizon", minimum=1),
        n_samples=_require_int(payload["n_samples"], label="n_samples", minimum=1),
        n_iterations=_require_int(payload["n_iterations"], label="n_iterations", minimum=1),
        elite_fraction=_require_fraction(payload["elite_fraction"], label="elite_fraction"),
        initial_observation=_as_float_tuple(
            payload["initial_observation"], label="initial_observation"
        ),
        planner_return=planner_return,
        zero_return=zero_return,
        random_return=random_return,
        baseline_best=baseline_best,
        regret=regret,
        actions=actions,
        env_id=env_id,
        checkpoint=_optional_note(payload["checkpoint"], label="checkpoint"),
    )


def write_action_sequence(path: str | Path, report: ActionSequenceReport) -> None:
    """Write a one-shot action sequence as UTF-8 JSON."""

    if not isinstance(report, ActionSequenceReport):
        raise TypeError("report must be an ActionSequenceReport")
    _write_json(path, _sequence_payload(report))


def read_action_sequence(path: str | Path) -> ActionSequenceReport:
    """Read a sequence written by :func:`write_action_sequence`."""

    payload = _read_object(path, label="action sequence")
    if set(payload) != _SEQUENCE_KEYS:
        raise ValueError("action sequence is missing required fields or has unexpected fields")
    if payload["format"] != ACTION_SEQUENCE_FORMAT:
        raise ValueError(
            f"unsupported action sequence format {payload['format']!r}; "
            f"expected {ACTION_SEQUENCE_FORMAT!r}"
        )
    if payload["objective"] != OBJECTIVE_NAME:
        raise ValueError(f"unsupported objective {payload['objective']!r}; expected {OBJECTIVE_NAME!r}")
    actions = _action_rows(payload["actions"], label="actions")
    horizon = _require_int(payload["horizon"], label="horizon", minimum=1)
    if len(actions) != horizon:
        raise ValueError("horizon must match the action list")
    return ActionSequenceReport(
        observation=_as_float_tuple(payload["observation"], label="observation"),
        actions=actions,
        predicted_return=_require_finite(payload["predicted_return"], label="predicted_return"),
        horizon=horizon,
        n_samples=_require_int(payload["n_samples"], label="n_samples", minimum=1),
        n_iterations=_require_int(payload["n_iterations"], label="n_iterations", minimum=1),
        elite_fraction=_require_fraction(payload["elite_fraction"], label="elite_fraction"),
        seed=_require_int(payload["seed"], label="seed", minimum=0),
        checkpoint=_optional_note(payload["checkpoint"], label="checkpoint"),
    )


def format_plan_report(report: PlanReport) -> str:
    """Plain-text summary of a planning report. The JSON file is the record."""

    if not isinstance(report, PlanReport):
        raise TypeError("report must be a PlanReport")
    lines = [
        f"format: {PLAN_REPORT_FORMAT}",
        f"objective: {OBJECTIVE_NAME}",
        f"regret_definition: {REGRET_DEFINITION}",
        f"env_id: {report.env_id}",
        f"seed: {report.seed}",
        f"cem_seed: {report.cem_seed}",
        f"n_steps: {report.n_steps}",
        f"planner_steps: {report.planner_steps}",
        f"horizon: {report.horizon}",
        f"n_samples: {report.n_samples}",
        f"n_iterations: {report.n_iterations}",
        f"planner_return: {_format_float(report.planner_return)}",
        f"zero_return: {_format_float(report.zero_return)}",
        f"random_return: {_format_float(report.random_return)}",
        f"baseline_best: {_format_float(report.baseline_best)}",
        f"regret: {_format_float(report.regret)}",
    ]
    if report.checkpoint is not None:
        lines.append(f"checkpoint: {report.checkpoint}")
    return "\n".join(lines)


def format_action_sequence(report: ActionSequenceReport) -> str:
    """Plain-text summary of a one-shot action sequence."""

    if not isinstance(report, ActionSequenceReport):
        raise TypeError("report must be an ActionSequenceReport")
    lines = [
        f"format: {ACTION_SEQUENCE_FORMAT}",
        f"objective: {OBJECTIVE_NAME}",
        f"predicted_return: {_format_float(report.predicted_return)}",
        f"horizon: {report.horizon}",
        f"n_samples: {report.n_samples}",
        f"n_iterations: {report.n_iterations}",
        f"seed: {report.seed}",
        "observation: " + " ".join(_format_float(value) for value in report.observation),
    ]
    for index, action in enumerate(report.actions):
        rendered = " ".join(_format_float(value) for value in action)
        lines.append(f"action[{index}]: {rendered}")
    if report.checkpoint is not None:
        lines.append(f"checkpoint: {report.checkpoint}")
    return "\n".join(lines)


def _plan_payload(report: PlanReport) -> dict[str, object]:
    return {
        "format": PLAN_REPORT_FORMAT,
        "objective": OBJECTIVE_NAME,
        "regret_definition": REGRET_DEFINITION,
        "env_id": report.env_id,
        "seed": report.seed,
        "cem_seed": report.cem_seed,
        "n_steps": report.n_steps,
        "planner_steps": report.planner_steps,
        "horizon": report.horizon,
        "n_samples": report.n_samples,
        "n_iterations": report.n_iterations,
        "elite_fraction": report.elite_fraction,
        "initial_observation": list(report.initial_observation),
        "planner_return": report.planner_return,
        "zero_return": report.zero_return,
        "random_return": report.random_return,
        "baseline_best": report.baseline_best,
        "regret": report.regret,
        "checkpoint": report.checkpoint,
        "actions": [list(action) for action in report.actions],
    }


def _sequence_payload(report: ActionSequenceReport) -> dict[str, object]:
    return {
        "format": ACTION_SEQUENCE_FORMAT,
        "objective": OBJECTIVE_NAME,
        "observation": list(report.observation),
        "actions": [list(action) for action in report.actions],
        "predicted_return": report.predicted_return,
        "horizon": report.horizon,
        "n_samples": report.n_samples,
        "n_iterations": report.n_iterations,
        "elite_fraction": report.elite_fraction,
        "seed": report.seed,
        "checkpoint": report.checkpoint,
    }


def _rows_from_tensor(actions: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    if not isinstance(actions, torch.Tensor) or actions.ndim != 2:
        raise ValueError("actions must be a rank-2 tensor")
    if not torch.isfinite(actions).all():
        raise ValueError("actions must be finite")
    rows = []
    for row in actions.detach().to(dtype=torch.float32, device="cpu").tolist():
        rows.append(_as_float_tuple(row, label="actions"))
    if not rows:
        raise ValueError("actions must be non-empty")
    return tuple(rows)


def _episode_return(trajectory: Trajectory) -> float:
    total = sum(step.reward for step in trajectory.transitions)
    return _clean_zero(float(total))


def _clean_zero(value: float) -> float:
    if value == 0.0:
        return 0.0
    return float(value)


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    location = Path(path)
    if location.exists() and location.is_dir():
        raise ValueError(f"report path is a directory: {location}")
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_object(path: str | Path, *, label: str) -> dict[str, object]:
    location = Path(path)
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _action_rows(value: object, *, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    rows = [_as_float_tuple(row, label=label) for row in value]
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{label} rows must share one width")
    return tuple(rows)


def _as_float_tuple(value: object, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of floats")
    if len(value) < 1:
        raise ValueError(f"{label} must be non-empty")
    return tuple(_require_finite(item, label=label) for item in value)


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


def _require_fraction(value: object, *, label: str) -> float:
    number = _require_finite(value, label=label)
    if number <= 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in (0, 1]")
    return number


def _format_float(value: float) -> str:
    return format(float(value), ".8g")
