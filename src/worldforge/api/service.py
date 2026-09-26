"""Call the Day 4–6 research loop from the HTTP models.

Importing this module does not import PyTorch. Each function loads the
optional ``ml`` extra on first use and raises :class:`MlExtraMissing`
when that import fails. User paths are resolved inside the data root
before that import. Nothing here opens a socket. An omitted ``out`` on
``POST /train`` is a new directory inside the data root.
"""

from __future__ import annotations

import importlib
import math
import tempfile
from pathlib import Path
from typing import NoReturn

from worldforge.api.schemas import (
    ActionSequenceResponse,
    EvalPlanRequest,
    ModelSpec,
    PlanReportResponse,
    PlanRequest,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from worldforge.data import load_corpus
from worldforge.envs.lotka_volterra import ACTION_LIMIT
from worldforge.sandbox import data_root, resolve_user_path
from worldforge.schemas import Trajectory

ML_EXTRA_DETAIL = (
    'model, train, prediction, and planning routes need the optional ml extra: '
    'pip install -e ".[ml]" (a CPU build of torch is enough)'
)


class MlExtraMissing(RuntimeError):
    """The route needs torch and the ``ml`` extra is not installed."""


def open_loop(body: PredictRequest) -> PredictResponse:
    """Score stored actions and return a ``worldforge.predict.v1`` body."""

    data_path, checkpoint_path = _sandboxed_inputs(body.data, body.checkpoint)
    _torch, world, checkpoint_mod, predict_api = _import_ml(
        "worldforge.models.world",
        "worldforge.models.checkpoint",
        "worldforge.eval",
    )
    model, checkpoint_note = _load_model(
        world.WorldModel,
        checkpoint_mod.load_checkpoint,
        body,
        seed=body.seed,
        checkpoint=checkpoint_path,
    )
    episodes, data_note = _episodes(data_path, body.trajectories)
    report = predict_api.open_loop_metrics(
        model,
        episodes,
        horizon=body.horizon,
        checkpoint=checkpoint_note,
        data=data_note,
    )
    return PredictResponse(
        format=predict_api.PREDICT_REPORT_FORMAT,
        horizons=list(report.horizons),
        mse_by_h=list(report.mse_by_h),
        mae_by_h=list(report.mae_by_h),
        residual_std_by_h=list(report.residual_std_by_h),
        n_episodes=report.n_episodes,
        n_episodes_by_h=list(report.n_episodes_by_h),
        checkpoint=report.checkpoint,
        data=report.data,
    )


def plan_sequence(body: PlanRequest) -> ActionSequenceResponse:
    """Search one clipped action sequence and return the Day 6 document."""

    data_path, checkpoint_path = _sandboxed_inputs(body.data, body.checkpoint)
    torch_mod, world, checkpoint_mod, plan_api = _import_ml(
        "worldforge.models.world",
        "worldforge.models.checkpoint",
        "worldforge.plan",
    )
    model, checkpoint_note = _load_model(
        world.WorldModel,
        checkpoint_mod.load_checkpoint,
        body,
        seed=body.seed,
        checkpoint=checkpoint_path,
    )
    observation = _plan_observation(body, model.config.obs_dim, plan_api, data_path)
    config = _cem_config(body, model.config.action_dim, plan_api.CEMConfig)
    planned = plan_api.plan_actions(
        model,
        torch_mod.tensor(observation, dtype=torch_mod.float32),
        config,
    )
    report = plan_api.action_sequence_report(
        planned,
        config,
        observation,
        checkpoint=checkpoint_note,
    )
    return ActionSequenceResponse(
        format=plan_api.ACTION_SEQUENCE_FORMAT,
        objective=plan_api.OBJECTIVE_NAME,
        observation=list(report.observation),
        actions=[list(action) for action in report.actions],
        predicted_return=report.predicted_return,
        horizon=report.horizon,
        n_samples=report.n_samples,
        n_iterations=report.n_iterations,
        elite_fraction=report.elite_fraction,
        seed=report.seed,
        checkpoint=report.checkpoint,
    )


def plan_regret(body: EvalPlanRequest) -> PlanReportResponse:
    """Roll closed-loop CEM and both baselines. Returns ``worldforge.plan.v1``."""

    _data_path, checkpoint_path = _sandboxed_inputs(None, body.checkpoint)
    _torch, world, checkpoint_mod, plan_api = _import_ml(
        "worldforge.models.world",
        "worldforge.models.checkpoint",
        "worldforge.plan",
    )
    model, checkpoint_note = _load_model(
        world.WorldModel,
        checkpoint_mod.load_checkpoint,
        body,
        seed=body.seed,
        checkpoint=checkpoint_path,
    )
    config = _cem_config(body, model.config.action_dim, plan_api.CEMConfig)
    report = plan_api.planning_regret(
        model,
        n_steps=body.n_steps,
        env_seed=body.env_seed,
        cem=config,
        checkpoint=checkpoint_note,
    )
    return PlanReportResponse(
        format=plan_api.PLAN_REPORT_FORMAT,
        objective=plan_api.OBJECTIVE_NAME,
        regret_definition=plan_api.REGRET_DEFINITION,
        env_id=report.env_id,
        seed=report.seed,
        cem_seed=report.cem_seed,
        n_steps=report.n_steps,
        planner_steps=report.planner_steps,
        horizon=report.horizon,
        n_samples=report.n_samples,
        n_iterations=report.n_iterations,
        elite_fraction=report.elite_fraction,
        initial_observation=list(report.initial_observation),
        planner_return=report.planner_return,
        zero_return=report.zero_return,
        random_return=report.random_return,
        baseline_best=report.baseline_best,
        regret=report.regret,
        checkpoint=report.checkpoint,
        actions=[list(action) for action in report.actions],
    )


def train(body: TrainRequest) -> TrainResponse:
    """Fit one short CPU run and return the final loss and checkpoint path."""

    root = data_root()
    data_path = _existing_user_path(body.data, label="data", root=root)
    out_path = None if body.out is None else resolve_user_path(body.out, label="out", root=root)
    _torch, world, train_api = _import_ml(
        "worldforge.models.world",
        "worldforge.train",
    )
    episodes, data_note = _episodes(data_path, body.trajectories)
    if out_path is None:
        directory: str | Path = Path(tempfile.mkdtemp(prefix="worldforge-train-", dir=root))
    else:
        directory = out_path
    config = train_api.TrainConfig(
        epochs=body.epochs,
        batch_size=body.batch_size,
        lr=body.lr,
        seed=body.seed,
        device=body.device,
        reconstruction_weight=body.reconstruction_weight,
        steps_per_epoch=body.steps_per_epoch,
        optimizer=body.optimizer,
    )
    model = world.WorldModel(body.to_config(), seed=body.seed)
    result = train_api.train_world_model(
        model,
        episodes,
        config,
        out=directory,
        data_path=data_note,
    )
    loss = float(result.final_train_loss)
    if not math.isfinite(loss):
        raise ValueError("final_train_loss is not finite")
    return TrainResponse(
        final_train_loss=loss,
        checkpoint=str(result.checkpoint),
        n_episodes=result.n_episodes,
        n_transitions=result.n_transitions,
        epochs=len(result.epochs),
        steps_per_epoch=result.steps_per_epoch,
    )


def _import_ml(*modules: str) -> tuple:
    """Import torch and ``modules``. Raise :class:`MlExtraMissing` if torch is absent."""

    try:
        imported = [importlib.import_module(name) for name in ("torch", *modules)]
    except ImportError as exc:
        _reraise_ml(exc)
    return tuple(imported)


def _reraise_ml(exc: ImportError) -> NoReturn:
    missing = getattr(exc, "name", None) or ""
    if missing == "torch" or missing.startswith("torch."):
        raise MlExtraMissing(f"{ML_EXTRA_DETAIL}. Import failed: {exc}") from exc
    raise exc


def _sandboxed_inputs(
    data: str | None,
    checkpoint: str | None,
) -> tuple[Path | None, Path | None]:
    """Resolve request paths inside the data root before any model import."""

    root = data_root()
    return (
        _existing_user_path(data, label="data", root=root),
        _existing_user_path(checkpoint, label="checkpoint", root=root),
    )


def _existing_user_path(value: str | None, *, label: str, root: Path) -> Path | None:
    if value is None:
        return None
    path = resolve_user_path(value, label=label, root=root)
    if not path.exists():
        raise ValueError(f"path does not exist: {path}")
    return path


def _load_model(
    world_model_cls: type,
    load_checkpoint: object,
    spec: ModelSpec,
    *,
    seed: int,
    checkpoint: Path | None,
) -> tuple[object, str | None]:
    if checkpoint is None:
        model = world_model_cls(spec.to_config(), seed=seed)
        return model, None
    if not callable(load_checkpoint):
        raise TypeError("load_checkpoint must be callable")
    model = load_checkpoint(checkpoint)
    return model, str(checkpoint)


def _episodes(
    data: Path | None,
    trajectories: list[Trajectory] | None,
) -> tuple[list[Trajectory], str]:
    if data is not None and trajectories is not None:
        raise ValueError("provide a data path or inline trajectories, not both")
    if trajectories is not None:
        if not trajectories:
            raise ValueError("trajectories must be non-empty")
        return list(trajectories), "inline"
    if data is None:
        raise ValueError("provide a data path or inline trajectories")
    return load_corpus(data), str(data)


def _plan_observation(
    body: PlanRequest,
    obs_dim: int,
    plan_api: object,
    data_path: Path | None,
) -> list[float]:
    # plan_api is the worldforge.plan module. Imported lazily with torch.
    if body.observation is not None:
        if len(body.observation) != obs_dim:
            raise ValueError(
                f"observation length {len(body.observation)} does not match obs_dim {obs_dim}"
            )
        return list(body.observation)
    if data_path is not None:
        episodes, _note = _episodes(data_path, None)
        if body.episode >= len(episodes):
            raise ValueError(f"episode {body.episode} is out of range for {len(episodes)} episodes")
        episode = episodes[body.episode]
        if not episode.transitions:
            raise ValueError("episode has no transitions")
        observation = list(episode.transitions[0].observation.values)
        if len(observation) != obs_dim:
            raise ValueError(
                f"observation length {len(observation)} does not match obs_dim {obs_dim}"
            )
        return observation
    return list(plan_api.default_target(obs_dim))  # type: ignore[attr-defined]


def _cem_config(body: PlanRequest | EvalPlanRequest, action_dim: int, cem_config: type) -> object:
    low = tuple(-ACTION_LIMIT for _ in range(action_dim))
    high = tuple(ACTION_LIMIT for _ in range(action_dim))
    return cem_config(
        horizon=body.horizon,
        n_samples=body.n_samples,
        n_iterations=body.n_iterations,
        elite_fraction=body.elite_fraction,
        seed=body.seed,
        action_low=low,
        action_high=high,
    )
