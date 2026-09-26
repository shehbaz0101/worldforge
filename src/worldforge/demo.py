"""One-command offline demo: sample corpus, tiny train, predict, and plan.

``run_demo`` reuses the Day 2 collector, the Day 4 trainer, the Day 5
open-loop eval, and the Day 6 planner. It does not open a socket itself.
After the optional ``ml`` extra imports, it installs the Day 8 offline
guard so a later connect to a non-loopback host fails. Paths passed in
are resolved inside the data root. The checked-in files under
``samples/`` are the default corpus. When they sit outside the root they
are copied into the output directory. When they are missing, the same
short episode list is collected into that directory.

Importing this module does not import PyTorch. :func:`run_demo` does.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from worldforge.data import EpisodeSpec, collect_corpus, load_corpus
from worldforge.data.dataset import FormatName
from worldforge.offline import install_offline_guard
from worldforge.rollout import Policy
from worldforge.sandbox import data_root, resolve_user_path

DEMO_FORMAT = "worldforge.demo.v1"
DEFAULT_OUT_DIR = "demo-run"
MAX_DEMO_EPOCHS = 5
MAX_DEMO_STEPS_PER_EPOCH = 8
_CORPUS_DIRNAME = "corpus"
_CHECKPOINT_DIRNAME = "checkpoint"
_PREDICT_NAME = "predict.json"
_PLAN_NAME = "plan.json"
_EVAL_PLAN_NAME = "eval-plan.json"
_SUMMARY_NAME = "summary.txt"
_SAMPLE_CONFIG = "demo.json"

ML_INSTALL_HINT = (
    'worldforge demo needs the optional ml extra: pip install -e ".[ml]" '
    "(a CPU build of torch is enough)."
)


class DemoDependencyError(RuntimeError):
    """The demo needs the optional ``ml`` extra."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DemoEpisode(_StrictModel):
    """One episode the demo collects when the sample corpus is missing."""

    seed: int
    policy: Policy = "random"


class DemoConfig(_StrictModel):
    """Caps and paths for one offline demo run.

    The numbers are intentionally small. ``epochs`` stops at
    :data:`MAX_DEMO_EPOCHS` and ``steps_per_epoch`` stops at
    :data:`MAX_DEMO_STEPS_PER_EPOCH`, matching the HTTP trainer. A longer
    fit belongs on ``worldforge train``.
    """

    format: Literal["worldforge.demo.v1"] = DEMO_FORMAT
    corpus: str = "trajectories"
    horizon: int = Field(default=4, ge=1, le=16)
    episodes: list[DemoEpisode] = Field(
        default_factory=lambda: [
            DemoEpisode(seed=0, policy="zero"),
            DemoEpisode(seed=1, policy="random"),
        ]
    )
    epochs: int = Field(default=1, ge=1, le=MAX_DEMO_EPOCHS)
    batch_size: int = Field(default=4, ge=1, le=32)
    steps_per_epoch: int = Field(default=1, ge=1, le=MAX_DEMO_STEPS_PER_EPOCH)
    lr: float = Field(default=1e-3, gt=0)
    seed: int = Field(default=0, ge=0)
    device: str = "cpu"
    latent_dim: int = Field(default=4, ge=1, le=32)
    hidden_dim: int = Field(default=16, ge=1, le=128)
    predict_horizon: int = Field(default=4, ge=1, le=16)
    plan_horizon: int = Field(default=2, ge=1, le=8)
    n_samples: int = Field(default=4, ge=1, le=32)
    n_iterations: int = Field(default=1, ge=1, le=4)
    elite_fraction: float = Field(default=0.5, gt=0, le=1)
    n_steps: int = Field(default=2, ge=1, le=8)
    env_seed: int = Field(default=0, ge=0)

    @field_validator("device")
    @classmethod
    def _cpu_only(cls, value: str) -> str:
        if value != "cpu":
            raise ValueError("device must be 'cpu'")
        return value

    def with_overrides(self, *, seed: int | None = None, epochs: int | None = None) -> DemoConfig:
        """Return a validated copy with the CLI seed and epoch overrides applied.

        ``seed`` sets the weight seed, the replay seed, the CEM seed, and
        the closed-loop environment seed together.
        """

        payload = self.model_dump()
        if seed is not None:
            payload["seed"] = seed
            payload["env_seed"] = seed
        if epochs is not None:
            payload["epochs"] = epochs
        return DemoConfig.model_validate(payload)


@dataclass(frozen=True)
class DemoResult:
    """Artifacts and reports from one :func:`run_demo` call."""

    corpus: Path
    corpus_source: str
    checkpoint: Path
    predict_path: Path
    plan_path: Path
    eval_plan_path: Path
    summary_path: Path
    train: object
    predict: object
    plan: object
    eval_plan: object


def bundled_samples_dir() -> Path | None:
    """Return the checked-in ``samples/`` directory, or ``None`` when it is absent.

    The search walks parents of this file so an editable install finds the
    repository ``samples/`` next to ``src/``.
    """

    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        candidate = parent / "samples" / _SAMPLE_CONFIG
        if candidate.is_file():
            return candidate.parent
    return None


def load_demo_config(path: str | Path | None = None) -> DemoConfig:
    """Load ``samples/demo.json``, or the built-in caps when that file is absent."""

    if path is None:
        samples = bundled_samples_dir()
        if samples is None:
            return DemoConfig()
        path = samples / _SAMPLE_CONFIG
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"demo config does not exist: {config_path}")
    try:
        return DemoConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"demo config is invalid: {config_path}") from exc


def prepare_demo_corpus(
    *,
    root: Path,
    out: Path,
    config: DemoConfig,
    data: Path | None = None,
) -> tuple[Path, str]:
    """Choose the corpus inside ``root`` and return ``(path, source)``.

    ``source`` is ``user`` for ``data``, ``samples`` when the checked-in
    corpus already lies inside ``root``, ``copied`` when those files were
    copied under ``out``, or ``built`` when they were collected there.
    """

    if data is not None:
        if not data.exists():
            raise ValueError(f"path does not exist: {data}")
        return data, "user"

    samples = bundled_samples_dir()
    if samples is not None:
        corpus = _sample_corpus(samples, config.corpus)
        if corpus.is_dir():
            if _inside(corpus, root):
                return corpus, "samples"
            destination = _corpus_destination(out, corpus)
            _copy_corpus(corpus, destination)
            return destination, "copied"

    destination = out / _CORPUS_DIRNAME
    _collect_demo_corpus(destination, config)
    return destination, "built"


def run_demo(
    out: str | Path,
    *,
    data: str | Path | None = None,
    config: DemoConfig | None = None,
    seed: int | None = None,
    epochs: int | None = None,
) -> DemoResult:
    """Train a tiny checkpoint and write prediction and planning reports.

    ``out`` and ``data`` must stay inside the data root. The root is
    ``WORLDFORGE_DATA_ROOT`` when that variable is set, and the current
    directory otherwise. ``worldforge demo --data-root`` sets it.
    """

    ml = _import_ml()
    install_offline_guard()
    settings = config if config is not None else load_demo_config()
    if not isinstance(settings, DemoConfig):
        raise TypeError("config must be a DemoConfig")
    settings = settings.with_overrides(seed=seed, epochs=epochs)
    if settings.device != "cpu":
        raise ValueError("device must be 'cpu'")

    root = data_root()
    output = resolve_user_path(out, label="--out", root=root)
    corpus_arg = None if data is None else resolve_user_path(data, label="--data", root=root)
    if output.exists() and not output.is_dir():
        raise ValueError(f"demo output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    corpus, source = prepare_demo_corpus(root=root, out=output, config=settings, data=corpus_arg)
    episodes = load_corpus(corpus)
    checkpoint_dir = output / _CHECKPOINT_DIRNAME
    model = ml["WorldModel"](
        ml["ModelConfig"](latent_dim=settings.latent_dim, hidden_dim=settings.hidden_dim),
        seed=settings.seed,
    )
    train_config = ml["TrainConfig"](
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        lr=settings.lr,
        seed=settings.seed,
        device=settings.device,
        steps_per_epoch=settings.steps_per_epoch,
    )
    trained = ml["train_world_model"](
        model,
        episodes,
        train_config,
        out=checkpoint_dir,
        data_path=corpus,
    )
    restored = ml["load_checkpoint"](checkpoint_dir)
    checkpoint_note = str(checkpoint_dir)
    predict = ml["open_loop_metrics"](
        restored,
        episodes,
        horizon=settings.predict_horizon,
        checkpoint=checkpoint_note,
        data=str(corpus),
    )
    predict_path = output / _PREDICT_NAME
    ml["write_predict_report"](predict_path, predict)

    observation = _first_observation(episodes, restored.config.obs_dim)
    cem = _cem_config(ml["CEMConfig"], settings, restored.config.action_dim)
    planned = ml["plan_actions"](
        restored,
        ml["torch"].tensor(observation, dtype=ml["torch"].float32),
        cem,
    )
    plan = ml["action_sequence_report"](
        planned,
        cem,
        observation,
        checkpoint=checkpoint_note,
    )
    plan_path = output / _PLAN_NAME
    ml["write_action_sequence"](plan_path, plan)

    eval_plan = ml["planning_regret"](
        restored,
        n_steps=settings.n_steps,
        env_seed=settings.env_seed,
        cem=cem,
        checkpoint=checkpoint_note,
    )
    eval_plan_path = output / _EVAL_PLAN_NAME
    ml["write_plan_report"](eval_plan_path, eval_plan)

    result = DemoResult(
        corpus=corpus,
        corpus_source=source,
        checkpoint=checkpoint_dir,
        predict_path=predict_path,
        plan_path=plan_path,
        eval_plan_path=eval_plan_path,
        summary_path=output / _SUMMARY_NAME,
        train=trained,
        predict=predict,
        plan=plan,
        eval_plan=eval_plan,
    )
    text = format_demo_summary(result)
    result.summary_path.write_text(text + "\n", encoding="utf-8")
    return result


def metrics_are_finite(result: DemoResult) -> bool:
    """Return whether the printed demo metrics are all finite."""

    if not isinstance(result, DemoResult):
        raise TypeError("result must be a DemoResult")
    predict = result.predict
    plan = result.plan
    regret = result.eval_plan
    train = result.train
    values = [
        train.final_train_loss,
        plan.predicted_return,
        regret.planner_return,
        regret.zero_return,
        regret.random_return,
        regret.baseline_best,
        regret.regret,
        *predict.mse_by_h,
    ]
    return all(math.isfinite(float(value)) for value in values)


def format_demo_summary(result: DemoResult) -> str:
    """Short plain-text summary of the demo metrics. The JSON files are the record."""

    if not isinstance(result, DemoResult):
        raise TypeError("result must be a DemoResult")
    predict = result.predict
    plan = result.plan
    regret = result.eval_plan
    train = result.train
    horizons = " ".join(str(horizon) for horizon in predict.horizons)
    mse = " ".join(_format_float(value) for value in predict.mse_by_h)
    lines = [
        "worldforge demo",
        f"corpus: {result.corpus}",
        f"corpus_source: {result.corpus_source}",
        f"episodes: {train.n_episodes}",
        f"transitions: {train.n_transitions}",
        f"checkpoint: {result.checkpoint}",
        f"final_train_loss: {_format_float(train.final_train_loss)}",
        f"predict_n_episodes: {predict.n_episodes}",
        f"predict_horizons: {horizons}",
        f"predict_mse_by_h: {mse}",
        f"plan_predicted_return: {_format_float(plan.predicted_return)}",
        f"plan_horizon: {plan.horizon}",
        f"planner_return: {_format_float(regret.planner_return)}",
        f"zero_return: {_format_float(regret.zero_return)}",
        f"random_return: {_format_float(regret.random_return)}",
        f"baseline_best: {_format_float(regret.baseline_best)}",
        f"regret: {_format_float(regret.regret)}",
        f"predict_report: {result.predict_path}",
        f"plan_report: {result.plan_path}",
        f"eval_plan_report: {result.eval_plan_path}",
    ]
    return "\n".join(lines)


def _import_ml() -> dict[str, object]:
    """Import torch and the train, predict, and plan libraries."""

    try:
        import importlib

        torch = importlib.import_module("torch")
        checkpoint = importlib.import_module("worldforge.models.checkpoint")
        config_mod = importlib.import_module("worldforge.models.config")
        world = importlib.import_module("worldforge.models.world")
        train_api = importlib.import_module("worldforge.train")
        eval_api = importlib.import_module("worldforge.eval")
        plan_api = importlib.import_module("worldforge.plan")
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "torch" or str(missing).startswith("torch."):
            raise DemoDependencyError(f"{ML_INSTALL_HINT} Import failed: {exc}") from exc
        raise
    return {
        "torch": torch,
        "WorldModel": world.WorldModel,
        "ModelConfig": config_mod.ModelConfig,
        "load_checkpoint": checkpoint.load_checkpoint,
        "TrainConfig": train_api.TrainConfig,
        "train_world_model": train_api.train_world_model,
        "open_loop_metrics": eval_api.open_loop_metrics,
        "write_predict_report": eval_api.write_predict_report,
        "CEMConfig": plan_api.CEMConfig,
        "plan_actions": plan_api.plan_actions,
        "action_sequence_report": plan_api.action_sequence_report,
        "write_action_sequence": plan_api.write_action_sequence,
        "planning_regret": plan_api.planning_regret,
        "write_plan_report": plan_api.write_plan_report,
    }


def _cem_config(cem_config: type, settings: DemoConfig, action_dim: int) -> object:
    from worldforge.envs.lotka_volterra import ACTION_LIMIT

    low = tuple(-ACTION_LIMIT for _ in range(action_dim))
    high = tuple(ACTION_LIMIT for _ in range(action_dim))
    return cem_config(
        horizon=settings.plan_horizon,
        n_samples=settings.n_samples,
        n_iterations=settings.n_iterations,
        elite_fraction=settings.elite_fraction,
        seed=settings.seed,
        action_low=low,
        action_high=high,
    )


def _first_observation(episodes: object, obs_dim: int) -> list[float]:
    if not episodes:
        raise ValueError("corpus has no episodes")
    episode = episodes[0]
    if not episode.transitions:
        raise ValueError("episode has no transitions")
    observation = list(episode.transitions[0].observation.values)
    if len(observation) != obs_dim:
        raise ValueError(
            f"observation length {len(observation)} does not match obs_dim {obs_dim}"
        )
    return observation


def _sample_corpus(samples: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not relative.strip():
        raise ValueError("demo corpus path must stay inside samples/")
    corpus = (samples / raw).resolve()
    try:
        corpus.relative_to(samples.resolve())
    except ValueError as exc:
        raise ValueError("demo corpus path must stay inside samples/") from exc
    return corpus


def _corpus_destination(out: Path, source: Path) -> Path:
    destination = (out / _CORPUS_DIRNAME).resolve()
    resolved_source = source.resolve()
    if destination == resolved_source or _inside(destination, resolved_source):
        raise ValueError(
            "demo output corpus would overlap the sample corpus; choose another --out"
        )
    if _inside(resolved_source, destination):
        raise ValueError(
            "demo output corpus would overlap the sample corpus; choose another --out"
        )
    return destination


def _copy_corpus(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _collect_demo_corpus(destination: Path, config: DemoConfig) -> None:
    specs = tuple(EpisodeSpec(seed=episode.seed, policy=episode.policy) for episode in config.episodes)
    formats: tuple[FormatName, ...] = ("jsonl", "json")
    collect_corpus(
        destination,
        horizon=config.horizon,
        specs=specs,
        formats=formats,
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _format_float(value: float) -> str:
    return format(float(value), ".8g")
