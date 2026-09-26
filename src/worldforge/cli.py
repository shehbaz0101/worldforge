"""Command line interface.

``version`` prints the package version.
``env-demo`` rolls the default kinetic environment for a few steps and
prints a trajectory summary plus the final ASCII state.
``collect`` writes an offline trajectory corpus.
``dataset-info`` summarizes a corpus directory or a trajectory file.
``model-info`` prints the Day 3 architecture sizes.
``forward-smoke`` runs one encode / step / decode. It does not train.
``train`` fits that model on an offline corpus and writes a checkpoint.
Dataset commands do not import PyTorch. The model and train commands need
the optional ``ml`` extra. None of these commands use the network.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from pydantic import ValidationError

from worldforge import __version__
from worldforge.data import collect_corpus, format_dataset_info, inspect_dataset, load_corpus
from worldforge.data.dataset import DEFAULT_HORIZON, DEFAULT_N_EPISODES, FormatName
from worldforge.envs import DEFAULT_ENV_ID, make_env
from worldforge.models.config import (
    DEFAULT_ACTION_DIM,
    DEFAULT_DYNAMICS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_OBS_DIM,
    KNOWN_DYNAMICS,
    ModelConfig,
)
from worldforge.rollout import POLICIES, format_summary, rollout

DEFAULT_DEMO_STEPS = 12
MAX_DEMO_STEPS = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldforge",
        description="WorldForge scientific dynamics sandbox.",
    )
    parser.add_argument("--version", action="version", version=f"worldforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Print the package version")

    demo = subparsers.add_parser(
        "env-demo",
        help=f"Roll the default {DEFAULT_ENV_ID} environment and print a trajectory summary",
    )
    demo.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_DEMO_STEPS,
        help=f"episode length, 1-{MAX_DEMO_STEPS} (default: {DEFAULT_DEMO_STEPS})",
    )
    demo.add_argument(
        "--seed",
        type=int,
        default=0,
        help="reset seed (default: 0)",
    )
    demo.add_argument(
        "--policy",
        choices=POLICIES,
        default="zero",
        help="zero dose, a seeded uniform dose, or an open-loop sine dose (default: zero)",
    )

    collect = subparsers.add_parser(
        "collect",
        help="Roll the default environment and write an offline trajectory corpus",
    )
    collect.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory (manifest.json plus the requested trajectory files)",
    )
    collect.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help=f"episode count when --seeds is omitted (default: {DEFAULT_N_EPISODES})",
    )
    collect.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"steps requested per episode (default: {DEFAULT_HORIZON})",
    )
    collect.add_argument(
        "--seed",
        type=int,
        default=0,
        help="first episode seed; later episodes use seed, seed+1, ... (default: 0)",
    )
    collect.add_argument(
        "--seeds",
        default=None,
        help="comma-separated episode seeds; overrides --seed and sets the episode count",
    )
    collect.add_argument(
        "--policy",
        choices=POLICIES,
        default="random",
        help="zero dose, a seeded uniform dose, or an open-loop sine dose (default: random)",
    )
    collect.add_argument(
        "--format",
        choices=("jsonl", "json", "both"),
        default="both",
        help="corpus JSONL, one JSON file per episode, or both (default: both)",
    )

    info = subparsers.add_parser(
        "dataset-info",
        help="Summarize a corpus directory, a corpus JSONL file, or one trajectory file",
    )
    info.add_argument(
        "path",
        type=Path,
        help="corpus directory, corpus JSONL, episode JSONL, or episode JSON",
    )

    model_info = subparsers.add_parser(
        "model-info",
        help="Print Day 3 encoder, dynamics, and decoder sizes",
    )
    _add_model_config_args(model_info)

    smoke = subparsers.add_parser(
        "forward-smoke",
        help="Encode, step, and decode one transition without training",
    )
    _add_model_config_args(smoke)
    smoke.add_argument(
        "--seed",
        type=int,
        default=0,
        help="weight init seed (default: 0)",
    )
    smoke.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="corpus directory or trajectory file; default is a synthetic equilibrium state",
    )
    smoke.add_argument(
        "--episode",
        type=int,
        default=0,
        help="episode index inside --fixture (default: 0)",
    )
    smoke.add_argument(
        "--transition",
        type=int,
        default=0,
        help="transition index inside that episode (default: 0)",
    )

    train = subparsers.add_parser(
        "train",
        help="Fit the baseline world model on an offline corpus and write a checkpoint",
    )
    train.add_argument(
        "--data",
        type=Path,
        required=True,
        help="fixture directory, corpus JSONL, or one trajectory file",
    )
    train.add_argument(
        "--out",
        type=Path,
        required=True,
        help="checkpoint directory (config.json, weights.pt, metrics.jsonl, train.json)",
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="optimizer passes (default: 3)",
    )
    train.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="transitions per replay batch (default: 8)",
    )
    train.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="optimizer learning rate (default: 1e-3)",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=0,
        help="weight init and replay seed (default: 0)",
    )
    train.add_argument(
        "--device",
        default="cpu",
        help="torch device; this command accepts cpu only (default: cpu)",
    )
    train.add_argument(
        "--recon-weight",
        type=float,
        default=1.0,
        help="weight on reconstruction MSE; 0 trains prediction only (default: 1)",
    )
    train.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="replay batches per epoch (default: ceil(transitions / batch size))",
    )
    train.add_argument(
        "--optimizer",
        choices=("adam", "sgd"),
        default="adam",
        help="adam or sgd (default: adam)",
    )
    _add_model_config_args(train)
    return parser


def _add_model_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=DEFAULT_OBS_DIM,
        help=f"observation features (default: {DEFAULT_OBS_DIM}, Lotka-Volterra)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=DEFAULT_ACTION_DIM,
        help=f"action features (default: {DEFAULT_ACTION_DIM}, Lotka-Volterra dose)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=DEFAULT_LATENT_DIM,
        help=f"latent width (default: {DEFAULT_LATENT_DIM})",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=f"MLP hidden width (default: {DEFAULT_HIDDEN_DIM})",
    )
    parser.add_argument(
        "--dynamics",
        choices=KNOWN_DYNAMICS,
        default=DEFAULT_DYNAMICS,
        help="mlp is the Day 3 residual baseline; rssm is reserved and not implemented",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(f"worldforge {__version__}")
        return 0
    if args.command == "env-demo":
        return _env_demo(parser, args)
    if args.command == "collect":
        return _collect(parser, args)
    if args.command == "dataset-info":
        return _dataset_info(parser, args)
    if args.command == "model-info":
        return _model_info(parser, args)
    if args.command == "forward-smoke":
        return _forward_smoke(parser, args)
    if args.command == "train":
        return _train(parser, args)
    parser.error(f"unknown command {args.command}")
    return 2


def _env_demo(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.steps < 1 or args.steps > MAX_DEMO_STEPS:
        parser.error(f"--steps must be between 1 and {MAX_DEMO_STEPS}")
    env = make_env(horizon=args.steps)
    trajectory = rollout(env, n_steps=args.steps, seed=args.seed, policy=args.policy)
    print(format_summary(trajectory, policy=args.policy))
    print(env.render_ascii())
    return 0


def _collect(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        seeds = None if args.seeds is None else _parse_seeds(args.seeds)
        formats = _parse_formats(args.format)
        collect_corpus(
            args.out,
            n_episodes=args.n_episodes,
            horizon=args.horizon,
            seed=args.seed,
            seeds=seeds,
            policy=args.policy,
            formats=formats,
        )
        info = inspect_dataset(args.out)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2
    print(format_dataset_info(info))
    return 0


def _dataset_info(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")
    try:
        info = inspect_dataset(args.path)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2
    print(format_dataset_info(info))
    return 0


def _model_info(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    _torch, world_model_cls, summary = _load_model_runtime(parser)
    try:
        config = _model_config_from_args(args)
        model = world_model_cls(config, seed=0)
    except (TypeError, ValueError, ValidationError, NotImplementedError) as exc:
        parser.error(str(exc))
        return 2
    print(summary.format_model_info(model))
    return 0


def _forward_smoke(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    torch, world_model_cls, summary = _load_model_runtime(parser)
    try:
        config = _model_config_from_args(args)
        source, episode_index, transition_index, observation, action, nxt = _smoke_transition(
            args, config
        )
        model = world_model_cls(config, seed=args.seed)
        model.eval()
        obs = torch.tensor(observation, dtype=torch.float32)
        act = torch.tensor(action, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(obs, act)
        finite = all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                prediction.latent,
                prediction.next_latent,
                prediction.reconstructed,
                prediction.predicted_observation,
            )
        )
        text = summary.format_forward_smoke(
            source=source,
            seed=args.seed,
            episode=episode_index,
            transition=transition_index,
            model=model,
            observation=observation,
            action=action,
            next_observation=nxt,
            prediction=prediction,
        )
    except (TypeError, ValueError, ValidationError, NotImplementedError, OSError) as exc:
        parser.error(str(exc))
        return 2
    print(text)
    if not finite:
        return 1
    return 0


def _load_model_runtime(parser: argparse.ArgumentParser) -> tuple[object, type, object]:
    """Import torch and the world model. ``parser.error`` exits when torch is missing."""

    try:
        torch = importlib.import_module("torch")
        world = importlib.import_module("worldforge.models.world")
        summary = importlib.import_module("worldforge.models.summary")
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "torch" or missing.startswith("torch."):
            parser.error(
                'model and train commands need the optional ml extra: pip install -e ".[ml]" '
                "(a CPU build of torch is enough). "
                f"Import failed: {exc}"
            )
        raise
    return torch, world.WorldModel, summary


def _train(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.data.exists():
        parser.error(f"path does not exist: {args.data}")
    world_model_cls, train_api = _load_train_runtime(parser)
    try:
        config = train_api.TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            reconstruction_weight=args.recon_weight,
            steps_per_epoch=args.steps_per_epoch,
            optimizer=args.optimizer,
        )
        model = world_model_cls(_model_config_from_args(args), seed=args.seed)
        episodes = load_corpus(args.data)
        result = train_api.train_world_model(
            model,
            episodes,
            config,
            out=args.out,
            data_path=args.data,
        )
    except (TypeError, ValueError, ValidationError, NotImplementedError, OSError) as exc:
        parser.error(str(exc))
        return 2
    print(train_api.format_train_report(result))
    return 0


def _load_train_runtime(parser: argparse.ArgumentParser) -> tuple[type, object]:
    """Import the world model and the trainer. ``parser.error`` exits without torch."""

    _load_model_runtime(parser)
    try:
        world = importlib.import_module("worldforge.models.world")
        train_api = importlib.import_module("worldforge.train")
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "torch" or missing.startswith("torch."):
            parser.error(
                'model and train commands need the optional ml extra: pip install -e ".[ml]" '
                "(a CPU build of torch is enough). "
                f"Import failed: {exc}"
            )
        raise
    return world.WorldModel, train_api


def _model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dynamics=args.dynamics,
    )


def _smoke_transition(
    args: argparse.Namespace,
    config: ModelConfig,
) -> tuple[str, int | None, int | None, list[float], list[float], list[float] | None]:
    if args.fixture is None:
        observation = [1.0] * config.obs_dim
        action = [0.0] * config.action_dim
        return "synthetic", None, None, observation, action, None
    if args.episode < 0 or args.transition < 0:
        raise ValueError("--episode and --transition must be >= 0")
    if not args.fixture.exists():
        raise ValueError(f"path does not exist: {args.fixture}")
    episodes = load_corpus(args.fixture)
    if args.episode >= len(episodes):
        raise ValueError(f"episode {args.episode} is out of range for {len(episodes)} episodes")
    episode = episodes[args.episode]
    if args.transition >= len(episode.transitions):
        raise ValueError(
            f"transition {args.transition} is out of range for "
            f"{len(episode.transitions)} transitions"
        )
    step = episode.transitions[args.transition]
    observation = list(step.observation.values)
    action = list(step.action.values)
    nxt = list(step.next_observation.values)
    if len(observation) != config.obs_dim or len(nxt) != config.obs_dim:
        raise ValueError(
            f"observation length {len(observation)} does not match obs_dim {config.obs_dim}"
        )
    if len(action) != config.action_dim:
        raise ValueError(f"action length {len(action)} does not match action_dim {config.action_dim}")
    return "fixture", args.episode, args.transition, observation, action, nxt


def _parse_seeds(text: str) -> list[int]:
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("--seeds must be a comma-separated list of integers")
    seeds: list[int] = []
    for part in parts:
        if part[0] == "+" or (part[0] == "-" and len(part) > 1):
            body = part[1:]
            sign = part[0]
        else:
            body = part
            sign = ""
        if not body.isdecimal():
            raise ValueError("--seeds must be a comma-separated list of integers")
        seeds.append(int(sign + body))
    return seeds


def _parse_formats(name: str) -> tuple[FormatName, ...]:
    if name == "both":
        return ("jsonl", "json")
    if name == "jsonl":
        return ("jsonl",)
    if name == "json":
        return ("json",)
    raise ValueError("--format must be jsonl, json, or both")
