"""Command line interface.

``version`` prints the package version.
``env-demo`` rolls the default kinetic environment for a few steps and
prints a trajectory summary plus the final ASCII state.
``collect`` writes an offline trajectory corpus.
``dataset-info`` summarizes a corpus directory or a trajectory file.
These commands do not use the network, and they do not train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worldforge import __version__
from worldforge.data import collect_corpus, format_dataset_info, inspect_dataset
from worldforge.data.dataset import DEFAULT_HORIZON, DEFAULT_N_EPISODES, FormatName
from worldforge.envs import DEFAULT_ENV_ID, make_env
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
    return parser


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
