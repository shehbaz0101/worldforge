"""Command line interface.

``version`` prints the package version.
``env-demo`` rolls the default kinetic environment for a few steps and
prints a trajectory summary plus the final ASCII state. It does not use
the network, and it does not train a model.
"""

from __future__ import annotations

import argparse

from worldforge import __version__
from worldforge.envs import DEFAULT_ENV_ID, make_env
from worldforge.rollout import format_summary, rollout

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
        choices=("zero", "random"),
        default="zero",
        help="zero dose, or a seeded uniform dose (default: zero)",
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
