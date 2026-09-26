"""Rewrite the checked-in demo corpus.

Run from a checkout with WorldForge installed (``pip install -e .``):

    python scripts/regenerate_samples.py

The command is offline. It reads ``samples/demo.json`` and overwrites
``samples/trajectories/``. Commit the result; ``worldforge demo`` and the
unit tests load those files and do not need to regenerate them.
"""

from __future__ import annotations

from pathlib import Path

from worldforge.data import EpisodeSpec, collect_corpus
from worldforge.demo import load_demo_config

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
CORPUS = SAMPLES / "trajectories"


def main() -> None:
    config = load_demo_config(SAMPLES / "demo.json")
    specs = tuple(
        EpisodeSpec(seed=episode.seed, policy=episode.policy) for episode in config.episodes
    )
    manifest = collect_corpus(
        CORPUS,
        horizon=config.horizon,
        specs=specs,
        formats=("jsonl", "json"),
    )
    print(f"wrote {manifest.n_episodes} episodes to {CORPUS}")


if __name__ == "__main__":
    main()
