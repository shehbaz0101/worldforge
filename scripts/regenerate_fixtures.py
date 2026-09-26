"""Rewrite the checked-in trajectory fixtures.

Run from a checkout with WorldForge installed (``pip install -e .``):

    python scripts/regenerate_fixtures.py

The command is offline. It rolls the default environment and overwrites
``tests/fixtures/trajectories/``. Commit the result; CI loads those files and
does not regenerate them.
"""

from __future__ import annotations

from pathlib import Path

from worldforge.data import EpisodeSpec, collect_corpus

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "trajectories"
HORIZON = 8
SPECS = (
    EpisodeSpec(seed=0, policy="zero"),
    EpisodeSpec(seed=1, policy="zero"),
    EpisodeSpec(seed=2, policy="random"),
    EpisodeSpec(seed=3, policy="random"),
    EpisodeSpec(seed=4, policy="sine_dose"),
    EpisodeSpec(seed=5, policy="sine_dose"),
)


def main() -> None:
    manifest = collect_corpus(
        FIXTURE_DIR,
        horizon=HORIZON,
        specs=SPECS,
        formats=("jsonl", "json"),
    )
    print(f"wrote {manifest.n_episodes} episodes to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
