# WorldForge

WorldForge is a research-grade latent world model for a scientific dynamics sandbox. It learns a compact state from trajectories of a deterministic laboratory-style system, rolls that state forward for many steps, and uses the model for planning. Offline evals measure open-loop prediction and control.

This revision collects those trajectories. `worldforge collect` rolls the default Lotka-Volterra environment with a zero dose, a seeded random dose, or an open-loop sine dose, and writes a corpus. `worldforge dataset-info` summarizes a corpus file or directory. A checked-in fixture set under `tests/fixtures/trajectories/` keeps tests offline. No model is trained here, and nothing in this revision calls the network.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

`version` prints the package version. `env-demo` resets the default Lotka-Volterra environment on a seed, applies a short zero dose, and prints a trajectory summary. Prey is `x0` and predator is `x1`. The same commands are `python -m worldforge version` and `python -m worldforge env-demo`.

```bash
worldforge version
worldforge env-demo
```

```bash
worldforge env-demo --seed 1 --steps 8 --policy random
```

`--policy random` draws a dose from `random.Random(seed)`, so the seed still fixes the episode. `--policy sine_dose` is an open-loop schedule and does not use the seed for the dose.

`collect` writes a directory. `corpus.jsonl` is one trajectory JSON object per line. `episodes/ep_XXXX.json` is one trajectory document per episode. `manifest.json` records the seeds and policies (`worldforge.corpus.v1`). `dataset-info` prints episode and transition counts for that directory, for `corpus.jsonl`, or for a single trajectory file.

```bash
worldforge collect --out corpus --n-episodes 4 --horizon 8 --seed 0 --policy random
worldforge dataset-info corpus
worldforge dataset-info tests/fixtures/trajectories
```

The same calls work as `python -m worldforge collect` and `python -m worldforge dataset-info`. Collection is offline. The fixture directory is the small golden set (6 episodes, horizon 8, policies `zero`, `random`, and `sine_dose`). Regenerate it with `python scripts/regenerate_fixtures.py` and commit the files. The format is described in [tests/fixtures/trajectories/README.md](tests/fixtures/trajectories/README.md).

```python
from worldforge.data import load_corpus, split_trajectories

episodes = load_corpus("tests/fixtures/trajectories")
parts = split_trajectories(episodes, seed=0, ratios=(0.5, 0.25, 0.25))
```

`split_trajectories` is deterministic for a seed. The default ratios are 0.8, 0.1, 0.1. `sample_transition_batch` draws a deterministic replay batch from stored transitions.

## Roadmap

1. **Day 1 — Scaffold.** Kinetic environment, trajectory schema, CLI, pytest CI.
2. **Day 2 — Dataset.** Collect JSONL trajectories, split them, and sample a replay batch.
3. **Day 3 — Encoder.** Map an observation to a compact latent code.
4. **Day 4 — Dynamics.** One-step latent transition.
5. **Day 5 — Multi-step loss.** Train on rolled-out latent trajectories.
6. **Day 6 — Trainer.** Offline training loop and a local checkpoint.
7. **Day 7 — Prediction eval.** Open-loop rollout error on held-out seeds.
8. **Day 8 — Planner.** Model-based action search.
9. **Day 9 — Control eval.** Offline return of the planner against the zero dose.
10. **Day 10 — Freeze.** Status note and a pinned demo path.

## Docs

- [Architecture](docs/architecture.md) — components, and what this revision ships.
- [Day 1 log](docs/daily/2026-09-26.md)
- [Day 2 log](docs/daily/2026-09-26-day2.md)
- [Project status](PROJECT_STATUS.md)

## Tests

```bash
pytest
```

CI runs that suite on Python 3.11 and 3.12. No test uses the network. Optional lint:

```bash
pip install -e ".[lint]"
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
