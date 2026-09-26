# WorldForge

WorldForge is a research-grade latent world model for a scientific dynamics sandbox. It learns a compact state from trajectories of a deterministic laboratory-style system, rolls that state forward for many steps, and uses the model for planning. Offline evals measure open-loop prediction and control. Day 1 is the substrate only: a tiny kinetic environment, a trajectory schema, and the package. No model is trained here, and nothing in this revision calls the network.

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

`--policy random` draws a dose from `random.Random(seed)`, so the seed still fixes the episode.

## Roadmap

1. **Day 1 — Scaffold.** Kinetic environment, trajectory schema, CLI, pytest CI.
2. **Day 2 — Dataset.** Collect JSONL trajectories and sample a replay batch.
3. **Day 3 — Encoder.** Map an observation to a compact latent code.
4. **Day 4 — Dynamics.** One-step latent transition.
5. **Day 5 — Multi-step loss.** Train on rolled-out latent trajectories.
6. **Day 6 — Trainer.** Offline training loop and a local checkpoint.
7. **Day 7 — Prediction eval.** Open-loop rollout error on held-out seeds.
8. **Day 8 — Planner.** Model-based action search.
9. **Day 9 — Control eval.** Offline return of the planner against the zero dose.
10. **Day 10 — Freeze.** Status note and a pinned demo path.

## Docs

- [Architecture](docs/architecture.md) — components, and what Day 1 ships.
- [Day 1 log](docs/daily/2026-09-26.md)
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
