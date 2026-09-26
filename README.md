# WorldForge

WorldForge is a research-grade latent world model for a scientific dynamics sandbox. It learns a compact state from trajectories of a deterministic laboratory-style system, rolls that state forward for many steps, and uses the model for planning. Offline evals measure open-loop prediction and control.

This revision adds the Day 4 training loop on top of the Day 3 latent model and the offline corpus. `worldforge collect` rolls the default Lotka-Volterra environment with a zero dose, a seeded random dose, or an open-loop sine dose, and writes a corpus. `worldforge dataset-info` summarizes a corpus file or directory. `worldforge model-info` prints the encoder, dynamics, and decoder sizes. `worldforge forward-smoke` runs one encode, step, and decode and does not train. `worldforge train` fits the baseline on a corpus and writes a checkpoint. A checked-in fixture set under `tests/fixtures/trajectories/` keeps tests offline. Nothing in this revision calls the network or downloads a weight file.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The environment, schema, and corpus commands do not need PyTorch. The model and the trainer need the `ml` extra. Install a CPU build of torch first so pip does not replace it with a larger CUDA wheel, then install the extra (torch plus numpy; numpy only keeps the torch import from warning):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,ml]"
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

`model-info` prints the Day 3 architecture. The baseline dynamics are a deterministic residual MLP: `z` and an action map to the next `z`. `forward-smoke` runs that map once. `--fixture` selects one stored transition. Without it, the input is the coexistence state `(1, 1)` and a zero dose. `rssm` is accepted as a name and is not implemented.

```bash
worldforge model-info
worldforge forward-smoke --fixture tests/fixtures/trajectories --seed 0
```

```python
from worldforge.models import ModelConfig, WorldModel, load_checkpoint, save_checkpoint

model = WorldModel(ModelConfig(latent_dim=8), seed=0)
save_checkpoint("checkpoints/day3", model)
restored = load_checkpoint("checkpoints/day3")
```

The checkpoint directory is `config.json` plus `weights.pt`. It stores the config and the `state_dict`. It does not store an optimizer. Importing `worldforge` does not import torch; importing `WorldModel` does.

`train` fits the baseline on a corpus and writes a checkpoint. The loss is the MSE of the one-step prediction, plus a reconstruction MSE scaled by `--recon-weight` (default 1; `0` trains prediction only). `--seed` fixes both the initial weights and the replay batches. `--device cpu` is the only accepted device. On the golden fixtures the defaults are 3 epochs and batch size 8 (6 batches per epoch) and the run finishes in a few seconds.

```bash
worldforge train \
  --data tests/fixtures/trajectories \
  --out checkpoints/day4 \
  --epochs 3 \
  --batch-size 8 \
  --lr 1e-3 \
  --seed 0 \
  --device cpu
```

The command prints one line per epoch and `final_train_loss`. `checkpoints/day4` contains the Day 3 checkpoint files plus `metrics.jsonl` and `train.json`.

```python
from worldforge.data import load_corpus
from worldforge.models import ModelConfig, WorldModel, load_checkpoint
from worldforge.train import TrainConfig, train_world_model

episodes = load_corpus("tests/fixtures/trajectories")
model = WorldModel(ModelConfig(), seed=0)
result = train_world_model(
    model,
    episodes,
    TrainConfig(epochs=3, batch_size=8, lr=1e-3, seed=0, device="cpu"),
    out="checkpoints/day4",
)
print(result.final_train_loss)
restored = load_checkpoint("checkpoints/day4")
```

`train_world_model` accepts any sequence of trajectories. Pass `split_trajectories(episodes, seed=0).train` to fit the train split only. The CLI trains every episode in `--data`.

## Roadmap

1. **Day 1 — Scaffold.** Kinetic environment, trajectory schema, CLI, pytest CI.
2. **Day 2 — Dataset.** Collect JSONL trajectories, split them, and sample a replay batch.
3. **Day 3 — Encoder and baseline dynamics.** Latent MLP, one-step residual transition, decoder, checkpoint.
4. **Day 4 — Training loop.** One-step prediction MSE, optional reconstruction weight, Adam or SGD, checkpoint, and `worldforge train`. This revision.
5. **Day 5 — Multi-step loss.** Train on rolled-out latent trajectories.
6. **Day 6 — Saved run.** A trained checkpoint on the offline split, using the Day 3 format.
7. **Day 7 — Prediction eval.** Open-loop rollout error on held-out seeds.
8. **Day 8 — Planner.** Model-based action search.
9. **Day 9 — Control eval.** Offline return of the planner against the zero dose.
10. **Day 10 — Freeze.** Status note and a pinned demo path.

## Docs

- [Architecture](docs/architecture.md) — components, and what this revision ships.
- [Day 1 log](docs/daily/2026-09-26.md)
- [Day 2 log](docs/daily/2026-09-26-day2.md)
- [Day 3 log](docs/daily/2026-09-26-day3.md)
- [Day 4 log](docs/daily/2026-09-26-day4.md)
- [Project status](PROJECT_STATUS.md)

## Tests

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,ml]"
pytest
```

`tests/test_models.py` and `tests/test_train.py` import torch, so the full suite needs the `ml` extra. The train tests run a few optimizer steps on the golden fixtures. CI runs that suite on Python 3.11 and 3.12 after installing the CPU wheel. No test uses the network or downloads weights. Optional lint:

```bash
pip install -e ".[lint]"
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
