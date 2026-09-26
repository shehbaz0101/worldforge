# WorldForge

WorldForge is a research-grade latent world model for a scientific dynamics sandbox. It learns a compact state from trajectories of a deterministic Lotka-Volterra system, rolls that state forward, and uses the model for planning. Offline evals measure open-loop prediction and closed-loop regret.

v0.1.0 freezes that surface. [PROJECT_STATUS.md](PROJECT_STATUS.md) lists what shipped. [CHANGELOG.md](CHANGELOG.md) is the release note.

## Install

Python 3.11 or newer. Install a CPU build of torch first so pip does not replace it with a CUDA wheel, then install the `ml` extra:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"
```

That extra is torch plus numpy. Numpy keeps the torch import from warning. The model, the trainer, the open-loop eval, the planner, `worldforge demo`, and the predict, plan, and train HTTP routes need it. The environment, the schema, the corpus commands, `worldforge serve`, and `GET /health` do not. FastAPI and uvicorn are installed with the package, so those paths also run from `pip install -e .`. Tests need `pip install -e ".[dev,ml]"`. Lint needs `pip install -e ".[lint]"`.

## Quickstart

```bash
worldforge demo
worldforge serve
```

`worldforge demo` trains a tiny checkpoint on the checked-in sample corpus and prints prediction and planning metrics. It does not call the network. `worldforge serve` binds to `127.0.0.1:8000`. `GET /health` does not need torch. The same calls are `python -m worldforge demo` and `python -m worldforge serve`.

## Demo

The demo reads `samples/trajectories` (two episodes, horizon 4). It needs the `ml` extra from [Install](#install).

```bash
worldforge demo
```

The default output directory is `demo-run/` (`checkpoint/`, `predict.json`, `plan.json`, `eval-plan.json`, and `summary.txt`). The fit is capped at 5 epochs and 8 replay steps. Caps and the episode list live in `samples/demo.json`. When `samples/trajectories` is already inside the data root, the demo reads it in place. Otherwise it copies those files under the output directory, or collects the same two episodes if the samples are missing. `--data` selects another corpus. Paths stay inside `--data-root` (or `WORLDFORGE_DATA_ROOT`, or the current directory). A longer fit stays on `worldforge train`.

`worldforge serve` is the local HTTP API. The default bind is `127.0.0.1:8000`. `GET /health` does not need torch. The predict, plan, and train routes do, and they answer HTTP 503 when the `ml` extra is missing. `--data-root` is the path sandbox, and `--rate-limit` / `--rate-window` cap the expensive POST routes (HTTP 429, `Retry-After`). The server refuses non-loopback connections. Details are in [HTTP API](#http-api).

```bash
worldforge serve
worldforge serve --host 127.0.0.1 --port 8000 --data-root . --rate-limit 60 --rate-window 60
```

The rest of the CLI sits under the same install. `worldforge collect` rolls the default Lotka-Volterra environment with a zero dose, a seeded random dose, or an open-loop sine dose, and writes a corpus. `worldforge dataset-info` summarizes a corpus file or directory. `worldforge model-info` prints the encoder, dynamics, and decoder sizes. `worldforge forward-smoke` runs one encode, step, and decode and does not train. `worldforge train` fits the baseline on a corpus and writes a checkpoint. `worldforge rollout` and `worldforge eval-predict` roll that checkpoint (or an untrained seeded model) open-loop and write per-horizon MSE. `worldforge plan` searches one short action sequence with CEM. `worldforge eval-plan` runs that planner closed-loop and writes regret against a zero dose and a random dose. Corpus, checkpoint, and output paths stay inside a data root. A checked-in fixture set under `tests/fixtures/trajectories/` keeps tests offline. The package does not call the network or download a weight file.

## Environment smoke

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

`rollout` and `eval-predict` score an open-loop latent rollout. The encoder runs on the first observation only. Each later step applies the stored action in latent space and decodes a predicted observation. Horizon 1 matches one-step `predict`. The JSON report lists `horizons`, `mse_by_h`, `mae_by_h`, `residual_std_by_h`, and `n_episodes`. `--checkpoint` loads a Day 4 directory. Without it, `--seed` builds an untrained model. `--horizon` caps the score; the default uses every stored step. On the golden fixtures that is 6 episodes and horizons 1..8, and the command does not train.

```bash
worldforge eval-predict \
  --checkpoint checkpoints/day4 \
  --data tests/fixtures/trajectories \
  --out reports/predict.json \
  --horizon 8

worldforge rollout \
  --data tests/fixtures/trajectories \
  --out reports/untrained.json \
  --seed 0 \
  --horizon 4
```

```python
from worldforge.data import load_corpus
from worldforge.eval import open_loop_metrics, rollout_latent
from worldforge.models import load_checkpoint

model = load_checkpoint("checkpoints/day4")
episodes = load_corpus("tests/fixtures/trajectories")
report = open_loop_metrics(model, episodes, horizon=8, checkpoint="checkpoints/day4")
print(report.horizons, report.mse_by_h, report.n_episodes)
```

`import worldforge` does not import torch. Importing `worldforge.eval` or `worldforge.plan` does.

`plan` searches one clipped action sequence with the cross-entropy method. Candidates are rolled in latent space. The score is the sum of regulation rewards on the decoded states: negative L1 distance from the coexistence equilibrium `(1, 1)`, the same reward the environment uses. The search does not step the environment. Defaults are a horizon of 3, 8 samples, and 2 iterations, so the command stays small. `--obs` sets the starting state. `--data` uses the first observation of a stored episode. Without either, the start is the equilibrium.

`eval-plan` is closed loop. At each environment step it plans again and applies only the first action. It then rolls the same seed with a zero dose and with the seeded random dose. `regret` is `baseline_best - planner_return`. `baseline_best` is the better of those two baselines. A negative regret means the planner scored higher than both. The JSON format is `worldforge.plan.v1`.

```bash
worldforge plan \
  --obs 1.5,0.6 \
  --out reports/plan.json \
  --horizon 3 \
  --samples 8 \
  --iterations 2 \
  --seed 0

worldforge eval-plan \
  --out reports/plan-regret.json \
  --n-steps 4 \
  --env-seed 0 \
  --horizon 3 \
  --samples 8 \
  --iterations 2 \
  --seed 0
```

```python
import torch

from worldforge.models import ModelConfig, WorldModel
from worldforge.plan import CEMConfig, plan_actions, planning_regret

model = WorldModel(ModelConfig(), seed=0)
plan = plan_actions(model, torch.tensor([1.5, 0.6]), CEMConfig(horizon=3, seed=0))
print(plan.actions, plan.predicted_return)
report = planning_regret(model, n_steps=4, env_seed=0, cem=CEMConfig(seed=0))
print(report.planner_return, report.zero_return, report.random_return, report.regret)
```

## HTTP API

`worldforge serve` runs the FastAPI app. The default bind is `127.0.0.1:8000`. `--host` and `--port` change it. WorldForge is offline-by-design: importing the API installs a socket guard that refuses non-loopback TCP connects, and the process does not download weights. Loopback stays open so a client on this machine can call the server. The guard stays on for the life of the process. There is no authentication.

`GET /health` returns `{"status": "ok", "version": ...}` and does not import torch. It is not rate limited. The other routes need the `ml` extra. If torch is missing they return HTTP 503. A bad body, a missing path, or a path outside the data root is HTTP 422.

User paths (`data`, `checkpoint`, `out`, and the same CLI flags) must resolve inside the data root. The root is `--data-root`, or `WORLDFORGE_DATA_ROOT`, or the current directory. Relative paths resolve against that root. Absolute paths are accepted only when they resolve inside it. `..` and symlinks are resolved before the check, so a traversal or a symlink that leaves the root is rejected. `POST /train` with `out` omitted creates the checkpoint directory inside the data root.

`POST /rollout`, `POST /eval/predict`, `POST /plan`, `POST /eval/plan`, and `POST /train` share one in-process sliding window per client address. The default is 60 requests per 60 seconds. Over the limit the response is HTTP 429 with a `Retry-After` header (seconds). `TestClient` has no separate peer address, so those calls share one bucket. Knobs:

| Knob | Default | Meaning |
| --- | --- | --- |
| `--data-root`, `WORLDFORGE_DATA_ROOT` | current directory | Allowed root for corpus, checkpoint, and output paths. |
| `--rate-limit`, `WORLDFORGE_RATE_LIMIT` | 60 | Combined cap on the expensive POSTs, per client, per window. Minimum 1. |
| `--rate-window`, `WORLDFORGE_RATE_WINDOW_SECONDS` | 60 | Window length in seconds. |

`POST /rollout` and `POST /eval/predict` are the same call. The body is a corpus path (`data`) or a list of trajectory documents (`trajectories`), plus either a `checkpoint` directory or a `seed` and the architecture fields. `horizon` caps the score. The response is `worldforge.predict.v1`, the same JSON the CLI writes. Inline episodes are recorded as `data: "inline"`. When `checkpoint` is set, the architecture fields are ignored.

`POST /plan` returns `worldforge.action_sequence.v1`. `observation` is the start state. `data` uses the first observation of one stored episode instead. With neither, the start is the regulation target, `(1, 1)` for the default width. `seed`, `horizon`, `n_samples`, `n_iterations`, and `elite_fraction` are the CEM settings. The CLI flags `--samples` and `--iterations` are `n_samples` and `n_iterations` here, matching the report.

`POST /eval/plan` returns `worldforge.plan.v1`. `n_steps` is the closed-loop length. `env_seed` resets the environment and seeds the random baseline. `seed` is the CEM seed. Defaults match `worldforge eval-plan` (4 steps, horizon 3, 8 samples, 2 iterations).

`POST /train` is a short CPU fit, not the unbounded CLI trainer. The default is 1 epoch, 1 step, and batch size 1. `epochs` is at most 5 and `steps_per_epoch` is at most 8. The response is `final_train_loss` and the checkpoint directory. `out` chooses that directory and must stay inside the data root. When `out` is omitted the server creates a directory inside the data root and returns its path. The caller deletes it. Longer runs stay on `worldforge train`.

```bash
worldforge serve
worldforge serve --host 127.0.0.1 --port 8000 --data-root . --rate-limit 60 --rate-window 60
```

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/eval/predict \
  -H 'content-type: application/json' \
  -d '{"data":"tests/fixtures/trajectories","horizon":4,"seed":0}'

curl -s -X POST http://127.0.0.1:8000/plan \
  -H 'content-type: application/json' \
  -d '{"observation":[1.5,0.6],"horizon":3,"n_samples":8,"n_iterations":2,"seed":0}'

curl -s -X POST http://127.0.0.1:8000/eval/plan \
  -H 'content-type: application/json' \
  -d '{"n_steps":4,"env_seed":0,"horizon":3,"n_samples":8,"n_iterations":2,"seed":0}'

curl -s -X POST http://127.0.0.1:8000/train \
  -H 'content-type: application/json' \
  -d '{"data":"tests/fixtures/trajectories","epochs":1,"steps_per_epoch":1,"out":"checkpoints/api-smoke"}'
```

```python
from fastapi.testclient import TestClient

from worldforge.api import app

with TestClient(app) as client:
    assert client.get("/health").json()["status"] == "ok"
    plan = client.post(
        "/plan",
        json={"observation": [1.5, 0.6], "horizon": 2, "n_samples": 4, "n_iterations": 1},
    )
    print(plan.json()["actions"], plan.json()["predicted_return"])
```

`TestClient` is how the unit tests call the app. They do not bind a port. The dev extra installs `httpx2`, which Starlette uses when it is present. `import worldforge` does not import the API or torch. `import worldforge.api` imports FastAPI and does not import torch.

## Roadmap

1. **Day 1 — Scaffold.** Kinetic environment, trajectory schema, CLI, pytest CI.
2. **Day 2 — Dataset.** Collect JSONL trajectories, split them, and sample a replay batch.
3. **Day 3 — Encoder and baseline dynamics.** Latent MLP, one-step residual transition, decoder, checkpoint.
4. **Day 4 — Training loop.** One-step prediction MSE, optional reconstruction weight, Adam or SGD, checkpoint, and `worldforge train`.
5. **Day 5 — Open-loop prediction.** Multi-horizon latent rollout, per-horizon MSE, and `worldforge rollout` / `worldforge eval-predict`.
6. **Day 6 — Planner.** CEM on latent rollouts, closed-loop regret against the zero dose and a random dose, `worldforge plan` / `worldforge eval-plan`.
7. **Day 7 — HTTP API.** FastAPI service: health, open-loop metrics, one CEM plan, closed-loop regret, and a short train. `worldforge serve`.
8. **Day 8 — Hardening.** Path sandbox, in-process rate limits, offline socket guard.
9. **Day 9 — Demo.** `worldforge demo` on `samples/`: tiny train, open-loop predict, one plan, and closed-loop regret.
10. **Day 10 — Freeze.** v0.1.0. Status, changelog, and this README. A multi-step rollout loss and authentication stay out of the release.

## Docs

- [Project status](PROJECT_STATUS.md) — v0.1.0 freeze: what shipped, install, limits.
- [Changelog](CHANGELOG.md) — v0.1.0 highlights.
- [Architecture](docs/architecture.md) — components in this release.
- [Day 1 log](docs/daily/2026-09-26.md)
- [Day 2 log](docs/daily/2026-09-26-day2.md)
- [Day 3 log](docs/daily/2026-09-26-day3.md)
- [Day 4 log](docs/daily/2026-09-26-day4.md)
- [Day 5 log](docs/daily/2026-09-26-day5.md)
- [Day 6 log](docs/daily/2026-09-26-day6.md)
- [Day 7 log](docs/daily/2026-09-26-day7.md)
- [Day 8 log](docs/daily/2026-09-26-day8.md)
- [Day 9 log](docs/daily/2026-09-26-day9.md)
- [Day 10 log](docs/daily/2026-09-26-day10.md)

## Tests

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,ml]"
pytest -m "not integration"
```

`tests/test_models.py`, `tests/test_train.py`, `tests/test_predict.py`, `tests/test_plan.py`, `tests/test_api.py`, and the `worldforge demo` loop in `tests/test_demo.py` import torch, so the full suite needs the `ml` extra. The corpus checks in `tests/test_demo.py` do not. If torch is missing, those loop tests skip and `worldforge demo` exits with the same `ml` extra message as `worldforge train`. The train tests run a few optimizer steps on the golden fixtures. The prediction tests roll those same fixtures and do not train. The planner tests use a tiny network or a fixed toy dynamics map, with a handful of CEM samples. The API tests use FastAPI's `TestClient` (health, plan, predict, a one-step regret report, a one-step train, path escapes, and HTTP 429). They do not bind a port and they are not marked `integration`. Pytest keeps its temporary directory under `.pytest-tmp` in the current directory so those paths stay inside the default data root. CI runs that suite on Python 3.11 and 3.12 after installing the CPU wheel. No test uses the network or downloads weights. Optional lint:

```bash
pip install -e ".[lint]"
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
