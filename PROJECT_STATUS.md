# Project status

WorldForge is a research-grade latent world model for a scientific dynamics
sandbox. It learns a compact state from Lotka-Volterra trajectories, rolls
that state forward, plans with CEM, and records open-loop prediction and
closed-loop regret. The same loop is a CLI and a localhost FastAPI service.
An offline demo trains on a checked-in sample corpus.

**Status:** Project B is frozen at v0.1.0.

## Days 1–10

| Day | Delivered |
| --- | --- |
| 1 | Package scaffold, Lotka-Volterra env, trajectory schema, pytest CI |
| 2 | Offline corpus (`worldforge collect`, `dataset-info`), splits, fixtures |
| 3 | MLP encoder, residual dynamics, decoder, `WorldModel`, checkpoints |
| 4 | One-step training loop and `worldforge train` |
| 5 | Multi-horizon rollout and `worldforge rollout` / `eval-predict` |
| 6 | CEM planner and closed-loop regret (`worldforge plan` / `eval-plan`) |
| 7 | FastAPI service and `worldforge serve` |
| 8 | Path sandbox, per-client rate limit, offline socket guard |
| 9 | `worldforge demo` on `samples/trajectories` |
| 10 | Freeze at v0.1.0, this status note, changelog |

## Install

Python 3.11 or newer. Install a CPU build of torch first so the extra does
not pull a CUDA wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"
```

`worldforge demo`, `worldforge train`, the open-loop eval, the planner, and
the predict, plan, and train HTTP routes need that extra. The environment,
the schema, the corpus commands, `worldforge serve`, and `GET /health` run
from `pip install -e .` and do not import torch. Tests use
`pip install -e ".[dev,ml]"`. Lint uses `pip install -e ".[lint]"`.

## Quickstart

```bash
worldforge demo
worldforge serve
```

`worldforge demo` reads `samples/demo.json` and the two-episode corpus in
`samples/trajectories/`. It trains a small CPU checkpoint, scores an
open-loop rollout, plans one short action sequence, and writes closed-loop
regret. Output goes to `demo-run/` unless `--out` is set. `worldforge serve`
binds to `127.0.0.1:8000` and does not open an outbound connection.

## Tests and CI

CI runs on pull requests and on pushes to `main`. The test job installs the
CPU torch wheel, then `pip install -e ".[dev,ml]"`, and runs
`pytest -m "not integration"` on Python 3.11 and 3.12. A lint job runs
`ruff check .` (rules E4, E7, E9, F, I). No test is marked `integration`.
HTTP tests use FastAPI `TestClient` and do not bind a port. Nothing in the
suite downloads weights.

## Known limits

- CPU demo caps. `worldforge demo` and `POST /train` allow at most 5 epochs
  and 8 steps per epoch. The demo defaults are one epoch and one replay step
  on two episodes of horizon 4. A longer fit stays on `worldforge train`.
- Localhost bind. `worldforge serve` defaults to `127.0.0.1:8000`. There is
  no authentication.
- Offline-by-design. Importing the API installs a socket guard that refuses
  non-loopback connects. The package does not download weights.
- Free and public only. No API keys, no paid services, and no private data.
- `dynamics="rssm"` is reserved and raises `NotImplementedError`. Training
  is one-step observation MSE, not a multi-step rollout loss.

## Release tag

`pyproject.toml` and `worldforge.__version__` are `0.1.0`. The annotated tag
`v0.1.0` should point at the squash-merge commit of this freeze on `main`.
A squash merge discards the branch tip, so the tag is not created on
`feat/day10-freeze`. After the squash SHA is on `main`:

```bash
git fetch origin main
git tag -a v0.1.0 <squash-sha> -m "WorldForge v0.1.0"
git push origin v0.1.0
```

Do not force-update a tag that already exists.
