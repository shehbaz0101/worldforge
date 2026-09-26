# Project status

WorldForge is Project B: a research-grade latent world model for a scientific
dynamics sandbox. The aim is compact state dynamics learned from trajectories,
multi-step rollouts, model-based planning, and offline prediction and control
evals.

**Status:** Day 8.

Day 1 added the package, the default Lotka-Volterra environment, the
trajectory schema, and pytest CI on Python 3.11 and 3.12. Day 2 collects
offline trajectory corpora (`worldforge collect`), summarizes them
(`worldforge dataset-info`), and splits them into train, val, and test.
Golden episodes live in `tests/fixtures/trajectories/` so CI stays offline.
Day 3 adds a CPU latent model: an MLP encoder, a deterministic residual-MLP
dynamics baseline, a small decoder, and `WorldModel` (`encode`, `step`,
`decode`, `predict`). `save_checkpoint` writes `config.json` and `weights.pt`
(`worldforge.checkpoint.v1`). Day 4 trains that model on stored transitions.
The loss is one-step observation MSE plus an optional reconstruction term.
`worldforge train` runs Adam or SGD on CPU, prints `final_train_loss`, and
writes the Day 3 checkpoint plus `metrics.jsonl` and `train.json`. Day 5
rolls a checkpoint, or an untrained seeded model, open-loop on stored
actions and decodes predicted observations at horizons 1..H. `worldforge
rollout` and `worldforge eval-predict` write `worldforge.predict.v1` with
`horizons`, `mse_by_h`, and `n_episodes` (plus MAE and a residual standard
deviation by horizon). Day 6 plans with that model. CEM searches a short
clipped action sequence. The objective is `regulation_l1`: the
undiscounted sum of Lotka-Volterra regulation rewards on decoded states,
the same negative L1 distance from `(1, 1)` that the environment returns.
`worldforge plan` writes one sequence (`worldforge.action_sequence.v1`).
`worldforge eval-plan` replans on the real environment for a few steps
and writes `worldforge.plan.v1`. `regret` is `baseline_best -
planner_return`, where `baseline_best` is the better of a zero dose and
the Day 1 random dose. Negative regret means the planner beat both.
Torch is the optional `ml` extra. See
[docs/daily/2026-09-26-day6.md](docs/daily/2026-09-26-day6.md).

Day 7 serves that loop over HTTP. `worldforge serve` binds to `127.0.0.1:8000`
by default and does not open an outbound connection. `GET /health` returns
`ok` and the package version without loading torch. `POST /rollout` and
`POST /eval/predict` return `worldforge.predict.v1` from a checkpoint or a
seed, and from a corpus path or a small inline trajectory list. `POST /plan`
returns `worldforge.action_sequence.v1`. `POST /eval/plan` returns
`worldforge.plan.v1`. `POST /train` runs a short CPU fit (at most 5 epochs
and 8 steps per epoch) and returns `final_train_loss` plus the checkpoint
directory. FastAPI and uvicorn are core dependencies. The predict, plan, and
train routes need the `ml` extra and answer 503 when it is missing. See
[docs/daily/2026-09-26-day7.md](docs/daily/2026-09-26-day7.md).

Day 8 hardens that service for local use. Corpus, checkpoint, and output
paths on the CLI and on the HTTP body must resolve inside `--data-root`
(`WORLDFORGE_DATA_ROOT`, or the current directory). A `..` traversal, an
absolute path outside that root, or a symlink that leaves it is a CLI
error or HTTP 422. `POST /rollout`, `POST /eval/predict`, `POST /plan`,
`POST /eval/plan`, and `POST /train` share a per-client limit, 60 requests
per 60 seconds by default. Over the limit the response is HTTP 429 with
`Retry-After`. `GET /health` is not limited. Importing the API installs a
socket guard that refuses non-loopback connects, so the server does not
download weights. There is still no authentication. See
[docs/architecture.md](docs/architecture.md) and
[docs/daily/2026-09-26-day8.md](docs/daily/2026-09-26-day8.md).
