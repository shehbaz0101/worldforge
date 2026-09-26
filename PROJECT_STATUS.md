# Project status

WorldForge is Project B: a research-grade latent world model for a scientific
dynamics sandbox. The aim is compact state dynamics learned from trajectories,
multi-step rollouts, model-based planning, and offline prediction and control
evals.

**Status:** Day 4.

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
writes the Day 3 checkpoint plus `metrics.jsonl` and `train.json`. Torch is
the optional `ml` extra. The planner, multi-step eval, and HTTP API are later
days. See [docs/architecture.md](docs/architecture.md) and
[docs/daily/2026-09-26-day4.md](docs/daily/2026-09-26-day4.md).
