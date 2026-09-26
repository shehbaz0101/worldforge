# Project status

WorldForge is Project B: a research-grade latent world model for a scientific
dynamics sandbox. The aim is compact state dynamics learned from trajectories,
multi-step rollouts, model-based planning, and offline prediction and control
evals.

**Status:** Day 5.

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
deviation by horizon). Torch is the optional `ml` extra. The planner and
the HTTP API are later days. See [docs/architecture.md](docs/architecture.md)
and [docs/daily/2026-09-26-day5.md](docs/daily/2026-09-26-day5.md).
