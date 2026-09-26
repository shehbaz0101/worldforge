# Project status

WorldForge is Project B: a research-grade latent world model for a scientific
dynamics sandbox. The aim is compact state dynamics learned from trajectories,
multi-step rollouts, model-based planning, and offline prediction and control
evals.

**Status:** Day 1 in progress.

Day 1 adds the package, the default Lotka-Volterra environment, the
trajectory schema, and pytest CI on Python 3.11 and 3.12. The encoder,
dynamics model, trainer, planner, eval harness, and HTTP API are later
days. See [docs/architecture.md](docs/architecture.md).
