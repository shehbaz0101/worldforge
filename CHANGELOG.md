# Changelog

## 0.1.0 — 2026-09-26

First public freeze of the latent world-model sandbox.

- Lotka-Volterra environment, trajectory schema, and offline corpus tools.
- CPU latent world model (encoder, residual dynamics, decoder) and a one-step trainer.
- Open-loop multi-horizon prediction metrics.
- CEM planner and closed-loop regret against a zero dose and a random dose.
- Localhost FastAPI service (`worldforge serve`) with a path sandbox, a rate limit, and an offline socket guard.
- Offline demo (`worldforge demo`) on the checked-in sample corpus.

[PROJECT_STATUS.md](PROJECT_STATUS.md) is the longer status note.
