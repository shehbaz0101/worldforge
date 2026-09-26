"""FastAPI application for the offline research loop.

``GET /health`` reports the package version and does not load a model.
``POST /rollout`` and ``POST /eval/predict`` score an open-loop latent
rollout. ``POST /plan`` returns one CEM action sequence. ``POST /eval/plan``
returns closed-loop regret. ``POST /train`` runs a short CPU fit.

The process does not open an outbound connection. Checkpoint and corpus
paths are local files the caller can already read. There is no
authentication and no rate limit in this revision.

Run ``uvicorn worldforge.api:app`` or ``worldforge serve``. The CLI binds
to ``127.0.0.1:8000``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

import worldforge.api.service as api_service
from worldforge import __version__
from worldforge.api.schemas import (
    ActionSequenceResponse,
    EvalPlanRequest,
    HealthResponse,
    PlanReportResponse,
    PlanRequest,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

_T = TypeVar("_T")

app = FastAPI(
    title="WorldForge",
    version=__version__,
    description=(
        "Offline research API for open-loop prediction, CEM planning, "
        "closed-loop regret, and a short CPU training run."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the process is up. Does not import torch."""

    return HealthResponse(status="ok", version=__version__)


@app.post("/rollout", response_model=PredictResponse)
def rollout(body: PredictRequest) -> PredictResponse:
    """Open-loop metrics. Same body as ``POST /eval/predict``."""

    return _predict(body)


@app.post("/eval/predict", response_model=PredictResponse)
def eval_predict(body: PredictRequest) -> PredictResponse:
    """Open-loop metrics from a checkpoint or a seeded model."""

    return _predict(body)


@app.post("/plan", response_model=ActionSequenceResponse)
def plan(body: PlanRequest) -> ActionSequenceResponse:
    """Plan one clipped action sequence from an observation."""

    return _call(lambda: api_service.plan_sequence(body))


@app.post("/eval/plan", response_model=PlanReportResponse)
def eval_plan(body: EvalPlanRequest) -> PlanReportResponse:
    """Closed-loop planning regret against a zero dose and a random dose."""

    return _call(lambda: api_service.plan_regret(body))


@app.post("/train", response_model=TrainResponse)
def train(body: TrainRequest) -> TrainResponse:
    """Run a short CPU fit and return the final loss and checkpoint path."""

    return _call(lambda: api_service.train(body))


def _predict(body: PredictRequest) -> PredictResponse:
    return _call(lambda: api_service.open_loop(body))


def _call(fn: Callable[[], _T]) -> _T:
    try:
        return fn()
    except api_service.MlExtraMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError, ValidationError, NotImplementedError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
