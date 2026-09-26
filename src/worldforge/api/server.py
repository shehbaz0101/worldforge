"""FastAPI application for the offline research loop.

``GET /health`` reports the package version and does not load a model.
``POST /rollout`` and ``POST /eval/predict`` score an open-loop latent
rollout. ``POST /plan`` returns one CEM action sequence. ``POST /eval/plan``
returns closed-loop regret. ``POST /train`` runs a short CPU fit.

Importing this module installs an offline socket guard. The process does
not open an outbound connection and does not download weights. Corpus,
checkpoint, and output paths must resolve inside the data root
(``WORLDFORGE_DATA_ROOT`` or the current directory). The expensive POST
routes share a per-client rate limit. Over the limit the response is
HTTP 429 with ``Retry-After``. ``GET /health`` is not limited. There is
no authentication.

Run ``uvicorn worldforge.api:app`` or ``worldforge serve``. The CLI binds
to ``127.0.0.1:8000``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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
from worldforge.offline import install_offline_guard
from worldforge.ratelimit import LIMITED_PATHS, RateLimitConfigError, get_limiter

_T = TypeVar("_T")

_RATE_LIMITED: dict[int | str, dict[str, object]] = {
    429: {
        "description": (
            "Too many requests on the expensive POST routes. "
            "Retry-After is the wait in seconds. GET /health is not limited."
        ),
        "headers": {
            "Retry-After": {
                "description": "Seconds to wait before retrying.",
                "schema": {"type": "integer", "minimum": 1},
            }
        },
    }
}

app = FastAPI(
    title="WorldForge",
    version=__version__,
    description=(
        "Offline research API for open-loop prediction, CEM planning, "
        "closed-loop regret, and a short CPU training run. "
        "File paths must stay inside the data root. "
        "POST /rollout, /eval/predict, /plan, /eval/plan, and /train "
        "share a per-client rate limit."
    ),
)


class _RateLimitMiddleware:
    """Count expensive POSTs. Leave every other request, including health, alone."""

    def __init__(self, asgi_app: Callable[..., Awaitable[None]]) -> None:
        self._app = asgi_app

    async def __call__(self, scope: dict[str, object], receive: object, send: object) -> None:
        if scope.get("type") == "http" and scope.get("method") == "POST":
            path = scope.get("path")
            if isinstance(path, str) and path in LIMITED_PATHS:
                blocked = _limited_response(scope)
                if blocked is not None:
                    await blocked(scope, receive, send)
                    return
        await self._app(scope, receive, send)


def _limited_response(scope: dict[str, object]) -> JSONResponse | None:
    try:
        limiter = get_limiter()
    except RateLimitConfigError as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    client = scope.get("client")
    if isinstance(client, tuple) and client and isinstance(client[0], str):
        key = client[0]
    else:
        key = "global"
    retry_after = limiter.check(key)
    if retry_after is None:
        return None
    window = limiter.window_seconds
    window_text = str(int(window)) if window == int(window) else str(window)
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                f"rate limit exceeded: {limiter.limit} requests "
                f"per {window_text} seconds for this client"
            )
        },
        headers={"Retry-After": str(retry_after)},
    )


app.add_middleware(_RateLimitMiddleware)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the process is up. Does not import torch. Not rate limited."""

    return HealthResponse(status="ok", version=__version__)


@app.post("/rollout", response_model=PredictResponse, responses=_RATE_LIMITED)
def rollout(body: PredictRequest) -> PredictResponse:
    """Open-loop metrics. Same body as ``POST /eval/predict``.

    ``data`` and ``checkpoint`` must resolve inside the data root.
    HTTP 429 when the client is over the shared POST limit.
    """

    return _predict(body)


@app.post("/eval/predict", response_model=PredictResponse, responses=_RATE_LIMITED)
def eval_predict(body: PredictRequest) -> PredictResponse:
    """Open-loop metrics from a checkpoint or a seeded model.

    ``data`` and ``checkpoint`` must resolve inside the data root.
    HTTP 429 when the client is over the shared POST limit.
    """

    return _predict(body)


@app.post("/plan", response_model=ActionSequenceResponse, responses=_RATE_LIMITED)
def plan(body: PlanRequest) -> ActionSequenceResponse:
    """Plan one clipped action sequence from an observation.

    ``data`` and ``checkpoint`` must resolve inside the data root.
    HTTP 429 when the client is over the shared POST limit.
    """

    return _call(lambda: api_service.plan_sequence(body))


@app.post("/eval/plan", response_model=PlanReportResponse, responses=_RATE_LIMITED)
def eval_plan(body: EvalPlanRequest) -> PlanReportResponse:
    """Closed-loop planning regret against a zero dose and a random dose.

    ``checkpoint`` must resolve inside the data root.
    HTTP 429 when the client is over the shared POST limit.
    """

    return _call(lambda: api_service.plan_regret(body))


@app.post("/train", response_model=TrainResponse, responses=_RATE_LIMITED)
def train(body: TrainRequest) -> TrainResponse:
    """Run a short CPU fit and return the final loss and checkpoint path.

    ``data`` and ``out`` must resolve inside the data root. An omitted
    ``out`` is created inside that root. HTTP 429 when the client is over
    the shared POST limit.
    """

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


install_offline_guard()
