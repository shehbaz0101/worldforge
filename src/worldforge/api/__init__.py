"""HTTP API for the offline research loop.

Importing this package loads FastAPI and does not load PyTorch.
``GET /health`` never does. The predict, plan, and train routes import
the optional ``ml`` extra when they are called.

The FastAPI instance lives in :mod:`worldforge.api.server` and is
re-exported here so ``uvicorn worldforge.api:app`` and
``from worldforge.api import app`` see the same object. The module is
not named ``app`` so that re-export does not shadow a submodule.
"""

from worldforge.api.server import app

__all__ = ["app"]
