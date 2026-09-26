"""WorldForge: latent world models for a scientific dynamics sandbox.

The environment, trajectory schema, and offline corpus import without
PyTorch. The Day 3 latent model is :mod:`worldforge.models`, the Day 4
trainer is :mod:`worldforge.train`, the Day 5 open-loop eval is
:mod:`worldforge.eval`, and the Day 6 planner is :mod:`worldforge.plan`.
Those four need the optional ``ml`` extra (``torch``). The Day 7 HTTP
API is :mod:`worldforge.api`. Importing it loads FastAPI, installs the
offline socket guard, and does not load PyTorch; the predict, plan, and
train routes do, when called. Day 8 keeps user paths inside a data root
and rate-limits the expensive POST routes.
"""

from worldforge.data import (
    collect_corpus,
    load_corpus,
    sample_transition_batch,
    split_trajectories,
)
from worldforge.envs import DEFAULT_ENV_ID, LotkaVolterraEnv, make_env
from worldforge.rollout import rollout
from worldforge.schemas import Action, Observation, Trajectory, Transition

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ENV_ID",
    "Action",
    "LotkaVolterraEnv",
    "Observation",
    "Trajectory",
    "Transition",
    "__version__",
    "collect_corpus",
    "load_corpus",
    "make_env",
    "rollout",
    "sample_transition_batch",
    "split_trajectories",
]
