"""Scientific dynamics environments.

``lotka_volterra`` is the only registered environment, and it is the default.
See :class:`worldforge.envs.lotka_volterra.LotkaVolterraEnv` for why this
kinetic ODE is the sandbox instead of a reaction-diffusion grid.
"""

from __future__ import annotations

from worldforge.envs.lotka_volterra import ENV_ID, LotkaVolterraEnv

DEFAULT_ENV_ID = ENV_ID

__all__ = ["DEFAULT_ENV_ID", "LotkaVolterraEnv", "make_env"]


def make_env(env_id: str = DEFAULT_ENV_ID, *, horizon: int = 32) -> LotkaVolterraEnv:
    """Build a registered environment.

    ``horizon`` is the step limit. The rollout helper should use the same
    length when the caller wants one transition per requested step.
    """

    if env_id != DEFAULT_ENV_ID:
        raise ValueError(
            f"unknown env_id {env_id!r}; registered environments: {DEFAULT_ENV_ID!r}"
        )
    return LotkaVolterraEnv(horizon=horizon)
