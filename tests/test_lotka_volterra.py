"""Unit tests for the default kinetic environment. No network."""

from __future__ import annotations

import math

import pytest

from worldforge.envs import DEFAULT_ENV_ID, make_env
from worldforge.envs.lotka_volterra import (
    EQUILIBRIUM,
    STEP_DT,
    LotkaVolterraEnv,
    lotka_volterra_field,
)
from worldforge.envs.protocol import DynamicsEnv


def test_env_matches_protocol() -> None:
    env = make_env()
    assert isinstance(env, DynamicsEnv)
    assert env.env_id == DEFAULT_ENV_ID == "lotka_volterra"


def test_vector_field_is_zero_at_equilibrium() -> None:
    assert lotka_volterra_field(1.0, 1.0) == (0.0, 0.0)
    assert lotka_volterra_field(1.0, 1.0, prey_dose=0.25) == (0.25, 0.0)


def test_reset_is_deterministic() -> None:
    first = LotkaVolterraEnv().reset(seed=5)
    second = LotkaVolterraEnv().reset(seed=5)
    assert first == second
    assert all(math.isfinite(value) and value > 0.0 for value in first)


def test_step_shapes() -> None:
    env = make_env(horizon=4)
    obs = env.reset(seed=0)
    assert len(obs) == env.observation_dim == 2
    nxt, reward, terminated, truncated, info = env.step((0.0, 0.0))
    assert len(nxt) == 2
    assert all(math.isfinite(value) for value in nxt)
    assert isinstance(reward, float) and math.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1
    assert info["t"] == pytest.approx(STEP_DT)
    assert info["action_clipped"] is False


def test_equilibrium_is_fixed_under_zero_dose() -> None:
    env = LotkaVolterraEnv(horizon=4)
    assert env.reset(seed=1, state=EQUILIBRIUM) == EQUILIBRIUM
    for index in range(4):
        obs, reward, terminated, truncated, _info = env.step((0.0, 0.0))
        assert obs == EQUILIBRIUM
        assert reward == 0.0
        assert terminated is False
        assert truncated is (index == 3)


def test_positive_prey_dose_increases_prey_at_equilibrium() -> None:
    env = LotkaVolterraEnv(horizon=2)
    env.reset(seed=0, state=EQUILIBRIUM)
    stocked, _reward, _terminated, _truncated, info = env.step((0.25, 0.0))
    assert stocked[0] > EQUILIBRIUM[0]
    assert info["prey_dose"] == pytest.approx(0.25)

    harvested = LotkaVolterraEnv(horizon=2)
    harvested.reset(seed=0, state=EQUILIBRIUM)
    obs, _reward, _terminated, _truncated, _info = harvested.step((-0.25, 0.0))
    assert obs[0] < EQUILIBRIUM[0]


def test_action_outside_bounds_is_clipped() -> None:
    env = LotkaVolterraEnv(horizon=1)
    env.reset(seed=0, state=EQUILIBRIUM)
    _obs, _reward, _terminated, _truncated, info = env.step((1.0, -1.0))
    assert info["action_clipped"] is True
    assert info["prey_dose"] == pytest.approx(0.25)
    assert info["predator_dose"] == pytest.approx(-0.25)


def test_step_before_reset_raises() -> None:
    env = LotkaVolterraEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step((0.0, 0.0))
    with pytest.raises(RuntimeError, match="reset"):
        env.render_ascii()


def test_bad_action_and_horizon_raise() -> None:
    env = LotkaVolterraEnv(horizon=2)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="length"):
        env.step((0.0,))
    with pytest.raises(ValueError, match="finite"):
        env.step((float("nan"), 0.0))
    with pytest.raises(TypeError):
        env.step((True, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon"):
        LotkaVolterraEnv(horizon=0)


def test_render_ascii_names_both_populations() -> None:
    env = LotkaVolterraEnv(horizon=2)
    env.reset(seed=0, state=(1.0, 1.0))
    text = env.render_ascii()
    assert "prey" in text
    assert "predator" in text
    assert "1.0000" in text
    assert "#" in text


def test_episode_truncates_at_horizon() -> None:
    env = LotkaVolterraEnv(horizon=3)
    env.reset(seed=2)
    flags = []
    for _ in range(3):
        _obs, _reward, terminated, truncated, _info = env.step((0.0, 0.0))
        flags.append((terminated, truncated))
    assert flags == [(False, False), (False, False), (False, True)]
    with pytest.raises(RuntimeError, match="finished"):
        env.step((0.0, 0.0))


def test_short_uncontrolled_episodes_stay_positive() -> None:
    for seed in range(10):
        env = LotkaVolterraEnv(horizon=32)
        env.reset(seed=seed)
        for _ in range(32):
            obs, reward, terminated, _truncated, _info = env.step((0.0, 0.0))
            assert all(math.isfinite(value) and value > 0.0 for value in obs)
            assert math.isfinite(reward)
            assert terminated is False


def test_make_env_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown env_id"):
        make_env("gray_scott")
