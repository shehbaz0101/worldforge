"""Trajectory schema, JSON round trip, and seeded rollouts. No network."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from worldforge.envs import make_env
from worldforge.rollout import rollout
from worldforge.schemas import Observation, Trajectory


def test_same_seed_same_trajectory() -> None:
    first = rollout(make_env(horizon=12), n_steps=12, seed=7, policy="random")
    second = rollout(make_env(horizon=12), n_steps=12, seed=7, policy="random")
    assert first == second
    assert first.env_id == "lotka_volterra"
    assert first.seed == 7
    assert first.horizon == 12
    assert len(first.transitions) == 12
    assert first.transitions[-1].truncated is True
    assert first.transitions[-1].terminated is False
    assert all(not item.truncated for item in first.transitions[:-1])
    assert len(first.transitions[0].observation.values) == 2
    assert len(first.transitions[0].action.values) == 2


def test_different_seeds_differ() -> None:
    first = rollout(make_env(horizon=8), n_steps=8, seed=0, policy="zero")
    second = rollout(make_env(horizon=8), n_steps=8, seed=1, policy="zero")
    assert first.transitions[0].observation.values != second.transitions[0].observation.values


def test_random_policy_is_deterministic_and_moves_the_state() -> None:
    zero = rollout(make_env(horizon=8), n_steps=8, seed=4, policy="zero")
    random_episode = rollout(make_env(horizon=8), n_steps=8, seed=4, policy="random")
    again = rollout(make_env(horizon=8), n_steps=8, seed=4, policy="random")
    assert random_episode == again
    assert random_episode.transitions[0].observation == zero.transitions[0].observation
    assert random_episode.transitions[0].action.values != [0.0, 0.0]
    assert (
        random_episode.transitions[0].next_observation
        != zero.transitions[0].next_observation
    )
    assert all(item.action.values == [0.0, 0.0] for item in zero.transitions)


def test_rollout_stops_when_the_env_truncates() -> None:
    episode = rollout(make_env(horizon=3), n_steps=10, seed=0, policy="zero")
    assert episode.horizon == 10
    assert len(episode.transitions) == 3
    assert episode.transitions[-1].truncated is True


def test_json_round_trip() -> None:
    episode = rollout(make_env(horizon=6), n_steps=6, seed=9, policy="random")
    restored = Trajectory.from_json(episode.to_json())
    assert restored == episode


def test_jsonl_round_trip(tmp_path: Path) -> None:
    episode = rollout(make_env(horizon=5), n_steps=5, seed=2, policy="zero")
    restored = Trajectory.from_jsonl(episode.to_jsonl())
    assert restored == episode

    path = tmp_path / "episode.jsonl"
    episode.write_jsonl(path)
    assert Trajectory.read_jsonl(path) == episode

    json_path = tmp_path / "episode.json"
    episode.write_json(json_path)
    assert Trajectory.read_json(json_path) == episode
    assert json_path.read_text(encoding="utf-8").endswith("\n")


def test_json_rejects_unknown_fields() -> None:
    episode = rollout(make_env(horizon=1), n_steps=1, seed=0)
    payload = json.loads(episode.to_json())
    payload["note"] = "extra"
    with pytest.raises(ValidationError):
        Trajectory.from_json(json.dumps(payload))


def test_observation_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        Observation(values=[float("nan")], t=0.0)
    with pytest.raises(ValidationError):
        Observation(values=[1.0], t=-1.0)


def test_jsonl_requires_a_metadata_header() -> None:
    with pytest.raises(ValueError, match="metadata"):
        Trajectory.from_jsonl('{"record":"transition"}\n')
    with pytest.raises(ValueError, match="empty"):
        Trajectory.from_jsonl("\n")


def test_sine_dose_is_deterministic_open_loop() -> None:
    env = make_env(horizon=16)
    first = rollout(env, n_steps=16, seed=4, policy="sine_dose")
    second = rollout(make_env(horizon=16), n_steps=16, seed=4, policy="sine_dose")
    zero = rollout(make_env(horizon=16), n_steps=16, seed=4, policy="zero")
    assert first == second
    assert first.transitions[0].observation == zero.transitions[0].observation
    prey = [step.action.values[0] for step in first.transitions]
    predator = [step.action.values[1] for step in first.transitions]
    assert prey[0] == pytest.approx(0.0)
    assert predator[0] == pytest.approx(0.25)
    assert prey != predator
    assert any(step.action.values != [0.0, 0.0] for step in first.transitions)
    assert first.transitions[0].next_observation != zero.transitions[0].next_observation
    for transition in first.transitions:
        assert all(-0.25 <= value <= 0.25 for value in transition.action.values)
    other_seed = rollout(make_env(horizon=16), n_steps=16, seed=5, policy="sine_dose")
    assert [step.action.values for step in other_seed.transitions] == [
        step.action.values for step in first.transitions
    ]
    assert other_seed.transitions[0].observation != first.transitions[0].observation


def test_random_doses_stay_inside_a_short_episode() -> None:
    episode = rollout(make_env(horizon=32), n_steps=32, seed=11, policy="random")
    assert len(episode.transitions) == 32
    assert episode.transitions[-1].terminated is False
    for transition in episode.transitions:
        assert all(math.isfinite(value) for value in transition.next_observation.values)
        assert all(-0.25 <= value <= 0.25 for value in transition.action.values)
