"""Corpus collect, load, split, and replay batch. No network."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldforge.data import (
    EpisodeSpec,
    collect_corpus,
    inspect_dataset,
    load_corpus,
    load_manifest,
    sample_transition_batch,
    split_trajectories,
)
from worldforge.envs import make_env
from worldforge.rollout import rollout
from worldforge.schemas import Trajectory


def test_collect_round_trip_matches_rollout(tmp_path: Path) -> None:
    out = tmp_path / "corpus"
    manifest = collect_corpus(
        out,
        n_episodes=3,
        horizon=5,
        seed=10,
        policy="random",
    )
    loaded = load_corpus(out)
    assert manifest.n_episodes == 3
    assert manifest.policy == "random"
    assert manifest.env_id == "lotka_volterra"
    assert len(loaded) == 3
    for index, episode in enumerate(loaded):
        expected = rollout(make_env(horizon=5), n_steps=5, seed=10 + index, policy="random")
        assert episode == expected
        assert Trajectory.read_json(out / f"episodes/ep_{index:04d}.json") == expected
    from_jsonl = load_corpus(out / "corpus.jsonl")
    assert from_jsonl == loaded
    again = collect_corpus(tmp_path / "copy", n_episodes=3, horizon=5, seed=10, policy="random")
    assert again == manifest
    assert (out / "corpus.jsonl").read_bytes() == (tmp_path / "copy" / "corpus.jsonl").read_bytes()


def test_explicit_seeds_and_json_only(tmp_path: Path) -> None:
    out = tmp_path / "json-only"
    collect_corpus(out, seeds=(2, 4), horizon=4, policy="zero", formats=("json",))
    assert not (out / "corpus.jsonl").exists()
    loaded = load_corpus(out)
    assert [episode.seed for episode in loaded] == [2, 4]
    assert all(
        step.action.values == [0.0, 0.0] for episode in loaded for step in episode.transitions
    )
    collect_corpus(out, seeds=(2,), horizon=4, policy="zero", formats=("json",))
    names = sorted(path.name for path in (out / "episodes").glob("ep_*.json"))
    assert names == ["ep_0000.json"]
    assert load_corpus(out)[0].seed == 2


def test_switching_format_drops_the_stale_corpus_file(tmp_path: Path) -> None:
    out = tmp_path / "switch"
    collect_corpus(out, n_episodes=1, horizon=2, seed=0, policy="zero")
    collect_corpus(out, n_episodes=1, horizon=2, seed=1, policy="zero", formats=("json",))
    assert not (out / "corpus.jsonl").exists()
    assert load_corpus(out)[0].seed == 1
    collect_corpus(out, n_episodes=1, horizon=2, seed=3, policy="random", formats=("jsonl",))
    assert load_corpus(out)[0].seed == 3
    assert list((out / "episodes").glob("ep_*.json")) == []


def test_mixed_specs(tmp_path: Path) -> None:
    out = tmp_path / "mixed"
    specs = (
        EpisodeSpec(seed=0, policy="zero"),
        EpisodeSpec(seed=1, policy="sine_dose"),
    )
    manifest = collect_corpus(out, horizon=4, specs=specs, formats=("jsonl",))
    assert manifest.policy == "mixed"
    loaded = load_corpus(out)
    assert loaded[0] == rollout(make_env(horizon=4), n_steps=4, seed=0, policy="zero")
    assert loaded[1] == rollout(make_env(horizon=4), n_steps=4, seed=1, policy="sine_dose")
    info = inspect_dataset(out)
    assert info.kind == "directory"
    assert info.policies == ("zero", "sine_dose")
    assert info.n_episodes == 2
    assert info.n_transitions == 8


def test_single_episode_jsonl_still_loads(tmp_path: Path) -> None:
    episode = rollout(make_env(horizon=3), n_steps=3, seed=1, policy="zero")
    path = tmp_path / "one.jsonl"
    episode.write_jsonl(path)
    assert load_corpus(path) == [episode]
    assert inspect_dataset(path).kind == "trajectory-jsonl"
    assert inspect_dataset(path).policies == ()


def test_split_is_deterministic_and_order_independent() -> None:
    episodes = [
        rollout(make_env(horizon=4), n_steps=4, seed=seed, policy="zero") for seed in range(8)
    ]
    first = split_trajectories(episodes, seed=0)
    second = split_trajectories(list(reversed(episodes)), seed=0)
    assert first == second
    assert first.counts == (6, 1, 1)
    assert first.seed == 0
    other = split_trajectories(episodes, seed=1)
    assert other.counts == first.counts
    assert other.train != first.train
    combined = list(first.train + first.val + first.test)
    assert sorted(combined, key=lambda item: item.seed) == episodes


def test_split_rejects_bad_ratios() -> None:
    episodes = [rollout(make_env(horizon=2), n_steps=2, seed=0, policy="zero")]
    with pytest.raises(ValueError, match="sum to 1"):
        split_trajectories(episodes, seed=0, ratios=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="non-empty"):
        split_trajectories([], seed=0)


def test_sample_transition_batch_is_deterministic() -> None:
    episodes = [
        rollout(make_env(horizon=6), n_steps=6, seed=seed, policy="random") for seed in range(4)
    ]
    first = sample_transition_batch(episodes, batch_size=5, seed=0)
    second = sample_transition_batch(episodes, batch_size=5, seed=0)
    other = sample_transition_batch(tuple(reversed(episodes)), batch_size=5, seed=1)
    assert first == second
    assert len(first) == 5
    assert first.seeds != other.seeds or first.actions != other.actions
    pool = {
        (episode.seed, tuple(step.action.values), step.reward)
        for episode in episodes
        for step in episode.transitions
    }
    chosen = {
        (seed, action, reward)
        for seed, action, reward in zip(first.seeds, first.actions, first.rewards, strict=True)
    }
    assert chosen <= pool
    with pytest.raises(ValueError, match="exceeds"):
        sample_transition_batch(episodes, batch_size=100, seed=0)


def test_collect_rejects_disagreement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        collect_corpus(tmp_path / "bad", n_episodes=2, seeds=(0, 1, 2))
    with pytest.raises(ValueError, match=">= 1"):
        collect_corpus(tmp_path / "zero", n_episodes=0)


def test_manifest_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "corpus"
    written = collect_corpus(out, n_episodes=2, horizon=3, seed=0, policy="sine_dose")
    assert load_manifest(out) == written
    bare = tmp_path / "bare.jsonl"
    bare.write_bytes((out / "corpus.jsonl").read_bytes())
    assert inspect_dataset(bare).policies == ()
    assert inspect_dataset(bare).kind == "corpus-jsonl"
