"""Checked-in trajectory fixtures stay aligned with the collector. No network."""

from __future__ import annotations

from pathlib import Path

from worldforge.data import inspect_dataset, load_corpus, load_manifest
from worldforge.envs import make_env
from worldforge.rollout import rollout

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "trajectories"


def test_fixture_files_exist() -> None:
    assert (FIXTURES / "manifest.json").is_file()
    assert (FIXTURES / "corpus.jsonl").is_file()
    assert (FIXTURES / "README.md").is_file()
    episodes = list((FIXTURES / "episodes").glob("ep_*.json"))
    assert 4 <= len(episodes) <= 8


def test_fixtures_match_a_fresh_rollout() -> None:
    manifest = load_manifest(FIXTURES)
    loaded = load_corpus(FIXTURES)
    from_jsonl = load_corpus(FIXTURES / "corpus.jsonl")
    assert manifest.format == "worldforge.corpus.v1"
    assert manifest.env_id == "lotka_volterra"
    assert manifest.n_episodes == len(loaded) == len(from_jsonl)
    assert 4 <= manifest.n_episodes <= 8
    assert loaded == from_jsonl
    policies = {record.policy for record in manifest.episodes}
    assert policies == {"zero", "random", "sine_dose"}
    for record, episode in zip(manifest.episodes, loaded, strict=True):
        expected = rollout(
            make_env(horizon=manifest.horizon),
            n_steps=manifest.horizon,
            seed=record.seed,
            policy=record.policy,
        )
        assert episode == expected
        assert record.steps == len(episode.transitions)
        assert record.file is not None
        assert load_corpus(FIXTURES / record.file) == [episode]
    info = inspect_dataset(FIXTURES)
    assert info.n_episodes == manifest.n_episodes
    assert info.policies == ("zero", "random", "sine_dose")
    assert info.env_ids == ("lotka_volterra",)
