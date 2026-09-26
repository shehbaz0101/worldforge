"""Day 9 offline demo. The full loop needs the ml extra; corpus checks do not."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from worldforge.cli import main
from worldforge.data import EpisodeSpec, collect_corpus, load_corpus
from worldforge.demo import (
    MAX_DEMO_EPOCHS,
    DemoConfig,
    DemoResult,
    bundled_samples_dir,
    load_demo_config,
    metrics_are_finite,
    prepare_demo_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "trajectories"


def test_demo_config_matches_checked_in_file() -> None:
    config = load_demo_config()
    assert config.format == "worldforge.demo.v1"
    assert config.corpus == "trajectories"
    assert config.horizon == 4
    assert [(episode.seed, episode.policy) for episode in config.episodes] == [
        (0, "zero"),
        (1, "random"),
    ]
    assert config.epochs == 1
    assert config.steps_per_epoch == 1
    assert config.latent_dim == 4
    assert config.hidden_dim == 16
    assert config.plan_horizon == 2
    assert config.n_samples == 4
    assert config.n_iterations == 1
    assert config.n_steps == 2
    assert config.device == "cpu"
    assert bundled_samples_dir() == ROOT / "samples"


def test_demo_config_rejects_a_long_run() -> None:
    with pytest.raises(ValidationError):
        DemoConfig(epochs=MAX_DEMO_EPOCHS + 1)
    with pytest.raises(ValidationError):
        DemoConfig(device="cuda")
    with pytest.raises(ValidationError):
        load_demo_config().with_overrides(epochs=MAX_DEMO_EPOCHS + 1)


def test_sample_corpus_matches_collection(tmp_path: Path) -> None:
    config = load_demo_config()
    fresh = tmp_path / "fresh"
    collect_corpus(
        fresh,
        horizon=config.horizon,
        specs=tuple(
            EpisodeSpec(seed=episode.seed, policy=episode.policy) for episode in config.episodes
        ),
        formats=("jsonl", "json"),
    )
    assert load_corpus(fresh) == load_corpus(SAMPLES)
    assert (SAMPLES / "manifest.json").is_file()
    assert (SAMPLES / "corpus.jsonl").is_file()
    assert (SAMPLES / "episodes" / "ep_0001.json").is_file()


def test_prepare_uses_samples_inside_the_data_root(tmp_path: Path) -> None:
    config = load_demo_config()
    out = tmp_path / "run"
    corpus, source = prepare_demo_corpus(root=ROOT, out=out, config=config)
    assert source == "samples"
    assert corpus == SAMPLES
    assert not out.exists()


def test_prepare_copies_samples_outside_the_data_root(tmp_path: Path) -> None:
    config = load_demo_config()
    out = tmp_path / "run"
    corpus, source = prepare_demo_corpus(root=tmp_path, out=out, config=config)
    assert source == "copied"
    assert corpus == out / "corpus"
    assert load_corpus(corpus) == load_corpus(SAMPLES)


def test_prepare_builds_when_samples_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_demo_config()
    monkeypatch.setattr("worldforge.demo.bundled_samples_dir", lambda: None)
    out = tmp_path / "run"
    corpus, source = prepare_demo_corpus(root=tmp_path, out=out, config=config)
    assert source == "built"
    assert corpus == out / "corpus"
    assert load_corpus(corpus) == load_corpus(SAMPLES)


def test_prepare_user_corpus_is_not_copied(tmp_path: Path) -> None:
    config = load_demo_config()
    out = tmp_path / "run"
    corpus, source = prepare_demo_corpus(root=ROOT, out=out, config=config, data=SAMPLES)
    assert source == "user"
    assert corpus == SAMPLES


def test_prepare_refuses_to_nest_the_copy_inside_the_samples(tmp_path: Path) -> None:
    config = load_demo_config()
    with pytest.raises(ValueError, match="overlap"):
        prepare_demo_corpus(root=tmp_path, out=SAMPLES, config=config)
    assert not (SAMPLES / "corpus").exists()


def test_metrics_are_finite_rejects_nan(tmp_path: Path) -> None:
    result = DemoResult(
        corpus=tmp_path,
        corpus_source="samples",
        checkpoint=tmp_path,
        predict_path=tmp_path,
        plan_path=tmp_path,
        eval_plan_path=tmp_path,
        summary_path=tmp_path,
        train=SimpleNamespace(final_train_loss=float("nan")),
        predict=SimpleNamespace(mse_by_h=(0.1,)),
        plan=SimpleNamespace(predicted_return=0.0),
        eval_plan=SimpleNamespace(
            planner_return=0.0,
            zero_return=0.0,
            random_return=0.0,
            baseline_best=0.0,
            regret=0.0,
        ),
    )
    assert metrics_are_finite(result) is False
    result.train.final_train_loss = 0.2
    assert metrics_are_finite(result) is True


def test_demo_rejects_an_output_outside_the_data_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "demo",
                "--data-root",
                str(tmp_path),
                "--out",
                str(tmp_path.parent / "demo-escape"),
            ]
        )
    assert caught.value.code == 2


def test_demo_rejects_too_many_epochs() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["demo", "--epochs", str(MAX_DEMO_EPOCHS + 1), "--out", "unused"])
    assert caught.value.code == 2


def test_demo_rejects_a_non_cpu_device() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["demo", "--device", "cuda", "--out", "unused"])
    assert caught.value.code == 2


def test_demo_reports_a_missing_ml_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    demo_mod = importlib.import_module("worldforge.demo")

    def missing(*_args: object, **_kwargs: object) -> object:
        raise demo_mod.DemoDependencyError(
            demo_mod.ML_INSTALL_HINT + " Import failed: No module named 'torch'"
        )

    monkeypatch.setattr(demo_mod, "run_demo", missing)
    with pytest.raises(SystemExit) as caught:
        main(["demo", "--out", "unused-demo"])
    assert caught.value.code == 2
    assert 'pip install -e ".[ml]"' in capsys.readouterr().err


def test_demo_command_trains_and_prints_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "run"
    assert main(["demo", "--data-root", str(tmp_path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "worldforge demo" in text
    assert "corpus_source: copied" in text
    assert "episodes: 2" in text
    assert "transitions: 8" in text
    assert "final_train_loss:" in text
    assert "predict_mse_by_h:" in text
    assert "predict_horizons: 1 2 3 4" in text
    assert "plan_predicted_return:" in text
    assert "regret:" in text
    assert (out / "checkpoint" / "weights.pt").is_file()
    assert (out / "checkpoint" / "config.json").is_file()
    assert (out / "predict.json").is_file()
    assert (out / "plan.json").is_file()
    assert (out / "eval-plan.json").is_file()
    assert (out / "summary.txt").read_text(encoding="utf-8") == text
    assert (out / "corpus" / "corpus.jsonl").is_file()


def test_demo_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    demo_mod = importlib.import_module("worldforge.demo")
    monkeypatch.setenv("WORLDFORGE_DATA_ROOT", str(tmp_path))
    first = demo_mod.run_demo(tmp_path / "one")
    second = demo_mod.run_demo(tmp_path / "two")
    assert first.corpus_source == second.corpus_source == "copied"
    assert first.train.final_train_loss == second.train.final_train_loss
    assert first.predict.mse_by_h == second.predict.mse_by_h
    assert first.plan.actions == second.plan.actions
    assert first.eval_plan.regret == second.eval_plan.regret
    left = torch.load(first.checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    right = torch.load(second.checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def test_demo_uses_samples_when_they_are_inside_the_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "inside"
    assert main(["demo", "--data-root", str(ROOT), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "corpus_source: samples" in text
    assert f"corpus: {SAMPLES}" in text
    assert not (out / "corpus").exists()
    assert (out / "eval-plan.json").is_file()
