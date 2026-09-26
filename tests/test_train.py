"""Day 4 one-step trainer. CPU only. No planner and no network."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from worldforge.cli import main
from worldforge.data import load_corpus
from worldforge.models import ModelConfig, WorldModel, load_checkpoint
from worldforge.models.config import CHECKPOINT_FORMAT
from worldforge.train import (
    TRAIN_RUN_FORMAT,
    TrainConfig,
    format_train_report,
    one_step_loss,
    read_metrics,
    read_run,
    replay_batch_seed,
    train_world_model,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "trajectories"


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=4,
        hidden_dim=8,
        encoder_hidden_layers=1,
        dynamics_hidden_layers=1,
        decoder_hidden_layers=1,
    )


def _short_config(**overrides: object) -> TrainConfig:
    settings: dict[str, object] = {
        "epochs": 3,
        "batch_size": 4,
        "lr": 1e-3,
        "seed": 0,
        "device": "cpu",
        "reconstruction_weight": 1.0,
        "steps_per_epoch": 1,
        "optimizer": "adam",
    }
    settings.update(overrides)
    return TrainConfig(**settings)  # type: ignore[arg-type]


def _state(model: WorldModel) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _same_weights(left: WorldModel, right: WorldModel) -> bool:
    return all(torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items())


def test_one_step_loss_matches_mse() -> None:
    model = WorldModel(_tiny_config(), seed=0)
    observation = torch.tensor([[1.0, 1.1], [0.8, 0.9]])
    action = torch.tensor([[0.0, 0.1], [0.2, -0.1]])
    nxt = torch.tensor([[1.05, 1.0], [0.85, 0.95]])
    total, prediction_loss, reconstruction_loss = one_step_loss(
        model, observation, action, nxt, reconstruction_weight=0.25
    )
    forward = model(observation, action)
    expected_prediction = F.mse_loss(forward.predicted_observation, nxt)
    expected_reconstruction = F.mse_loss(forward.reconstructed, observation)
    assert torch.equal(prediction_loss, expected_prediction)
    assert torch.equal(reconstruction_loss, expected_reconstruction)
    assert torch.equal(total, expected_prediction + 0.25 * expected_reconstruction)

    prediction_only, _, _ = one_step_loss(
        model, observation, action, nxt, reconstruction_weight=0.0
    )
    assert torch.equal(prediction_only, expected_prediction)


def test_replay_batch_seeds_stay_inside_an_epoch() -> None:
    assert replay_batch_seed(3, 0, 0) == 3
    assert replay_batch_seed(3, 0, 1) == 4
    assert replay_batch_seed(3, 1, 0) == 3 + 1_000_003
    with pytest.raises(ValueError, match="step"):
        replay_batch_seed(0, 0, 1_000_003)


def test_short_run_on_fixtures_is_finite_and_updates_weights(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    model = WorldModel(_tiny_config(), seed=2)
    before = _state(model)
    result = train_world_model(
        model,
        episodes,
        _short_config(),
        out=tmp_path / "run",
        data_path=FIXTURES,
    )
    assert result.n_episodes == 6
    assert result.n_transitions == 48
    assert result.steps_per_epoch == 1
    assert len(result.epochs) == 3
    assert result.final_loss == result.epochs[-1].loss
    assert all(row.steps == 1 for row in result.epochs)
    assert all(_is_finite(row.loss) for row in result.epochs)
    assert all(_is_finite(row.prediction_loss) for row in result.epochs)
    assert all(_is_finite(row.reconstruction_loss) for row in result.epochs)
    assert any(not torch.equal(before[key], model.state_dict()[key]) for key in before)
    assert next(model.parameters()).device.type == "cpu"


def test_zero_reconstruction_weight_trains_prediction_only(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    result = train_world_model(
        WorldModel(_tiny_config(), seed=1),
        episodes,
        _short_config(epochs=2, reconstruction_weight=0.0, seed=4),
        out=tmp_path / "pred",
    )
    for row in result.epochs:
        assert row.loss == row.prediction_loss
        assert _is_finite(row.reconstruction_loss)


def test_checkpoint_round_trip_keeps_metrics_beside_weights(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    directory = tmp_path / "ckpt"
    directory.mkdir()
    notes = directory / "notes.txt"
    notes.write_text("keep\n", encoding="utf-8")
    model = WorldModel(_tiny_config(), seed=5)
    result = train_world_model(model, episodes, _short_config(seed=5), out=directory)
    assert notes.read_text(encoding="utf-8") == "keep\n"
    payload = (directory / "config.json").read_text(encoding="utf-8")
    assert CHECKPOINT_FORMAT in payload
    lowered = payload.lower() + (directory / "train.json").read_text(encoding="utf-8").lower()
    assert "secret" not in lowered
    assert "token" not in lowered
    assert "api_key" not in lowered

    loaded = load_checkpoint(directory)
    assert loaded.config == model.config
    assert next(loaded.parameters()).device.type == "cpu"
    assert _same_weights(model, loaded)
    observation = torch.tensor([1.2, 0.9])
    action = torch.tensor([0.05, -0.1])
    assert torch.equal(model.predict(observation, action), loaded.predict(observation, action))
    assert bool(torch.isfinite(loaded.predict(observation, action)).all())

    metrics = read_metrics(result.metrics_path)
    assert metrics == result.epochs
    run = read_run(result.run_path)
    assert run["format"] == TRAIN_RUN_FORMAT
    assert run["final_train_loss"] == result.final_loss
    assert run["n_transitions"] == 48
    assert run["optimizer"] == "adam"
    assert run["device"] == "cpu"
    assert "optimizer_state" not in run


def test_same_seed_matches_and_a_different_seed_diverges(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)

    def once(directory: Path, train_seed: int) -> tuple[float, WorldModel]:
        model = WorldModel(_tiny_config(), seed=9)
        result = train_world_model(
            model,
            episodes,
            _short_config(epochs=2, seed=train_seed, lr=1e-2),
            out=directory,
        )
        return result.final_loss, model

    loss_a, model_a = once(tmp_path / "a", 11)
    loss_b, model_b = once(tmp_path / "b", 11)
    loss_c, model_c = once(tmp_path / "c", 12)
    assert loss_a == loss_b
    assert _same_weights(model_a, model_b)
    assert _same_weights(model_a, load_checkpoint(tmp_path / "a"))
    assert loss_a != loss_c
    assert not _same_weights(model_a, model_c)

    metrics_a = (tmp_path / "a" / "metrics.jsonl").read_text(encoding="utf-8")
    metrics_b = (tmp_path / "b" / "metrics.jsonl").read_text(encoding="utf-8")
    assert metrics_a == metrics_b


def test_default_schedule_covers_the_fixture_pool(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    result = train_world_model(
        WorldModel(_tiny_config(), seed=0),
        episodes,
        TrainConfig(epochs=1, batch_size=8, seed=1),
        out=tmp_path / "cover",
    )
    assert result.steps_per_epoch == 6
    assert result.epochs[0].steps == 6
    assert len(read_metrics(result.metrics_path)) == 1
    assert _is_finite(result.final_loss)


def test_sgd_and_default_width_stay_finite(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    sgd = train_world_model(
        WorldModel(_tiny_config(), seed=3),
        episodes,
        _short_config(optimizer="sgd", epochs=2, lr=1e-2, seed=3),
        out=tmp_path / "sgd",
    )
    full = train_world_model(
        WorldModel(ModelConfig(), seed=0),
        episodes,
        TrainConfig(epochs=1, batch_size=8, steps_per_epoch=1, seed=0),
        out=tmp_path / "full",
    )
    assert _is_finite(sgd.final_loss)
    assert _is_finite(full.final_loss)
    assert read_run(sgd.run_path)["optimizer"] == "sgd"
    assert load_checkpoint(tmp_path / "full").config.hidden_dim == 64


def test_train_restores_rng_and_thread_count(tmp_path: Path) -> None:
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    train_world_model(
        WorldModel(_tiny_config(), seed=0),
        load_corpus(FIXTURES),
        _short_config(epochs=1),
        out=tmp_path / "rng",
    )
    assert torch.equal(before, torch.get_rng_state())
    assert torch.get_num_threads() == threads


def test_train_rejects_bad_settings(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    model = WorldModel(_tiny_config(), seed=0)
    with pytest.raises(ValueError, match="cpu"):
        TrainConfig(device="cuda")
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError, match="lr"):
        TrainConfig(lr=0.0)
    with pytest.raises(ValueError, match="reconstruction_weight"):
        TrainConfig(reconstruction_weight=-0.1)
    with pytest.raises(TypeError, match="seed"):
        TrainConfig(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch_size"):
        train_world_model(model, episodes, TrainConfig(batch_size=49), out=tmp_path / "big")
    blocked = tmp_path / "file.pt"
    blocked.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        train_world_model(model, episodes, _short_config(epochs=1), out=blocked)


def test_cli_train_prints_final_loss_and_repeats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def run(directory: Path) -> str:
        assert (
            main(
                [
                    "train",
                    "--data",
                    str(FIXTURES),
                    "--out",
                    str(directory),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "4",
                    "--steps-per-epoch",
                    "1",
                    "--lr",
                    "1e-3",
                    "--seed",
                    "0",
                    "--device",
                    "cpu",
                    "--hidden-dim",
                    "8",
                    "--latent-dim",
                    "4",
                    "--recon-weight",
                    "1",
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        assert captured.err == ""
        return captured.out

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")
    assert "episodes: 6" in first
    assert "transitions: 48" in first
    assert "optimizer: adam" in first
    assert "device: cpu" in first
    assert "steps_per_epoch: 1" in first
    assert "final_train_loss:" in first
    assert first.replace(str(tmp_path / "a"), "OUT") == second.replace(str(tmp_path / "b"), "OUT")
    loaded_a = load_checkpoint(tmp_path / "a")
    loaded_b = load_checkpoint(tmp_path / "b")
    assert _same_weights(loaded_a, loaded_b)
    assert loaded_a.config.latent_dim == 4
    assert loaded_a.config.hidden_dim == 8
    report = format_train_report(
        train_world_model(
            WorldModel(ModelConfig(latent_dim=4, hidden_dim=8), seed=0),
            load_corpus(FIXTURES),
            TrainConfig(epochs=2, batch_size=4, steps_per_epoch=1, seed=0, reconstruction_weight=1.0),
            out=tmp_path / "c",
            data_path=str(FIXTURES),
        )
    )
    assert first == report.replace(str(tmp_path / "c"), str(tmp_path / "a")) + "\n"


def test_cli_train_reads_corpus_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = FIXTURES / "corpus.jsonl"
    out = tmp_path / "jsonl-run"
    assert (
        main(
            [
                "train",
                "--data",
                str(corpus),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--steps-per-epoch",
                "1",
                "--hidden-dim",
                "8",
                "--optimizer",
                "sgd",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert f"data: {corpus}" in text
    assert "optimizer: sgd" in text
    assert "final_train_loss:" in text
    assert (out / "weights.pt").is_file()
    assert (out / "metrics.jsonl").is_file()
    assert read_run(out / "train.json")["n_episodes"] == 6


def test_cli_train_rejects_missing_data_cuda_and_rssm(tmp_path: Path) -> None:
    out = str(tmp_path / "unused")
    with pytest.raises(SystemExit) as missing:
        main(["train", "--data", "missing-corpus", "--out", out])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as cuda:
        main(["train", "--data", str(FIXTURES), "--out", out, "--device", "cuda"])
    assert cuda.value.code == 2
    with pytest.raises(SystemExit) as rssm:
        main(
            [
                "train",
                "--data",
                str(FIXTURES),
                "--out",
                out,
                "--dynamics",
                "rssm",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "1",
            ]
        )
    assert rssm.value.code == 2
    with pytest.raises(SystemExit) as huge:
        main(
            [
                "train",
                "--data",
                str(FIXTURES),
                "--out",
                out,
                "--batch-size",
                "1000",
                "--epochs",
                "1",
            ]
        )
    assert huge.value.code == 2


def test_console_script_train(tmp_path: Path) -> None:
    script = Path(sys.executable).with_name("worldforge")
    out = tmp_path / "script"
    completed = subprocess.run(
        [
            str(script),
            "train",
            "--data",
            str(FIXTURES),
            "--out",
            str(out),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--steps-per-epoch",
            "1",
            "--hidden-dim",
            "8",
            "--seed",
            "1",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "final_train_loss:" in completed.stdout
    assert completed.stderr == ""
    assert (out / "config.json").is_file()
    assert (out / "weights.pt").is_file()


def _is_finite(value: float) -> bool:
    return math.isfinite(value)
