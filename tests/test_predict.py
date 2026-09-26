"""Day 5 open-loop rollout and prediction metrics. CPU only. No planner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from worldforge.cli import main
from worldforge.data import load_corpus
from worldforge.eval import (
    PREDICT_REPORT_FORMAT,
    format_predict_report,
    open_loop_metrics,
    read_predict_report,
    rollout_latent,
    rollout_trajectory,
    trajectory_window,
    write_predict_report,
)
from worldforge.models import ModelConfig, WorldModel, load_checkpoint, save_checkpoint
from worldforge.schemas import Action, Observation, Trajectory, Transition

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


def _transition(observation: list[float], action: list[float], nxt: list[float], t: float) -> Transition:
    return Transition(
        observation=Observation(values=observation, t=t),
        action=Action(values=action),
        reward=0.0,
        next_observation=Observation(values=nxt, t=t + 0.2),
        terminated=False,
        truncated=False,
    )


def test_rollout_shapes_and_horizon_one_matches_predict() -> None:
    model = WorldModel(_tiny_config(), seed=0)
    observation = torch.tensor([1.0, 0.8])
    actions = torch.tensor([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.05]])
    rolled = rollout_latent(model, observation, actions)
    assert rolled.predicted_observations.shape == (3, 2)
    assert rolled.latents.shape == (3, 4)
    assert rolled.initial_latent.shape == (4,)
    assert rolled.predicted_observations.dtype == torch.float32
    assert torch.equal(rolled.predicted_observations[0], model.predict(observation, actions[0]))

    batched_obs = observation.unsqueeze(0).repeat(2, 1)
    batched_actions = actions.unsqueeze(0).repeat(2, 1, 1)
    batched_actions[1] = batched_actions[1] + 0.01
    batched = rollout_latent(model, batched_obs, batched_actions)
    assert batched.predicted_observations.shape == (2, 3, 2)
    assert batched.latents.shape == (2, 3, 4)
    assert batched.initial_latent.shape == (2, 4)
    assert torch.equal(
        batched.predicted_observations[:, 0],
        model.predict(batched_obs, batched_actions[:, 0]),
    )


def test_rollout_encodes_only_the_initial_observation() -> None:
    model = WorldModel(_tiny_config(), seed=1)
    observation = torch.tensor([1.2, 0.7])
    actions = torch.tensor([[0.2, 0.0], [-0.1, 0.1], [0.0, 0.0]])
    calls = {"n": 0}
    original = model.encode

    def _count(value: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return original(value)

    model.encode = _count  # type: ignore[method-assign]
    rolled = rollout_latent(model, observation, actions)
    assert calls["n"] == 1

    latent = original(observation)
    for index in range(actions.shape[0]):
        latent = model.step(latent, actions[index])
        assert torch.equal(rolled.latents[index], latent)
        assert torch.equal(rolled.predicted_observations[index], model.decode(latent))


def test_rollout_backward_reaches_parameters() -> None:
    model = WorldModel(_tiny_config(), seed=2)
    observation = torch.tensor([1.0, 1.0])
    actions = torch.tensor([[0.1, 0.0], [0.0, -0.1]])
    rolled = rollout_latent(model, observation, actions)
    rolled.predicted_observations.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_open_loop_metrics_on_fixtures_match_manual_mse() -> None:
    episodes = load_corpus(FIXTURES)
    model = WorldModel(_tiny_config(), seed=3)
    report = open_loop_metrics(model, episodes, horizon=4)
    assert report.n_episodes == 6
    assert report.horizons == (1, 2, 3, 4)
    assert len(report.mse_by_h) == 4
    assert len(report.mae_by_h) == 4
    assert len(report.residual_std_by_h) == 4
    assert report.n_episodes_by_h == (6, 6, 6, 6)
    assert all(value >= 0.0 for value in report.mse_by_h)
    assert all(value >= 0.0 for value in report.residual_std_by_h)
    assert report.checkpoint is None
    assert report.data is None

    squared: list[torch.Tensor] = []
    absolute: list[torch.Tensor] = []
    for episode in episodes:
        window = trajectory_window(model, episode, horizon=4)
        predicted = rollout_latent(model, window.observation, window.actions).predicted_observations
        assert torch.equal(predicted[0], model.predict(window.observation, window.actions[0]))
        error = (predicted - window.targets).detach()
        squared.append(error.square())
        absolute.append(error.abs())
    mse = torch.stack(squared).mean(dim=(0, 2))
    mae = torch.stack(absolute).mean(dim=(0, 2))
    for index, horizon in enumerate(report.horizons):
        assert horizon == index + 1
        assert report.mse_by_h[index] == pytest.approx(float(mse[index]))
        assert report.mae_by_h[index] == pytest.approx(float(mae[index]))


def test_full_fixture_horizon_is_eight_and_restores_train_mode() -> None:
    episodes = load_corpus(FIXTURES)
    model = WorldModel(_tiny_config(), seed=4)
    model.train()
    report = open_loop_metrics(model, episodes)
    assert model.training is True
    assert report.horizons == tuple(range(1, 9))
    assert report.n_episodes == 6
    assert report.n_episodes_by_h == (6,) * 8
    again = open_loop_metrics(model, episodes)
    assert again.mse_by_h == report.mse_by_h


def test_shorter_episodes_drop_out_of_later_horizons() -> None:
    episodes = load_corpus(FIXTURES)
    short = Trajectory(
        env_id=episodes[0].env_id,
        seed=episodes[0].seed,
        horizon=2,
        transitions=list(episodes[0].transitions[:2]),
    )
    long = episodes[1]
    model = WorldModel(_tiny_config(), seed=5)
    report = open_loop_metrics(model, [short, long], horizon=4)
    assert report.n_episodes == 2
    assert report.horizons == (1, 2, 3, 4)
    assert report.n_episodes_by_h == (2, 2, 1, 1)


def test_report_json_round_trip(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    model = WorldModel(_tiny_config(), seed=6)
    report = open_loop_metrics(
        model,
        episodes,
        horizon=2,
        checkpoint="checkpoints/day5",
        data=str(FIXTURES),
    )
    path = tmp_path / "nested" / "predict.json"
    write_predict_report(path, report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "format",
        "horizons",
        "mse_by_h",
        "mae_by_h",
        "residual_std_by_h",
        "n_episodes",
        "n_episodes_by_h",
        "checkpoint",
        "data",
    }
    assert payload["format"] == PREDICT_REPORT_FORMAT
    assert payload["horizons"] == [1, 2]
    assert payload["n_episodes"] == 6
    assert len(payload["mse_by_h"]) == 2
    assert "secret" not in path.read_text(encoding="utf-8").lower()
    restored = read_predict_report(path)
    assert restored.horizons == report.horizons
    assert restored.n_episodes == report.n_episodes
    assert restored.n_episodes_by_h == report.n_episodes_by_h
    assert restored.checkpoint == report.checkpoint
    assert restored.data == report.data
    assert restored.mse_by_h == pytest.approx(report.mse_by_h)
    assert restored.mae_by_h == pytest.approx(report.mae_by_h)
    assert restored.residual_std_by_h == pytest.approx(report.residual_std_by_h)
    text = format_predict_report(restored)
    assert "n_episodes: 6" in text
    assert "horizons: 1 2" in text
    assert "checkpoints/day5" in text


def test_checkpoint_rollout_matches_loaded_weights(tmp_path: Path) -> None:
    episodes = load_corpus(FIXTURES)
    model = WorldModel(_tiny_config(), seed=7)
    directory = tmp_path / "ckpt"
    save_checkpoint(directory, model)
    loaded = load_checkpoint(directory)
    direct = rollout_trajectory(model, episodes[0], horizon=3)
    restored = rollout_trajectory(loaded, episodes[0], horizon=3)
    assert torch.equal(direct.predicted_observations, restored.predicted_observations)
    assert torch.equal(
        direct.predicted_observations[0],
        loaded.predict(
            torch.tensor(episodes[0].transitions[0].observation.values),
            torch.tensor(episodes[0].transitions[0].action.values),
        ),
    )


def test_rejects_bad_rollout_inputs() -> None:
    model = WorldModel(_tiny_config(), seed=0)
    observation = torch.zeros(2)
    with pytest.raises(ValueError, match="horizon"):
        rollout_latent(model, observation, torch.zeros(0, 2))
    with pytest.raises(ValueError, match="actions"):
        rollout_latent(model, observation, torch.zeros(3))
    with pytest.raises(ValueError, match="leading"):
        rollout_latent(model, torch.zeros(2, 2), torch.zeros(3, 2))
    episodes = load_corpus(FIXTURES)
    with pytest.raises(ValueError, match="exceeds"):
        rollout_trajectory(model, episodes[0], horizon=99)
    with pytest.raises(ValueError, match="non-empty"):
        open_loop_metrics(model, [])
    with pytest.raises(ValueError, match="exceeds every"):
        open_loop_metrics(model, episodes, horizon=99)
    broken = Trajectory(
        env_id="lotka_volterra",
        seed=0,
        horizon=2,
        transitions=[
            _transition([1.0, 1.0], [0.0, 0.0], [1.1, 0.9], 0.0),
            _transition([9.0, 9.0], [0.0, 0.0], [1.2, 0.8], 0.2),
        ],
    )
    with pytest.raises(ValueError, match="chain"):
        trajectory_window(model, broken)


def test_eval_predict_and_rollout_cli_write_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "predict.json"
    assert (
        main(
            [
                "eval-predict",
                "--data",
                str(FIXTURES),
                "--out",
                str(out),
                "--horizon",
                "3",
                "--seed",
                "0",
                "--latent-dim",
                "4",
                "--hidden-dim",
                "8",
            ]
        )
        == 0
    )
    printed = capsys.readouterr()
    assert printed.err == ""
    assert f"format: {PREDICT_REPORT_FORMAT}" in printed.out
    assert "n_episodes: 6" in printed.out
    assert "horizons: 1 2 3" in printed.out
    assert "mse_by_h:" in printed.out
    assert f"report: {out}" in printed.out
    report = read_predict_report(out)
    assert report.horizons == (1, 2, 3)
    assert report.n_episodes == 6
    assert report.data == str(FIXTURES)
    assert report.checkpoint is None

    checkpoint = tmp_path / "ckpt"
    save_checkpoint(checkpoint, WorldModel(ModelConfig(latent_dim=4, hidden_dim=8), seed=0))
    rollout_out = tmp_path / "rollout.json"
    assert (
        main(
            [
                "rollout",
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(FIXTURES),
                "--out",
                str(rollout_out),
                "--horizon",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    loaded = read_predict_report(rollout_out)
    assert loaded.checkpoint == str(checkpoint)
    assert loaded.mse_by_h == report.mse_by_h
    assert loaded.n_episodes == 6


def test_predict_cli_rejects_a_missing_corpus_and_a_zero_horizon(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing:
        main(["eval-predict", "--data", "missing-corpus", "--out", str(tmp_path / "out.json")])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as horizon:
        main(
            [
                "rollout",
                "--data",
                str(FIXTURES),
                "--out",
                str(tmp_path / "out.json"),
                "--horizon",
                "0",
            ]
        )
    assert horizon.value.code == 2


def test_console_script_eval_predict(tmp_path: Path) -> None:
    script = Path(sys.executable).with_name("worldforge")
    out = tmp_path / "report.json"
    completed = subprocess.run(
        [
            str(script),
            "eval-predict",
            "--data",
            str(FIXTURES),
            "--out",
            str(out),
            "--horizon",
            "2",
            "--seed",
            "1",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "n_episodes: 6" in completed.stdout
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == PREDICT_REPORT_FORMAT
    assert payload["horizons"] == [1, 2]
    assert payload["n_episodes"] == 6
    assert len(payload["mse_by_h"]) == 2
