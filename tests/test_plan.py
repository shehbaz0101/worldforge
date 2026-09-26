"""Day 6 CEM planner and closed-loop regret. CPU only. No network."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from worldforge.cli import main
from worldforge.envs import make_env
from worldforge.models import ModelConfig, WorldModel, save_checkpoint
from worldforge.plan import (
    ACTION_SEQUENCE_FORMAT,
    OBJECTIVE_NAME,
    PLAN_REPORT_FORMAT,
    REGRET_DEFINITION,
    CEMConfig,
    action_sequence_report,
    format_plan_report,
    plan_actions,
    planning_regret,
    read_action_sequence,
    read_plan_report,
    regulation_reward,
    score_action_sequence,
    write_action_sequence,
    write_plan_report,
)
from worldforge.plan.objective import default_target

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "trajectories"
LIMIT = 0.25


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=4,
        hidden_dim=8,
        encoder_hidden_layers=1,
        dynamics_hidden_layers=1,
        decoder_hidden_layers=1,
    )


def _tiny_cem(**overrides: object) -> CEMConfig:
    settings: dict[str, object] = {
        "horizon": 2,
        "n_samples": 4,
        "n_iterations": 1,
        "seed": 0,
    }
    settings.update(overrides)
    return CEMConfig(**settings)  # type: ignore[arg-type]


def _identity(layer: torch.nn.Linear) -> None:
    width = layer.weight.shape[0]
    eye = torch.eye(width, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(eye)
        layer.bias.zero_()


def _shift_model() -> WorldModel:
    """Decoder and encoder are identity on positive states.

    One dynamics step adds the action to the latent, so the decoded
    population moves by the dose. The map is exact while every population
    stays positive, which the improvement test's horizon keeps true.
    """

    config = ModelConfig(
        obs_dim=2,
        action_dim=2,
        latent_dim=2,
        hidden_dim=2,
        encoder_hidden_layers=1,
        dynamics_hidden_layers=1,
        decoder_hidden_layers=1,
    )
    model = WorldModel(config, seed=0)
    _identity(model.encoder.net[0])
    _identity(model.encoder.net[2])
    _identity(model.decoder.net[0])
    _identity(model.decoder.net[2])
    with torch.no_grad():
        model.dynamics.net[0].weight.copy_(
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
        model.dynamics.net[0].bias.copy_(torch.tensor([1.0, 1.0]))
        model.dynamics.net[2].weight.copy_(torch.eye(2))
        model.dynamics.net[2].bias.copy_(torch.tensor([-1.0, -1.0]))
    model.eval()
    return model


def test_regulation_reward_matches_the_environment() -> None:
    env = make_env(horizon=4)
    env.reset(seed=2)
    nxt, reward, _terminated, _truncated, _info = env.step((0.1, -0.2))
    scored = regulation_reward(torch.tensor(nxt), torch.tensor(default_target(2)))
    assert float(scored) == pytest.approx(reward)
    assert default_target(2) == (1.0, 1.0)
    assert OBJECTIVE_NAME == "regulation_l1"


def test_shift_model_adds_the_action() -> None:
    model = _shift_model()
    observation = torch.tensor([1.5, 0.6])
    action = torch.tensor([-0.25, 0.25])
    predicted = model.predict(observation, action)
    assert torch.allclose(predicted, torch.tensor([1.25, 0.85]))


def test_cem_beats_zero_and_random_on_the_shift_model() -> None:
    model = _shift_model()
    observation = torch.tensor([1.5, 0.6])
    horizon = 2
    config = CEMConfig(horizon=horizon, n_samples=32, n_iterations=4, seed=0)
    plan = plan_actions(model, observation, config)
    zero = score_action_sequence(model, observation, torch.zeros(horizon, 2)).detach()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    draw = (torch.rand((24, horizon, 2), generator=generator) * (2 * LIMIT)) - LIMIT
    random_scores = [
        float(score_action_sequence(model, observation, draw[index]).detach())
        for index in range(draw.shape[0])
    ]
    random_mean = sum(random_scores) / len(random_scores)
    assert plan.predicted_return > float(zero)
    assert plan.predicted_return > random_mean
    assert plan.actions.shape == (horizon, 2)
    assert float(plan.actions[0, 0]) < -0.05
    assert float(plan.actions[0, 1]) > 0.05
    again = score_action_sequence(model, observation, plan.actions).detach()
    assert float(again) == pytest.approx(plan.predicted_return)


def test_plan_is_at_least_as_good_as_zero_and_stays_inside_bounds() -> None:
    model = WorldModel(_tiny_config(), seed=1)
    observation = torch.tensor([1.4, 0.7])
    config = CEMConfig(horizon=2, n_samples=4, n_iterations=1, seed=3, init_std=10.0)
    plan = plan_actions(model, observation, config)
    zero = float(score_action_sequence(model, observation, torch.zeros(2, 2)).detach())
    assert plan.predicted_return >= zero - 1e-6
    assert bool(torch.all(plan.actions <= LIMIT))
    assert bool(torch.all(plan.actions >= -LIMIT))
    assert plan.actions.dtype == torch.float32


def test_plan_is_deterministic_and_does_not_touch_the_global_rng() -> None:
    model = WorldModel(_tiny_config(), seed=2)
    model.train()
    observation = torch.tensor([1.2, 0.8])
    config = _tiny_cem(seed=5)
    torch.manual_seed(123)
    before_torch = torch.get_rng_state().clone()
    before_random = random.getstate()
    first = plan_actions(model, observation, config)
    second = plan_actions(model, observation, config)
    assert torch.equal(first.actions, second.actions)
    assert first.predicted_return == second.predicted_return
    assert torch.equal(before_torch, torch.get_rng_state())
    assert random.getstate() == before_random
    assert model.training is True


def test_closed_loop_report_schema_and_repeatability(tmp_path: Path) -> None:
    model = WorldModel(_tiny_config(), seed=0)
    config = _tiny_cem(seed=1)
    report = planning_regret(model, n_steps=2, env_seed=4, cem=config)
    assert report.env_id == "lotka_volterra"
    assert report.objective == OBJECTIVE_NAME
    assert report.n_steps == 2
    assert report.planner_steps == 2
    assert report.horizon == 2
    assert len(report.actions) == 2
    assert len(report.actions[0]) == 2
    assert report.baseline_best == max(report.zero_return, report.random_return)
    assert report.regret == pytest.approx(report.baseline_best - report.planner_return)
    assert all(
        math.isfinite(value)
        for value in (report.planner_return, report.zero_return, report.random_return, report.regret)
    )
    env = make_env(horizon=2)
    initial = env.reset(4)
    opening = plan_actions(model, torch.tensor(initial, dtype=torch.float32), config)
    assert report.actions[0] == tuple(float(value) for value in opening.actions[0].tolist())
    assert report.initial_observation == tuple(float(value) for value in initial)

    path = tmp_path / "plan.json"
    write_plan_report(path, report)
    loaded = read_plan_report(path)
    assert loaded == report
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == PLAN_REPORT_FORMAT
    assert payload["regret_definition"] == REGRET_DEFINITION
    assert set(payload) >= {"planner_return", "zero_return", "random_return", "baseline_best", "regret"}
    text = format_plan_report(report)
    assert f"format: {PLAN_REPORT_FORMAT}" in text
    assert f"regret_definition: {REGRET_DEFINITION}" in text
    assert "regret:" in text

    again = planning_regret(model, n_steps=2, env_seed=4, cem=config)
    assert again.actions == report.actions
    assert again.planner_return == report.planner_return
    assert again.regret == report.regret


def test_plan_report_rejects_a_broken_regret(tmp_path: Path) -> None:
    model = WorldModel(_tiny_config(), seed=0)
    report = planning_regret(model, n_steps=2, env_seed=0, cem=_tiny_cem())
    path = tmp_path / "plan.json"
    write_plan_report(path, report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["regret"] = payload["regret"] + 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="regret"):
        read_plan_report(path)
    payload["regret"] = report.regret
    payload["format"] = "worldforge.plan.v0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        read_plan_report(path)


def test_action_sequence_roundtrip(tmp_path: Path) -> None:
    model = WorldModel(_tiny_config(), seed=0)
    observation = torch.tensor([1.0, 1.0])
    config = _tiny_cem()
    plan = plan_actions(model, observation, config)
    report = action_sequence_report(plan, config, observation.tolist(), checkpoint="checkpoints/day6")
    path = tmp_path / "actions.json"
    write_action_sequence(path, report)
    loaded = read_action_sequence(path)
    assert loaded == report
    assert loaded.horizon == len(loaded.actions)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == ACTION_SEQUENCE_FORMAT
    assert payload["objective"] == OBJECTIVE_NAME
    assert payload["checkpoint"] == "checkpoints/day6"
    extra = dict(payload)
    extra["extra"] = 1
    path.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        read_action_sequence(path)


def test_plan_and_eval_plan_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sequence_path = tmp_path / "sequence.json"
    assert (
        main(
            [
                "plan",
                "--out",
                str(sequence_path),
                "--data",
                str(FIXTURES),
                "--episode",
                "0",
                "--horizon",
                "2",
                "--samples",
                "4",
                "--iterations",
                "1",
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
    assert f"format: {ACTION_SEQUENCE_FORMAT}" in printed.out
    assert f"report: {sequence_path}" in printed.out
    sequence = read_action_sequence(sequence_path)
    assert sequence.horizon == 2
    assert sequence.checkpoint is None
    assert len(sequence.observation) == 2

    checkpoint = tmp_path / "ckpt"
    save_checkpoint(checkpoint, WorldModel(_tiny_config(), seed=0))
    report_path = tmp_path / "regret.json"
    assert (
        main(
            [
                "eval-plan",
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(report_path),
                "--n-steps",
                "2",
                "--horizon",
                "2",
                "--samples",
                "4",
                "--iterations",
                "1",
                "--seed",
                "0",
                "--env-seed",
                "1",
            ]
        )
        == 0
    )
    printed = capsys.readouterr()
    assert f"format: {PLAN_REPORT_FORMAT}" in printed.out
    assert "regret_definition:" in printed.out
    assert f"checkpoint: {checkpoint}" in printed.out
    loaded = read_plan_report(report_path)
    assert loaded.checkpoint == str(checkpoint)
    assert loaded.seed == 1
    assert loaded.n_steps == 2
    assert loaded.planner_steps == 2


def test_plan_cli_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing:
        main(["plan", "--out", str(tmp_path / "out.json"), "--data", "missing-corpus"])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as both:
        main(
            [
                "plan",
                "--out",
                str(tmp_path / "out.json"),
                "--obs",
                "1,1",
                "--data",
                str(FIXTURES),
            ]
        )
    assert both.value.code == 2
    with pytest.raises(SystemExit) as horizon:
        main(
            [
                "eval-plan",
                "--out",
                str(tmp_path / "out.json"),
                "--horizon",
                "0",
                "--latent-dim",
                "4",
                "--hidden-dim",
                "8",
            ]
        )
    assert horizon.value.code == 2


def test_console_script_eval_plan(tmp_path: Path) -> None:
    script = Path(sys.executable).with_name("worldforge")
    out = tmp_path / "report.json"
    completed = subprocess.run(
        [
            str(script),
            "eval-plan",
            "--out",
            str(out),
            "--n-steps",
            "2",
            "--horizon",
            "2",
            "--samples",
            "4",
            "--iterations",
            "1",
            "--seed",
            "0",
            "--env-seed",
            "0",
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
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == PLAN_REPORT_FORMAT
    assert payload["regret_definition"] == REGRET_DEFINITION
    assert payload["n_steps"] == 2


def test_cem_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="horizon"):
        CEMConfig(horizon=0)
    with pytest.raises(ValueError, match="elite_fraction"):
        CEMConfig(elite_fraction=0.0)
    with pytest.raises(ValueError, match="min_std"):
        CEMConfig(init_std=0.01, min_std=0.1)
    with pytest.raises(TypeError, match="model"):
        plan_actions(object(), torch.zeros(2), _tiny_cem())  # type: ignore[arg-type]


def test_action_plan_type_guard() -> None:
    with pytest.raises(TypeError, match="ActionPlan"):
        action_sequence_report(
            object(),  # type: ignore[arg-type]
            _tiny_cem(),
            [1.0, 1.0],
        )
