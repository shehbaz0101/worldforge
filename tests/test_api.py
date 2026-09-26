"""HTTP API. TestClient only. No network and no integration marker."""

from __future__ import annotations

import importlib
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from worldforge import __version__
from worldforge.api.schemas import HTTP_MAX_EPOCHS, HTTP_MAX_STEPS_PER_EPOCH
from worldforge.api.service import ML_EXTRA_DETAIL
from worldforge.cli import main
from worldforge.data import load_corpus
from worldforge.envs.lotka_volterra import ACTION_LIMIT
from worldforge.eval import PREDICT_REPORT_FORMAT, open_loop_metrics, read_predict_report
from worldforge.models import ModelConfig, WorldModel, load_checkpoint, save_checkpoint
from worldforge.plan import (
    ACTION_SEQUENCE_FORMAT,
    PLAN_REPORT_FORMAT,
    REGRET_DEFINITION,
    CEMConfig,
    action_sequence_report,
    plan_actions,
    planning_regret,
    read_action_sequence,
    read_plan_report,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "trajectories"
TINY = {
    "latent_dim": 4,
    "hidden_dim": 8,
    "encoder_hidden_layers": 1,
    "dynamics_hidden_layers": 1,
    "decoder_hidden_layers": 1,
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("API test must stay offline")

    monkeypatch.setattr(urllib.request, "urlopen", explode)


@pytest.fixture
def client() -> Iterator[TestClient]:
    from worldforge.api import app

    with TestClient(app) as test_client:
        yield test_client


def _config() -> ModelConfig:
    return ModelConfig(
        latent_dim=4,
        hidden_dim=8,
        encoder_hidden_layers=1,
        dynamics_hidden_layers=1,
        decoder_hidden_layers=1,
    )


def _bounds() -> dict[str, tuple[float, float]]:
    low = (-ACTION_LIMIT, -ACTION_LIMIT)
    high = (ACTION_LIMIT, ACTION_LIMIT)
    return {"action_low": low, "action_high": high}


def _episode_payload() -> dict[str, object]:
    return {
        "env_id": "lotka_volterra",
        "seed": 0,
        "horizon": 1,
        "transitions": [
            {
                "observation": {"values": [1.0, 1.0], "t": 0.0},
                "action": {"values": [0.0, 0.0]},
                "reward": 0.0,
                "next_observation": {"values": [1.1, 0.9], "t": 0.2},
                "terminated": False,
                "truncated": False,
            }
        ],
    }


def test_health_reports_the_package_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_openapi_lists_the_day7_routes(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert set(paths) >= {"/health", "/rollout", "/eval/predict", "/plan", "/eval/plan", "/train"}
    assert "get" in paths["/health"]
    assert "post" in paths["/rollout"]
    assert "post" in paths["/eval/predict"]
    assert "post" in paths["/plan"]
    assert "post" in paths["/eval/plan"]
    assert "post" in paths["/train"]


def test_importing_the_api_does_not_import_torch() -> None:
    script = (
        "import sys\n"
        "import worldforge.api\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "assert worldforge.api.app.title == 'WorldForge'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_predict_on_fixtures_matches_the_library(client: TestClient, tmp_path: Path) -> None:
    body = {"data": str(FIXTURES), "horizon": 1, "seed": 0, **TINY}
    response = client.post("/eval/predict", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == PREDICT_REPORT_FORMAT
    assert payload["n_episodes"] == 6
    assert payload["horizons"] == [1]
    assert payload["checkpoint"] is None
    assert payload["data"] == str(FIXTURES)
    assert len(payload["mse_by_h"]) == 1
    assert all(value >= 0 for value in payload["mse_by_h"])

    direct = open_loop_metrics(
        WorldModel(_config(), seed=0),
        load_corpus(FIXTURES),
        horizon=1,
        data=str(FIXTURES),
    )
    assert payload["mse_by_h"] == pytest.approx(list(direct.mse_by_h))
    assert payload["mae_by_h"] == pytest.approx(list(direct.mae_by_h))

    report_path = tmp_path / "predict.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = read_predict_report(report_path)
    assert loaded.n_episodes == 6
    assert loaded.horizons == (1,)


def test_rollout_matches_eval_predict(client: TestClient) -> None:
    body = {"data": str(FIXTURES), "horizon": 1, "seed": 1, **TINY}
    rollout = client.post("/rollout", json=body)
    predict = client.post("/eval/predict", json=body)
    assert rollout.status_code == 200
    assert predict.status_code == 200
    assert rollout.json() == predict.json()


def test_predict_accepts_an_inline_trajectory(client: TestClient) -> None:
    response = client.post(
        "/eval/predict",
        json={"trajectories": [_episode_payload()], "horizon": 1, "seed": 0, **TINY},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"] == "inline"
    assert payload["n_episodes"] == 1
    assert payload["horizons"] == [1]
    assert payload["checkpoint"] is None


def test_predict_loads_a_checkpoint_and_ignores_architecture_flags(
    client: TestClient,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checkpoint"
    save_checkpoint(directory, WorldModel(_config(), seed=2))
    response = client.post(
        "/rollout",
        json={
            "data": str(FIXTURES),
            "checkpoint": str(directory),
            "horizon": 1,
            "seed": 0,
            "latent_dim": 32,
            "hidden_dim": 32,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint"] == str(directory)
    direct = open_loop_metrics(
        load_checkpoint(directory),
        load_corpus(FIXTURES),
        horizon=1,
        checkpoint=str(directory),
        data=str(FIXTURES),
    )
    assert payload["mse_by_h"] == pytest.approx(list(direct.mse_by_h))
    assert load_checkpoint(directory).config.latent_dim == 4


def test_plan_matches_the_library(client: TestClient, tmp_path: Path) -> None:
    observation = [1.5, 0.6]
    body = {
        "observation": observation,
        "horizon": 2,
        "n_samples": 4,
        "n_iterations": 1,
        "seed": 0,
        **TINY,
    }
    response = client.post("/plan", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == ACTION_SEQUENCE_FORMAT
    assert payload["objective"] == "regulation_l1"
    assert payload["observation"] == pytest.approx(observation)
    assert len(payload["actions"]) == 2
    assert payload["checkpoint"] is None
    for action in payload["actions"]:
        assert len(action) == 2
        assert all(abs(component) <= ACTION_LIMIT + 1e-6 for component in action)

    model = WorldModel(_config(), seed=0)
    cem = CEMConfig(horizon=2, n_samples=4, n_iterations=1, seed=0, **_bounds())
    library = action_sequence_report(
        plan_actions(model, torch.tensor(observation), cem),
        cem,
        observation,
    )
    assert len(payload["actions"]) == len(library.actions)
    for got, expected in zip(payload["actions"], library.actions, strict=True):
        assert got == pytest.approx(list(expected))
    assert payload["predicted_return"] == pytest.approx(library.predicted_return)

    again = client.post("/plan", json=body)
    assert again.status_code == 200
    assert again.json()["actions"] == payload["actions"]

    report_path = tmp_path / "plan.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = read_action_sequence(report_path)
    assert loaded.horizon == 2
    assert loaded.seed == 0


def test_plan_defaults_to_the_equilibrium(client: TestClient) -> None:
    response = client.post(
        "/plan",
        json={"horizon": 1, "n_samples": 2, "n_iterations": 1, "seed": 1, **TINY},
    )
    assert response.status_code == 200
    assert response.json()["observation"] == pytest.approx([1.0, 1.0])


def test_plan_can_start_from_a_corpus_episode(client: TestClient) -> None:
    episode = load_corpus(FIXTURES)[0]
    expected = list(episode.transitions[0].observation.values)
    response = client.post(
        "/plan",
        json={
            "data": str(FIXTURES),
            "episode": 0,
            "horizon": 1,
            "n_samples": 2,
            "n_iterations": 1,
            "seed": 0,
            **TINY,
        },
    )
    assert response.status_code == 200
    assert response.json()["observation"] == pytest.approx(expected)


def test_eval_plan_reports_regret(client: TestClient, tmp_path: Path) -> None:
    body = {
        "n_steps": 1,
        "env_seed": 0,
        "horizon": 1,
        "n_samples": 2,
        "n_iterations": 1,
        "seed": 0,
        **TINY,
    }
    response = client.post("/eval/plan", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == PLAN_REPORT_FORMAT
    assert payload["regret_definition"] == REGRET_DEFINITION
    assert payload["n_steps"] == 1
    assert payload["planner_steps"] == 1
    assert payload["horizon"] == 1
    assert len(payload["actions"]) == 1
    assert payload["baseline_best"] == pytest.approx(
        max(payload["zero_return"], payload["random_return"])
    )
    assert payload["regret"] == pytest.approx(payload["baseline_best"] - payload["planner_return"])

    library = planning_regret(
        WorldModel(_config(), seed=0),
        n_steps=1,
        env_seed=0,
        cem=CEMConfig(horizon=1, n_samples=2, n_iterations=1, seed=0, **_bounds()),
    )
    assert payload["regret"] == pytest.approx(library.regret)
    assert payload["planner_return"] == pytest.approx(library.planner_return)

    report_path = tmp_path / "regret.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = read_plan_report(report_path)
    assert loaded.env_id == "lotka_volterra"
    assert loaded.n_steps == 1


def test_train_returns_final_loss_and_a_checkpoint(client: TestClient, tmp_path: Path) -> None:
    out = tmp_path / "run"
    response = client.post(
        "/train",
        json={
            "trajectories": [_episode_payload()],
            "out": str(out),
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 1,
            "seed": 0,
            **TINY,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint"] == str(out)
    assert payload["epochs"] == 1
    assert payload["steps_per_epoch"] == 1
    assert payload["n_episodes"] == 1
    assert payload["n_transitions"] == 1
    assert math.isfinite(payload["final_train_loss"])
    assert (out / "config.json").is_file()
    assert (out / "weights.pt").is_file()
    assert (out / "metrics.jsonl").is_file()
    assert (out / "train.json").is_file()
    restored = load_checkpoint(out)
    assert restored.config.latent_dim == 4


def test_train_without_out_writes_a_temp_directory(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "trajectories": [_episode_payload()],
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 1,
            **TINY,
        },
    )
    assert response.status_code == 200
    directory = Path(response.json()["checkpoint"])
    try:
        assert directory.is_dir()
        assert (directory / "weights.pt").is_file()
        assert math.isfinite(response.json()["final_train_loss"])
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_plan_without_torch_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def blocked(name: str, package: str | None = None) -> object:
        if name == "torch" or name.startswith("torch."):
            exc = ModuleNotFoundError("No module named 'torch'")
            exc.name = "torch"
            raise exc
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked)
    health = client.get("/health")
    response = client.post(
        "/plan",
        json={"observation": [1.0, 1.0], "horizon": 1, "n_samples": 1, "n_iterations": 1},
    )
    assert health.status_code == 200
    assert response.status_code == 503
    assert ML_EXTRA_DETAIL in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"horizon": 1},
        {"data": str(FIXTURES), "trajectories": [_episode_payload()]},
        {"data": "  ", "horizon": 1},
        {"data": str(FIXTURES), "extra": True},
        {"data": str(FIXTURES), "dynamics": "gru"},
        {"data": str(FIXTURES), "horizon": 0},
    ],
)
def test_predict_rejects_bad_bodies(client: TestClient, payload: dict[str, object]) -> None:
    response = client.post("/eval/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_path_is_422(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/rollout", json={"data": str(tmp_path / "missing"), "horizon": 1})
    assert response.status_code == 422
    assert "does not exist" in response.json()["detail"]


def test_plan_rejects_observation_and_data(client: TestClient) -> None:
    response = client.post(
        "/plan",
        json={"observation": [1.0, 1.0], "data": str(FIXTURES), "horizon": 1},
    )
    assert response.status_code == 422


def test_plan_rejects_reserved_dynamics(client: TestClient) -> None:
    response = client.post(
        "/plan",
        json={
            "observation": [1.0, 1.0],
            "dynamics": "rssm",
            "horizon": 1,
            "n_samples": 1,
            "n_iterations": 1,
        },
    )
    assert response.status_code == 422
    assert "rssm" in response.json()["detail"]


def test_train_rejects_a_long_run(client: TestClient) -> None:
    too_many_epochs = client.post(
        "/train",
        json={"trajectories": [_episode_payload()], "epochs": HTTP_MAX_EPOCHS + 1},
    )
    too_many_steps = client.post(
        "/train",
        json={
            "trajectories": [_episode_payload()],
            "steps_per_epoch": HTTP_MAX_STEPS_PER_EPOCH + 1,
        },
    )
    assert too_many_epochs.status_code == 422
    assert too_many_steps.status_code == 422


def test_serve_invokes_uvicorn_without_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, object] = {}

    def fake_run(target: str, **kwargs: object) -> None:
        seen["target"] = target
        seen["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    assert main(["serve"]) == 0
    assert seen["target"] == "worldforge.api:app"
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is False

    assert main(["serve", "--host", "0.0.0.0", "--port", "9001", "--reload"]) == 0
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9001
    assert kwargs["reload"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--port", "0"],
        ["serve", "--port", "70000"],
        ["serve", "--host", " "],
    ],
)
def test_serve_rejects_a_bad_bind(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2
