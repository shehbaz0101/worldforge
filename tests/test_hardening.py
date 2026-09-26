"""Day 8 path sandbox, rate limit, and offline guard. TestClient only."""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from worldforge.api import app
from worldforge.cli import main
from worldforge.offline import OfflineError, install_offline_guard
from worldforge.ratelimit import RateLimiter, reset_rate_limiter
from worldforge.sandbox import PathSandboxError, resolve_user_path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "trajectories"
TINY = {
    "latent_dim": 4,
    "hidden_dim": 8,
    "encoder_hidden_layers": 1,
    "dynamics_hidden_layers": 1,
    "decoder_hidden_layers": 1,
}


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


def test_sandbox_allows_a_child_and_dotdot_that_stays_inside(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    child = resolve_user_path(nested, label="data", root=tmp_path)
    stayed = resolve_user_path(tmp_path / "a" / ".." / "a" / "b", label="data", root=tmp_path)
    relative = resolve_user_path("a/b", label="data", root=tmp_path)
    assert child == stayed == relative == nested.resolve()


def test_sandbox_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(PathSandboxError, match="escapes the data root"):
        resolve_user_path("..", label="data", root=tmp_path)
    with pytest.raises(PathSandboxError, match="path traversal"):
        resolve_user_path("nested/../../outside", label="--out", root=tmp_path)


def test_sandbox_rejects_an_absolute_escape(tmp_path: Path) -> None:
    with pytest.raises(PathSandboxError, match="absolute path outside the sandbox"):
        resolve_user_path("/etc/passwd", label="checkpoint", root=tmp_path)


def test_sandbox_rejects_a_symlink_that_leaves_the_root(tmp_path: Path) -> None:
    link = tmp_path / "escape"
    link.symlink_to("/etc/passwd")
    with pytest.raises(PathSandboxError, match="escapes the data root"):
        resolve_user_path(link, label="data", root=tmp_path)


def test_http_accepts_a_corpus_inside_the_root() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/eval/predict",
            json={
                "data": "tests/fixtures/../fixtures/trajectories",
                "horizon": 1,
                "seed": 0,
                **TINY,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_episodes"] == 6
    assert Path(payload["data"]) == FIXTURES.resolve()
    assert payload["format"] == "worldforge.predict.v1"


def test_http_rejects_traversal_and_absolute_escape() -> None:
    with TestClient(app) as client:
        traversal = client.post("/rollout", json={"data": "../etc/passwd", "horizon": 1})
        absolute = client.post("/eval/predict", json={"data": "/etc/passwd", "horizon": 1})
        checkpoint = client.post(
            "/plan",
            json={"observation": [1.0, 1.0], "checkpoint": "/etc/passwd", "horizon": 1},
        )
    for response in (traversal, absolute, checkpoint):
        assert response.status_code == 422
        assert "escapes the data root" in response.json()["detail"]


def test_http_rejects_a_symlink_escape(tmp_path: Path) -> None:
    link = tmp_path / "passwd-link"
    link.symlink_to("/etc/passwd")
    with TestClient(app) as client:
        response = client.post("/rollout", json={"data": str(link), "horizon": 1})
    assert response.status_code == 422
    assert "escapes the data root" in response.json()["detail"]


def test_http_missing_path_inside_the_root_is_not_an_escape() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/rollout",
            json={"data": "tests/fixtures/trajectories/missing.json", "horizon": 1},
        )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "does not exist" in detail
    assert "escapes" not in detail


def test_http_train_out_outside_the_root_is_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/train",
            json={"trajectories": [_episode_payload()], "out": "/etc/worldforge-day8-escape"},
        )
    assert response.status_code == 422
    assert "out escapes the data root" in response.json()["detail"]


def test_rate_limit_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFORGE_RATE_LIMIT", "2")
    monkeypatch.setenv("WORLDFORGE_RATE_WINDOW_SECONDS", "60")
    reset_rate_limiter()
    # Unknown fields are HTTP 422 and still count. Empty /plan is a valid plan.
    rejected = {"not_a_field": True}
    with TestClient(app) as client:
        first = client.post("/plan", json=rejected)
        second = client.post("/train", json=rejected)
        third = client.post("/eval/plan", json=rejected)
        health = client.get("/health")
    assert first.status_code == 422
    assert second.status_code == 422
    assert third.status_code == 429
    assert "rate limit exceeded" in third.json()["detail"]
    assert int(third.headers["retry-after"]) >= 1
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_health_stays_up_under_load_and_does_not_spend_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFORGE_RATE_LIMIT", "1")
    monkeypatch.setenv("WORLDFORGE_RATE_WINDOW_SECONDS", "60")
    reset_rate_limiter()
    with TestClient(app) as client:
        for _ in range(40):
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
        allowed = client.post("/rollout", json={})
        assert allowed.status_code == 422
        for _ in range(20):
            assert client.get("/health").status_code == 200
        blocked = client.post("/eval/predict", json={})
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["version"]


def test_openapi_documents_429_on_expensive_routes_only() -> None:
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
    for path in ("/rollout", "/eval/predict", "/plan", "/eval/plan", "/train"):
        assert "429" in paths[path]["post"]["responses"]
    assert "429" not in paths["/health"]["get"]["responses"]


def test_retry_after_is_the_rest_of_the_window() -> None:
    limiter = RateLimiter(1, 60)
    assert limiter.check("client", now=1000.0) is None
    assert limiter.check("client", now=1000.0) == 60
    assert limiter.check("client", now=1059.2) == 1
    assert limiter.check("client", now=1060.0) is None


def test_cli_rejects_traversal_and_absolute_escape(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as parent:
        main(["dataset-info", ".."])
    assert parent.value.code == 2
    assert "escapes the data root" in capsys.readouterr().err

    with pytest.raises(SystemExit) as absolute:
        main(["dataset-info", "/etc/passwd"])
    assert absolute.value.code == 2
    err = capsys.readouterr().err
    assert "escapes the data root" in err
    assert "does not exist" not in err


def test_cli_data_root_allows_a_directory_outside_the_cwd(
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = Path(tempfile.mkdtemp(prefix="worldforge-day8-"))
    try:
        with pytest.raises(SystemExit) as caught:
            main(
                [
                    "collect",
                    "--out",
                    str(outside / "corpus"),
                    "--n-episodes",
                    "1",
                    "--horizon",
                    "1",
                    "--policy",
                    "zero",
                ]
            )
        assert caught.value.code == 2
        assert "escapes the data root" in capsys.readouterr().err
        assert (
            main(
                [
                    "collect",
                    "--data-root",
                    str(outside),
                    "--out",
                    str(outside / "corpus"),
                    "--n-episodes",
                    "1",
                    "--horizon",
                    "1",
                    "--policy",
                    "zero",
                ]
            )
            == 0
        )
        assert (outside / "corpus" / "manifest.json").is_file()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_serve_rejects_a_bad_rate_limit() -> None:
    with pytest.raises(SystemExit) as limit:
        main(["serve", "--rate-limit", "0"])
    assert limit.value.code == 2
    with pytest.raises(SystemExit) as window:
        main(["serve", "--rate-window", "0"])
    assert window.value.code == 2


def test_serve_passes_data_root_and_rate_limit_then_restores_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import uvicorn

    seen: dict[str, object] = {}

    def fake_run(target: str, **kwargs: object) -> None:
        seen["target"] = target
        seen["root"] = os.environ.get("WORLDFORGE_DATA_ROOT")
        seen["limit"] = os.environ.get("WORLDFORGE_RATE_LIMIT")
        seen["window"] = os.environ.get("WORLDFORGE_RATE_WINDOW_SECONDS")
        seen["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    previous_root = os.environ.get("WORLDFORGE_DATA_ROOT")
    previous_limit = os.environ.get("WORLDFORGE_RATE_LIMIT")
    previous_window = os.environ.get("WORLDFORGE_RATE_WINDOW_SECONDS")
    root = tmp_path / "data-root"
    root.mkdir()
    assert (
        main(
            [
                "serve",
                "--data-root",
                str(root),
                "--rate-limit",
                "3",
                "--rate-window",
                "15",
            ]
        )
        == 0
    )
    assert seen["target"] == "worldforge.api:app"
    assert seen["root"] == str(root.resolve())
    assert seen["limit"] == "3"
    assert seen["window"] == "15.0"
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "127.0.0.1"
    assert os.environ.get("WORLDFORGE_DATA_ROOT") == previous_root
    assert os.environ.get("WORLDFORGE_RATE_LIMIT") == previous_limit
    assert os.environ.get("WORLDFORGE_RATE_WINDOW_SECONDS") == previous_window


def test_offline_guard_refuses_remote_connects_and_allows_loopback() -> None:
    install_offline_guard()
    with pytest.raises(OfflineError, match="offline-by-design"):
        socket.create_connection(("example.com", 80), timeout=0.2)
    with pytest.raises(OfflineError):
        socket.create_connection(("8.8.8.8", 53), timeout=0.2)
    try:
        connected = socket.create_connection(("127.0.0.1", 9), timeout=0.3)
    except OfflineError:
        raise
    except OSError:
        return
    connected.close()
