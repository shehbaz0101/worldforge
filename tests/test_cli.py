"""CLI smoke tests. No network."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from worldforge import __version__
from worldforge.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject() -> None:
    with ROOT.joinpath("pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["name"] == "worldforge"
    assert project["version"] == __version__ == "0.1.0"


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"worldforge {__version__}\n"
    assert captured.err == ""


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out == f"worldforge {__version__}\n"


def test_env_demo_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env-demo", "--steps", "4", "--seed", "3", "--policy", "zero"]) == 0
    out = capsys.readouterr().out
    assert "env_id: lotka_volterra" in out
    assert "seed: 3" in out
    assert "policy: zero" in out
    assert "horizon: 4" in out
    assert "steps: 4" in out
    assert "terminated: false" in out
    assert "truncated: true" in out
    assert "prey" in out
    assert "predator" in out


def test_env_demo_rejects_zero_steps() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["env-demo", "--steps", "0"])
    assert caught.value.code == 2


def test_console_script_version_and_env_demo() -> None:
    script = Path(sys.executable).with_name("worldforge")
    assert script.is_file()
    version = subprocess.run(
        [script, "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert version.returncode == 0
    assert version.stdout == f"worldforge {__version__}\n"
    assert version.stderr == ""

    demo = subprocess.run(
        [script, "env-demo", "--steps", "4", "--seed", "1"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert demo.returncode == 0
    assert "env_id: lotka_volterra" in demo.stdout
    assert "seed: 1" in demo.stdout
    assert demo.stderr == ""


def test_module_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "worldforge", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == f"worldforge {__version__}\n"
