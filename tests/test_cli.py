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


def test_collect_and_dataset_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "corpus"
    assert (
        main(
            [
                "collect",
                "--out",
                str(out),
                "--n-episodes",
                "3",
                "--horizon",
                "4",
                "--seed",
                "5",
                "--policy",
                "sine_dose",
            ]
        )
        == 0
    )
    collected = capsys.readouterr().out
    assert "episodes: 3" in collected
    assert "policies: sine_dose" in collected
    assert "kind: directory" in collected
    assert (out / "corpus.jsonl").is_file()
    assert (out / "manifest.json").is_file()
    assert (out / "episodes" / "ep_0002.json").is_file()

    assert main(["dataset-info", str(out / "corpus.jsonl")]) == 0
    info = capsys.readouterr().out
    assert "kind: corpus-jsonl" in info
    assert "episodes: 3" in info
    assert "transitions: 12" in info
    assert "policies: sine_dose" in info
    assert "lotka_volterra" in info


def test_collect_rejects_bad_seeds() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["collect", "--out", "unused", "--seeds", "1,a"])
    assert caught.value.code == 2


def test_dataset_info_reads_checked_in_fixtures(capsys: pytest.CaptureFixture[str]) -> None:
    fixtures = ROOT / "tests" / "fixtures" / "trajectories"
    assert main(["dataset-info", str(fixtures)]) == 0
    out = capsys.readouterr().out
    assert "kind: directory" in out
    assert "episodes: 6" in out
    assert "transitions: 48" in out
    assert "horizons: 8" in out
    assert "policies: zero random sine_dose" in out
    assert "env_ids: lotka_volterra" in out


def test_dataset_info_missing_path() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["dataset-info", "missing-corpus"])
    assert caught.value.code == 2


def test_env_demo_accepts_sine_dose(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env-demo", "--steps", "4", "--seed", "1", "--policy", "sine_dose"]) == 0
    out = capsys.readouterr().out
    assert "policy: sine_dose" in out
    assert "steps: 4" in out


def test_console_script_collect_and_dataset_info(tmp_path: Path) -> None:
    script = Path(sys.executable).with_name("worldforge")
    out = tmp_path / "corpus"
    collected = subprocess.run(
        [
            script,
            "collect",
            "--out",
            str(out),
            "--n-episodes",
            "2",
            "--horizon",
            "3",
            "--policy",
            "zero",
            "--format",
            "jsonl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert collected.returncode == 0, collected.stderr
    assert "episodes: 2" in collected.stdout
    assert (out / "corpus.jsonl").is_file()
    assert not (out / "episodes").exists()

    info = subprocess.run(
        [script, "dataset-info", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert info.returncode == 0, info.stderr
    assert "episodes: 2" in info.stdout
    assert "policies: zero" in info.stdout
    assert info.stderr == ""


def test_module_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "worldforge", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == f"worldforge {__version__}\n"
