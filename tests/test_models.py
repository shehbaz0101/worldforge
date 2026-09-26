"""Day 3 latent model. CPU only. No training loop and no network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from worldforge import __version__
from worldforge.cli import main
from worldforge.data import load_corpus, sample_transition_batch
from worldforge.envs.lotka_volterra import LotkaVolterraEnv
from worldforge.models import (
    ModelConfig,
    WorldModel,
    build_dynamics,
    format_model_info,
    load_checkpoint,
    save_checkpoint,
)
from worldforge.models.config import CHECKPOINT_FORMAT
from worldforge.models.summary import count_parameters, format_vector

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "trajectories"


def _affine_params(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def _mlp_params(in_features: int, hidden_dim: int, out_features: int, hidden_layers: int) -> int:
    sizes = [in_features, *([hidden_dim] * hidden_layers), out_features]
    return sum(
        _affine_params(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)
    )


def test_defaults_match_lotka_volterra() -> None:
    config = ModelConfig()
    assert config.obs_dim == LotkaVolterraEnv.observation_dim == 2
    assert config.action_dim == LotkaVolterraEnv.action_dim == 2
    assert config.latent_dim == 8
    assert config.hidden_dim == 64
    assert config.dynamics == "mlp"


def test_config_rejects_non_positive_width() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(latent_dim=0)


def test_encoder_dynamics_decoder_shapes() -> None:
    config = ModelConfig(latent_dim=8)
    model = WorldModel(config, seed=0)
    observation = torch.zeros(2)
    action = torch.zeros(2)
    prediction = model(observation, action)
    assert prediction.latent.shape == (8,)
    assert prediction.next_latent.shape == (8,)
    assert prediction.reconstructed.shape == (2,)
    assert prediction.predicted_observation.shape == (2,)
    assert torch.equal(prediction.predicted_observation, model.predict(observation, action))

    batched = model(torch.zeros(5, 2), torch.zeros(5, 2))
    assert batched.latent.shape == (5, 8)
    assert batched.next_latent.shape == (5, 8)
    assert batched.reconstructed.shape == (5, 2)
    assert batched.predicted_observation.shape == (5, 2)

    grid = model(torch.zeros(2, 3, 2), torch.zeros(2, 3, 2))
    assert grid.latent.shape == (2, 3, 8)
    assert grid.predicted_observation.shape == (2, 3, 2)


def test_parameter_counts_match_residual_mlp() -> None:
    config = ModelConfig()
    model = WorldModel(config, seed=0)
    encoder_n = _mlp_params(
        config.obs_dim, config.hidden_dim, config.latent_dim, config.encoder_hidden_layers
    )
    decoder_n = _mlp_params(
        config.latent_dim, config.hidden_dim, config.obs_dim, config.decoder_hidden_layers
    )
    dynamics_n = _mlp_params(
        config.latent_dim + config.action_dim,
        config.hidden_dim,
        config.latent_dim,
        config.dynamics_hidden_layers,
    )
    assert count_parameters(model.encoder) == encoder_n
    assert count_parameters(model.decoder) == decoder_n
    assert count_parameters(model.dynamics) == dynamics_n
    assert count_parameters(model) == encoder_n + decoder_n + dynamics_n
    assert model.encoder.describe().startswith("Linear(2, 64)")
    assert model.dynamics.describe().startswith("residual Linear(10, 64)")
    assert model.decoder.describe().startswith("Linear(8, 64)")


def test_seed_fixes_weights_and_forward() -> None:
    config = ModelConfig(hidden_dim=16)
    left = WorldModel(config, seed=4)
    right = WorldModel(config, seed=4)
    other = WorldModel(config, seed=5)
    for key, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[key])
    assert any(
        not torch.equal(value, other.state_dict()[key]) for key, value in left.state_dict().items()
    )
    observation = torch.tensor([1.2, 0.8])
    action = torch.tensor([0.1, -0.1])
    left_out = left(observation, action)
    right_out = right(observation, action)
    other_out = other(observation, action)
    assert torch.equal(left_out.latent, right_out.latent)
    assert torch.equal(left_out.next_latent, right_out.next_latent)
    assert torch.equal(left_out.predicted_observation, right_out.predicted_observation)
    assert not torch.equal(left_out.predicted_observation, other_out.predicted_observation)
    again = left(observation, action)
    assert torch.equal(left_out.predicted_observation, again.predicted_observation)


def test_init_does_not_advance_global_generator() -> None:
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    WorldModel(ModelConfig(hidden_dim=8), seed=7)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_train_and_eval_match() -> None:
    model = WorldModel(ModelConfig(hidden_dim=8), seed=1)
    observation = torch.tensor([0.5, 1.5])
    action = torch.zeros(2)
    model.train()
    training = model.predict(observation, action)
    model.eval()
    evaluation = model.predict(observation, action)
    assert torch.equal(training, evaluation)


def test_one_step_prediction_backward() -> None:
    model = WorldModel(ModelConfig(hidden_dim=16, latent_dim=4), seed=1)
    observation = torch.tensor([[1.0, 1.2], [0.7, 0.9]])
    action = torch.tensor([[0.0, 0.0], [0.1, -0.05]])
    target = torch.tensor([[1.1, 1.0], [0.8, 0.85]])
    loss = torch.nn.functional.mse_loss(model.predict(observation, action), target)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )


def test_rejects_bad_inputs() -> None:
    model = WorldModel(ModelConfig(latent_dim=4, hidden_dim=8), seed=0)
    with pytest.raises(TypeError, match="torch.Tensor"):
        model.encode([1.0, 1.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating"):
        model.encode(torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="feature dimension"):
        model.encode(torch.zeros(3))
    with pytest.raises(ValueError, match="leading shapes"):
        model.step(torch.zeros(4, 4), torch.zeros(3, 2))


def test_seed_must_be_a_non_negative_int() -> None:
    with pytest.raises(ValueError, match="seed"):
        WorldModel(ModelConfig(), seed=-1)
    with pytest.raises(TypeError, match="seed"):
        WorldModel(ModelConfig(), seed=True)  # type: ignore[arg-type]


def test_rssm_is_reserved() -> None:
    config = ModelConfig(dynamics="rssm", hidden_dim=8)
    with pytest.raises(NotImplementedError, match="rssm"):
        WorldModel(config, seed=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    with pytest.raises(NotImplementedError, match="mlp"):
        build_dynamics(config, generator)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ModelConfig(
        latent_dim=4,
        hidden_dim=8,
        encoder_hidden_layers=1,
        dynamics_hidden_layers=1,
        decoder_hidden_layers=1,
    )
    model = WorldModel(config, seed=3)
    directory = tmp_path / "ckpt"
    directory.mkdir()
    notes = directory / "notes.txt"
    notes.write_text("keep\n", encoding="utf-8")
    save_checkpoint(directory, model)
    assert notes.read_text(encoding="utf-8") == "keep\n"
    payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert set(payload) == {"format", "worldforge_version", "config"}
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["worldforge_version"] == __version__
    assert payload["config"]["latent_dim"] == 4
    raw = (directory / "config.json").read_text(encoding="utf-8").lower()
    assert "secret" not in raw
    assert "token" not in raw
    assert "api_key" not in raw
    loaded = load_checkpoint(directory)
    assert loaded.config == config
    assert next(loaded.parameters()).device.type == "cpu"
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    observation = torch.tensor([1.3, 0.9])
    action = torch.tensor([-0.05, 0.2])
    assert torch.equal(model.predict(observation, action), loaded.predict(observation, action))
    fresh = WorldModel(config, seed=0)
    assert any(
        not torch.equal(value, fresh.state_dict()[key])
        for key, value in loaded.state_dict().items()
    )


def test_checkpoint_rejects_a_bad_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        load_checkpoint(tmp_path / "missing")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not a checkpoint")
    model = WorldModel(ModelConfig(hidden_dim=8), seed=0)
    with pytest.raises(ValueError, match="not a directory"):
        save_checkpoint(weights, model)


def test_checkpoint_rejects_unknown_format_and_extra_keys(tmp_path: Path) -> None:
    model = WorldModel(ModelConfig(hidden_dim=8, latent_dim=4), seed=1)
    directory = tmp_path / "ok"
    save_checkpoint(directory, model)
    config_path = directory / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["format"] = "worldforge.checkpoint.v0"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(directory)
    save_checkpoint(directory, model)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["token"] = "nope"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config.json"):
        load_checkpoint(directory)


def test_forward_on_golden_fixture_transitions() -> None:
    episodes = load_corpus(FIXTURES)
    assert len(episodes) == 6
    observations: list[list[float]] = []
    actions: list[list[float]] = []
    for episode in episodes:
        for transition in episode.transitions:
            observations.append(list(transition.observation.values))
            actions.append(list(transition.action.values))
    assert len(observations) == 48
    model = WorldModel(ModelConfig(), seed=0)
    model.eval()
    observation = torch.tensor(observations, dtype=torch.float32)
    action = torch.tensor(actions, dtype=torch.float32)
    prediction = model(observation, action)
    assert prediction.latent.shape == (48, 8)
    assert prediction.next_latent.shape == (48, 8)
    assert prediction.reconstructed.shape == (48, 2)
    assert prediction.predicted_observation.shape == (48, 2)
    assert bool(torch.isfinite(prediction.predicted_observation).all())
    assert bool(torch.isfinite(prediction.reconstructed).all())
    assert bool(torch.isfinite(prediction.latent).all())
    one = model(observation[0], action[0])
    # A single row and the same row inside a batch can differ in the last bits.
    assert torch.allclose(one.predicted_observation, prediction.predicted_observation[0], atol=1e-6)
    assert torch.allclose(one.latent, prediction.latent[0], atol=1e-6)
    batch = sample_transition_batch(episodes, batch_size=8, seed=1)
    replay_obs = torch.tensor(batch.observations, dtype=torch.float32)
    replay_act = torch.tensor(batch.actions, dtype=torch.float32)
    replay = model.predict(replay_obs, replay_act)
    assert replay.shape == (8, 2)
    assert torch.equal(replay, model.predict(replay_obs, replay_act))


def test_model_info_matches_formatter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model-info"]) == 0
    model = WorldModel(ModelConfig(), seed=1)
    assert capsys.readouterr().out == format_model_info(model) + "\n"


def test_model_info_custom_width(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model-info", "--latent-dim", "4", "--hidden-dim", "16"]) == 0
    model = WorldModel(ModelConfig(latent_dim=4, hidden_dim=16), seed=0)
    assert capsys.readouterr().out == format_model_info(model) + "\n"


def test_model_info_rejects_rssm() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["model-info", "--dynamics", "rssm"])
    assert caught.value.code == 2


def test_forward_smoke_synthetic_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["forward-smoke", "--seed", "2"]) == 0
    first = capsys.readouterr().out
    assert "source: synthetic" in first
    assert "latent_shape: (8,)" in first
    assert "predicted_shape: (2,)" in first
    assert "finite: true" in first
    assert "observation: 1 1" in first
    assert "action: 0 0" in first
    assert main(["forward-smoke", "--seed", "2"]) == 0
    assert capsys.readouterr().out == first


def test_forward_smoke_golden_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    episode = load_corpus(FIXTURES)[0]
    step = episode.transitions[0]
    assert (
        main(
            [
                "forward-smoke",
                "--fixture",
                str(FIXTURES),
                "--episode",
                "0",
                "--transition",
                "0",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "source: fixture" in out
    assert "episode: 0" in out
    assert "transition: 0" in out
    assert "finite: true" in out
    assert "predicted_shape: (2,)" in out
    assert format_vector(step.observation.values) in out
    assert format_vector(step.action.values) in out
    assert format_vector(step.next_observation.values) in out


def test_forward_smoke_rejects_a_bad_index() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["forward-smoke", "--fixture", str(FIXTURES), "--transition", "99"])
    assert caught.value.code == 2


def test_console_script_model_info() -> None:
    script = Path(sys.executable).with_name("worldforge")
    completed = subprocess.run(
        [str(script), "model-info"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "dynamics: mlp" in completed.stdout
    assert "baseline: deterministic residual MLP" in completed.stdout
    assert "parameters:" in completed.stdout
    assert completed.stderr == ""


def test_root_import_does_not_import_torch() -> None:
    code = (
        "import sys, worldforge, worldforge.models.config, worldforge.cli; "
        "assert 'torch' not in sys.modules; "
        "assert 'worldforge.models.world' not in sys.modules; "
        "assert 'worldforge.train' not in sys.modules; "
        "assert 'worldforge.eval' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
