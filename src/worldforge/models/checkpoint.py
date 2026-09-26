"""Save and load a world model checkpoint.

A checkpoint is a directory:

* ``config.json`` — format id, package version, and :class:`ModelConfig`.
  No optimizer state, no host name, no credentials.
* ``weights.pt`` — the module ``state_dict`` only.

``load_checkpoint`` rebuilds the modules from the config and loads the
weights onto CPU. The init seed is not stored; the loaded tensors replace
the fresh initialization. Other files in the directory, including a Day 4
``metrics.jsonl`` or ``train.json``, are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from worldforge import __version__
from worldforge.models.config import CHECKPOINT_FORMAT, ModelConfig
from worldforge.models.world import WorldModel

_CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.pt"
_CONFIG_KEYS = frozenset({"format", "worldforge_version", "config"})


def save_checkpoint(path: str | Path, model: WorldModel) -> None:
    """Write ``config.json`` and ``weights.pt`` under ``path``.

    ``path`` is created if needed. A path that already names a file is
    rejected. Existing checkpoint files in a directory are overwritten;
    other files in that directory are left in place.
    """

    if not isinstance(model, WorldModel):
        raise TypeError("model must be a WorldModel")
    directory = Path(path)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "worldforge_version": __version__,
        "config": model.config.model_dump(mode="json"),
    }
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    directory.joinpath(_CONFIG_NAME).write_text(text, encoding="utf-8")
    torch.save(model.state_dict(), directory.joinpath(_WEIGHTS_NAME))


def load_checkpoint(path: str | Path) -> WorldModel:
    """Rebuild a :class:`WorldModel` from a directory written by :func:`save_checkpoint`."""

    directory = Path(path)
    if not directory.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {directory}")
    config_path = directory.joinpath(_CONFIG_NAME)
    weights_path = directory.joinpath(_WEIGHTS_NAME)
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"checkpoint directory must contain {_CONFIG_NAME} and {_WEIGHTS_NAME}: {directory}"
        )
    config = _read_config(config_path)
    model = WorldModel(config, seed=0)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{_WEIGHTS_NAME} must contain a state_dict")
    model.load_state_dict(state)
    return model


def _read_config(path: Path) -> ModelConfig:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint config is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _CONFIG_KEYS:
        raise ValueError(
            "checkpoint config.json must contain format, worldforge_version, and config"
        )
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {payload['format']!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    if not isinstance(payload["worldforge_version"], str) or not payload["worldforge_version"]:
        raise ValueError("checkpoint worldforge_version must be a non-empty string")
    try:
        return ModelConfig.model_validate(payload["config"])
    except ValidationError as exc:
        raise ValueError(f"checkpoint config is invalid: {exc}") from exc
