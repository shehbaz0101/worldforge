"""Trajectory schema for offline dynamics data.

A trajectory is the metadata of one episode (``env_id``, ``seed``,
``horizon``) plus the transitions collected from it. Models reject unknown
fields. JSON is the in-memory document. JSONL is one metadata line followed
by one transition per line, for a single trajectory file. A later dataset
day can concatenate those files. This module does not step an environment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_finite(value: float, *, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


class Observation(_StrictModel):
    """Environment state at one time.

    ``values`` is the state vector. For the default environment that is
    ``(prey, predator)``. ``t`` is simulation time in the environment's units.
    """

    values: list[float] = Field(min_length=1)
    t: float = Field(ge=0)

    @field_validator("values")
    @classmethod
    def _values_are_finite(cls, values: list[float]) -> list[float]:
        for value in values:
            _require_finite(value, label="observation")
        return values

    @field_validator("t")
    @classmethod
    def _time_is_finite(cls, value: float) -> float:
        return _require_finite(value, label="t")


class Action(_StrictModel):
    """Control applied on one transition.

    For the default environment, ``values`` is ``(prey_dose, predator_dose)``.
    """

    values: list[float] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _values_are_finite(cls, values: list[float]) -> list[float]:
        for value in values:
            _require_finite(value, label="action")
        return values


class Transition(_StrictModel):
    """One controlled step: state, action, reward, and next state."""

    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool

    @field_validator("reward")
    @classmethod
    def _reward_is_finite(cls, value: float) -> float:
        return _require_finite(value, label="reward")


class Trajectory(_StrictModel):
    """One episode.

    ``env_id``, ``seed``, and ``horizon`` are the episode metadata.
    ``horizon`` is the number of steps the collector asked for. The transition
    list is shorter when the environment terminates or truncates first.
    """

    env_id: str = Field(min_length=1)
    seed: int
    horizon: int = Field(ge=1)
    transitions: list[Transition]

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize this trajectory as a JSON document."""

        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> Trajectory:
        """Parse a JSON document produced by :meth:`to_json`."""

        return cls.model_validate_json(data)

    def to_jsonl(self) -> str:
        """Serialize as a metadata line plus one transition line each.

        The result ends with a newline. Blank lines are not written.
        """

        header = {
            "record": "meta",
            "env_id": self.env_id,
            "seed": self.seed,
            "horizon": self.horizon,
        }
        lines = [json.dumps(header, separators=(",", ":"))]
        for transition in self.transitions:
            payload = {"record": "transition", **transition.model_dump(mode="json")}
            lines.append(json.dumps(payload, separators=(",", ":")))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, data: str) -> Trajectory:
        """Parse one trajectory file written by :meth:`to_jsonl`."""

        lines = [line for line in data.splitlines() if line.strip()]
        if not lines:
            raise ValueError("trajectory JSONL is empty")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"trajectory JSONL metadata is not valid JSON: {exc}") from exc
        if not isinstance(header, dict):
            raise ValueError("trajectory JSONL metadata must be an object")
        allowed = {"record", "env_id", "seed", "horizon"}
        if set(header) != allowed or header.get("record") != "meta":
            raise ValueError("trajectory JSONL must start with a metadata record")
        transitions: list[dict[str, object]] = []
        for index, line in enumerate(lines[1:], start=2):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"trajectory JSONL line {index} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict) or payload.get("record") != "transition":
                raise ValueError(f"trajectory JSONL line {index} is not a transition record")
            body = {key: value for key, value in payload.items() if key != "record"}
            transitions.append(body)
        return cls.model_validate(
            {
                "env_id": header["env_id"],
                "seed": header["seed"],
                "horizon": header["horizon"],
                "transitions": transitions,
            }
        )

    def write_json(self, path: str | Path) -> None:
        """Write :meth:`to_json` to ``path`` as UTF-8 text."""

        text = self.to_json()
        if not text.endswith("\n"):
            text += "\n"
        Path(path).write_text(text, encoding="utf-8")

    def write_jsonl(self, path: str | Path) -> None:
        """Write :meth:`to_jsonl` to ``path`` as UTF-8 text."""

        Path(path).write_text(self.to_jsonl(), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> Trajectory:
        """Read a trajectory JSON file."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def read_jsonl(cls, path: str | Path) -> Trajectory:
        """Read a trajectory JSONL file."""

        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))
