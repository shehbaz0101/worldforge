"""Build, load, split, and summarize offline trajectory corpora.

A corpus is a directory:

* ``manifest.json`` — collection parameters and one record per episode
  (seed, policy, step count, optional episode file). Format id
  ``worldforge.corpus.v1``.
* ``corpus.jsonl`` — one :class:`~worldforge.schemas.Trajectory` JSON object
  per line, when the ``jsonl`` format is requested. This is a corpus file.
  It is not the single-episode JSONL from :meth:`Trajectory.to_jsonl`, which
  starts with a ``record: meta`` line and then one transition per line.
* ``episodes/ep_XXXX.json`` — one trajectory JSON document per episode, when
  the ``json`` format is requested. Same document as :meth:`Trajectory.to_json`.

``load_corpus`` accepts that directory, a corpus JSONL file, a single-episode
JSONL file, or one trajectory JSON file. ``split_trajectories`` assigns
episodes to train, val, and test with counts from the largest-remainder
method and membership from ``random.Random(seed)`` after a canonical sort.
``sample_transition_batch`` draws a deterministic uniform replay batch from
stored transitions. Nothing in this module uses the network or trains a model.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worldforge.envs import DEFAULT_ENV_ID, make_env
from worldforge.rollout import POLICIES, Policy, rollout
from worldforge.schemas import Trajectory

FormatName = Literal["jsonl", "json"]
FORMATS: tuple[FormatName, ...] = ("jsonl", "json")
CORPUS_FORMAT = "worldforge.corpus.v1"
DEFAULT_N_EPISODES = 4
DEFAULT_HORIZON = 8
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
_MANIFEST_NAME = "manifest.json"
_CORPUS_NAME = "corpus.jsonl"
_EPISODE_DIR = "episodes"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EpisodeRecord(_StrictModel):
    """One collected episode, as recorded in the corpus manifest."""

    index: int = Field(ge=0)
    seed: int
    policy: Policy
    steps: int = Field(ge=0)
    file: str | None = None


class CorpusManifest(_StrictModel):
    """Parameters and episode index for one ``collect_corpus`` directory."""

    format: Literal["worldforge.corpus.v1"] = CORPUS_FORMAT
    env_id: str = Field(min_length=1)
    horizon: int = Field(ge=1)
    n_episodes: int = Field(ge=1)
    policy: Literal["zero", "random", "sine_dose", "mixed"]
    episodes: list[EpisodeRecord] = Field(min_length=1)


@dataclass(frozen=True)
class EpisodeSpec:
    """Seed and open-loop policy for one episode.

    When ``collect_corpus`` receives ``specs``, each spec is rolled as written.
    That replaces ``n_episodes``, ``seed``, ``seeds``, and ``policy``.
    """

    seed: int
    policy: Policy = "random"


@dataclass(frozen=True)
class TrajectorySplit:
    """Deterministic train, val, and test partitions of one corpus."""

    train: tuple[Trajectory, ...]
    val: tuple[Trajectory, ...]
    test: tuple[Trajectory, ...]
    seed: int

    @property
    def counts(self) -> tuple[int, int, int]:
        return (len(self.train), len(self.val), len(self.test))


@dataclass(frozen=True)
class TransitionBatch:
    """A uniform sample of stored transitions, in sample order.

    This is a replay batch for a later trainer. It does not step the
    environment and it does not update any parameters.
    """

    observations: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    rewards: tuple[float, ...]
    next_observations: tuple[tuple[float, ...], ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    seeds: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass(frozen=True)
class DatasetInfo:
    """Summary of a corpus directory or a trajectory file."""

    path: str
    kind: str
    n_episodes: int
    n_transitions: int
    env_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    seeds: tuple[int, ...]
    policies: tuple[str, ...]
    n_terminated: int
    n_truncated: int


def collect_corpus(
    output_dir: str | Path,
    *,
    n_episodes: int | None = None,
    horizon: int = DEFAULT_HORIZON,
    seed: int = 0,
    seeds: Sequence[int] | None = None,
    policy: Policy = "random",
    env_id: str = DEFAULT_ENV_ID,
    formats: Sequence[FormatName] = FORMATS,
    specs: Sequence[EpisodeSpec] | None = None,
) -> CorpusManifest:
    """Roll ``env_id`` and write a corpus directory.

    With no ``specs`` and no ``seeds``, episode ``i`` uses ``seed + i`` and
    ``policy``. ``n_episodes`` defaults to 4 in that case. ``seeds`` lists the
    episode seeds explicitly and must agree with ``n_episodes`` when both are
    set. ``formats`` chooses ``jsonl`` (``corpus.jsonl``), ``json`` (one file
    per episode), or both. The manifest is always written.
    """

    resolved = _resolve_specs(
        n_episodes=n_episodes,
        seed=seed,
        seeds=seeds,
        policy=policy,
        specs=specs,
    )
    horizon = _require_int(horizon, label="horizon", minimum=1)
    env_id = _require_env_id(env_id)
    selected = _require_formats(formats)
    directory = _require_output_dir(output_dir)

    env = make_env(env_id, horizon=horizon)
    episodes = [
        rollout(env, n_steps=horizon, seed=spec.seed, policy=spec.policy) for spec in resolved
    ]
    return _write_corpus(
        directory,
        episodes,
        resolved,
        horizon=horizon,
        env_id=env_id,
        formats=selected,
    )


def load_corpus(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a corpus directory or a trajectory file.

    A directory prefers ``corpus.jsonl``. Without that file, episode paths in
    ``manifest.json`` are used, then ``episodes/*.json``. A ``.jsonl`` file
    that starts with ``record: meta`` is one episode. Any other ``.jsonl`` file
    is a corpus: one trajectory object per line. A ``.json`` file is one episode.
    """

    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(location)
    if location.is_file():
        return _load_file(location)
    if not location.is_dir():
        raise ValueError(f"path is not a file or directory: {location}")

    corpus_path = location / _CORPUS_NAME
    if corpus_path.is_file():
        return _read_jsonl(corpus_path)

    manifest_path = location / _MANIFEST_NAME
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        files = [_episode_file(location, record) for record in _sorted_records(manifest)]
        if files and all(file is not None for file in files):
            return [Trajectory.read_json(file) for file in files if file is not None]

    episode_dir = location / _EPISODE_DIR
    json_files = sorted(episode_dir.glob("*.json")) if episode_dir.is_dir() else []
    if not json_files:
        json_files = sorted(
            file for file in location.glob("*.json") if file.name != _MANIFEST_NAME
        )
    if not json_files:
        raise ValueError(f"no trajectories in {location}")
    return [Trajectory.read_json(file) for file in json_files]


def load_manifest(path: str | Path) -> CorpusManifest:
    """Read ``manifest.json``, or ``manifest.json`` inside a corpus directory."""

    location = Path(path)
    if location.is_dir():
        location = location / _MANIFEST_NAME
    if not location.is_file():
        raise FileNotFoundError(location)
    try:
        return CorpusManifest.model_validate_json(location.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"{location} is not a corpus manifest: {exc}") from exc


def split_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> TrajectorySplit:
    """Split episodes into train, val, and test.

    Counts use the largest-remainder method on ``ratios`` (train, val, test),
    which must be non-negative and sum to 1. Ties in the remainder go to the
    earlier split. Membership shuffles a canonical order — ``env_id``, seed,
    horizon, then the trajectory JSON — with ``random.Random(seed)``. The same
    episodes and the same seed produce the same split. Input order does not.
    A small corpus can leave val or test empty when its ratio rounds to zero;
    pass explicit ratios when every split must be non-empty.
    """

    if len(trajectories) < 1:
        raise ValueError("trajectories must be non-empty")
    seed = _require_int(seed, label="seed")
    train_n, val_n, _test_n = _split_sizes(len(trajectories), ratios)
    ordered = sorted(trajectories, key=_trajectory_sort_key)
    random.Random(seed).shuffle(ordered)
    return TrajectorySplit(
        train=tuple(ordered[:train_n]),
        val=tuple(ordered[train_n : train_n + val_n]),
        test=tuple(ordered[train_n + val_n :]),
        seed=seed,
    )


def sample_transition_batch(
    trajectories: Sequence[Trajectory],
    *,
    batch_size: int,
    seed: int,
) -> TransitionBatch:
    """Sample ``batch_size`` stored transitions without replacement.

    The pool is every transition of ``trajectories`` in the order given.
    ``random.Random(seed)`` shuffles that pool. The batch keeps sample order.
    ``batch_size`` cannot exceed the pool.
    """

    batch_size = _require_int(batch_size, label="batch_size", minimum=1)
    seed = _require_int(seed, label="seed")
    pool = [
        (trajectory.seed, transition)
        for trajectory in trajectories
        for transition in trajectory.transitions
    ]
    if batch_size > len(pool):
        raise ValueError(f"batch_size {batch_size} exceeds {len(pool)} transitions")
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    observations: list[tuple[float, ...]] = []
    actions: list[tuple[float, ...]] = []
    rewards: list[float] = []
    next_observations: list[tuple[float, ...]] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    seeds: list[int] = []
    for index in order[:batch_size]:
        episode_seed, transition = pool[index]
        observations.append(tuple(transition.observation.values))
        actions.append(tuple(transition.action.values))
        rewards.append(transition.reward)
        next_observations.append(tuple(transition.next_observation.values))
        terminated.append(transition.terminated)
        truncated.append(transition.truncated)
        seeds.append(episode_seed)
    return TransitionBatch(
        observations=tuple(observations),
        actions=tuple(actions),
        rewards=tuple(rewards),
        next_observations=tuple(next_observations),
        terminated=tuple(terminated),
        truncated=tuple(truncated),
        seeds=tuple(seeds),
    )


def inspect_dataset(path: str | Path) -> DatasetInfo:
    """Summarize a corpus directory or a trajectory file."""

    location = Path(path)
    episodes = load_corpus(location)
    if len(episodes) < 1:
        raise ValueError(f"no trajectories in {location}")
    kind = _kind(location)
    policies = _policies_for(location, episodes)
    env_ids: list[str] = []
    for episode in episodes:
        if episode.env_id not in env_ids:
            env_ids.append(episode.env_id)
    horizons = tuple(sorted({episode.horizon for episode in episodes}))
    terminated = 0
    truncated = 0
    n_transitions = 0
    for episode in episodes:
        n_transitions += len(episode.transitions)
        if episode.transitions and episode.transitions[-1].terminated:
            terminated += 1
        if episode.transitions and episode.transitions[-1].truncated:
            truncated += 1
    return DatasetInfo(
        path=str(location),
        kind=kind,
        n_episodes=len(episodes),
        n_transitions=n_transitions,
        env_ids=tuple(env_ids),
        horizons=horizons,
        seeds=tuple(episode.seed for episode in episodes),
        policies=policies,
        n_terminated=terminated,
        n_truncated=truncated,
    )


def format_dataset_info(info: DatasetInfo) -> str:
    """Plain-text summary for ``worldforge dataset-info`` and ``collect``."""

    policies = " ".join(info.policies) if info.policies else "unknown"
    lines = [
        f"path: {info.path}",
        f"kind: {info.kind}",
        f"episodes: {info.n_episodes}",
        f"transitions: {info.n_transitions}",
        f"env_ids: {_format_tokens(info.env_ids)}",
        f"horizons: {_format_ints(info.horizons)}",
        f"seeds: {_format_ints(info.seeds)}",
        f"policies: {policies}",
        f"terminated_episodes: {info.n_terminated}",
        f"truncated_episodes: {info.n_truncated}",
    ]
    return "\n".join(lines)


def _resolve_specs(
    *,
    n_episodes: int | None,
    seed: int,
    seeds: Sequence[int] | None,
    policy: Policy,
    specs: Sequence[EpisodeSpec] | None,
) -> list[EpisodeSpec]:
    if specs is not None:
        if len(specs) < 1:
            raise ValueError("specs must be non-empty")
        resolved = [_check_spec(item) for item in specs]
        return resolved
    if policy not in POLICIES:
        names = ", ".join(POLICIES)
        raise ValueError(f"policy must be one of: {names}")
    if seeds is not None:
        resolved_seeds = [_require_int(item, label="seed") for item in seeds]
        if len(resolved_seeds) < 1:
            raise ValueError("seeds must be non-empty")
        if n_episodes is not None and n_episodes != len(resolved_seeds):
            raise ValueError(
                f"n_episodes ({n_episodes}) does not match the number of seeds "
                f"({len(resolved_seeds)})"
            )
        return [EpisodeSpec(seed=item, policy=policy) for item in resolved_seeds]
    count = DEFAULT_N_EPISODES if n_episodes is None else n_episodes
    count = _require_int(count, label="n_episodes", minimum=1)
    base = _require_int(seed, label="seed")
    return [EpisodeSpec(seed=base + index, policy=policy) for index in range(count)]


def _check_spec(spec: EpisodeSpec) -> EpisodeSpec:
    if not isinstance(spec, EpisodeSpec):
        raise TypeError("specs must contain EpisodeSpec values")
    seed = _require_int(spec.seed, label="seed")
    if spec.policy not in POLICIES:
        names = ", ".join(POLICIES)
        raise ValueError(f"policy must be one of: {names}")
    return EpisodeSpec(seed=seed, policy=spec.policy)


def _write_corpus(
    directory: Path,
    episodes: Sequence[Trajectory],
    specs: Sequence[EpisodeSpec],
    *,
    horizon: int,
    env_id: str,
    formats: tuple[FormatName, ...],
) -> CorpusManifest:
    if len(episodes) != len(specs):
        raise RuntimeError("episode count does not match the spec list")
    write_json = "json" in formats
    _clear_unselected_outputs(directory, formats=formats, n_episodes=len(episodes))
    records: list[EpisodeRecord] = []
    if write_json:
        (directory / _EPISODE_DIR).mkdir(parents=True, exist_ok=True)
    for index, (episode, spec) in enumerate(zip(episodes, specs, strict=True)):
        relative: str | None = None
        if write_json:
            relative = f"{_EPISODE_DIR}/ep_{index:04d}.json"
            episode.write_json(directory / relative)
        records.append(
            EpisodeRecord(
                index=index,
                seed=spec.seed,
                policy=spec.policy,
                steps=len(episode.transitions),
                file=relative,
            )
        )
    if "jsonl" in formats:
        _write_corpus_jsonl(directory / _CORPUS_NAME, episodes)
    policies = {spec.policy for spec in specs}
    manifest = CorpusManifest(
        env_id=env_id,
        horizon=horizon,
        n_episodes=len(records),
        policy=specs[0].policy if len(policies) == 1 else "mixed",
        episodes=records,
    )
    _write_manifest(directory / _MANIFEST_NAME, manifest)
    return manifest


def _clear_unselected_outputs(
    directory: Path,
    *,
    formats: tuple[FormatName, ...],
    n_episodes: int,
) -> None:
    """Drop managed files this write will not replace.

    A later ``collect_corpus`` into the same directory should not leave a stale
    ``corpus.jsonl`` that :func:`load_corpus` would prefer over new episode JSON,
    or extra ``ep_*.json`` files from a longer previous run.
    """

    if "jsonl" not in formats:
        corpus_path = directory / _CORPUS_NAME
        if corpus_path.is_file():
            corpus_path.unlink()
    episode_dir = directory / _EPISODE_DIR
    if not episode_dir.is_dir():
        return
    for file in episode_dir.glob("ep_*.json"):
        if "json" not in formats:
            file.unlink()
            continue
        suffix = file.stem.removeprefix("ep_")
        if suffix.isdecimal() and int(suffix) >= n_episodes:
            file.unlink()


def _write_corpus_jsonl(path: Path, episodes: Sequence[Trajectory]) -> None:
    lines = [
        json.dumps(episode.model_dump(mode="json"), separators=(",", ":"), allow_nan=False)
        for episode in episodes
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path: Path, manifest: CorpusManifest) -> None:
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2)
    path.write_text(payload + "\n", encoding="utf-8")


def _load_file(path: Path) -> list[Trajectory]:
    if path.name == _MANIFEST_NAME or path.suffix == ".json":
        if path.name == _MANIFEST_NAME:
            raise ValueError(f"{path} is a manifest, not a trajectory file")
        return [Trajectory.read_json(path)]
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    raise ValueError(f"unsupported trajectory file: {path.name}")


def _read_jsonl(path: Path) -> list[Trajectory]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    try:
        first = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} line 1 is not valid JSON: {exc}") from exc
    if isinstance(first, dict) and first.get("record") == "meta":
        return [Trajectory.read_jsonl(path)]
    episodes: list[Trajectory] = []
    for index, line in enumerate(lines, start=1):
        try:
            episodes.append(Trajectory.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"{path} line {index} is not a trajectory: {exc}") from exc
    return episodes


def _episode_file(root: Path, record: EpisodeRecord) -> Path | None:
    if record.file is None:
        return None
    relative = Path(record.file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"episode path escapes the corpus directory: {record.file}")
    file_path = (root / relative).resolve()
    if not file_path.is_relative_to(root.resolve()):
        raise ValueError(f"episode path escapes the corpus directory: {record.file}")
    return file_path


def _sorted_records(manifest: CorpusManifest) -> list[EpisodeRecord]:
    return sorted(manifest.episodes, key=lambda record: record.index)


def _kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.suffix == ".json":
        return "trajectory-json"
    if path.suffix == ".jsonl":
        first_line = next(
            (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "",
        )
        try:
            first = json.loads(first_line) if first_line else None
        except json.JSONDecodeError:
            first = None
        if isinstance(first, dict) and first.get("record") == "meta":
            return "trajectory-jsonl"
        return "corpus-jsonl"
    raise ValueError(f"unsupported trajectory file: {path.name}")


def _policies_for(path: Path, episodes: Sequence[Trajectory]) -> tuple[str, ...]:
    manifest_path = _manifest_near(path)
    if manifest_path is None:
        return ()
    try:
        manifest = load_manifest(manifest_path)
    except (FileNotFoundError, ValueError):
        return ()
    root = manifest_path.parent
    if path.is_dir() or path.name == _CORPUS_NAME:
        if manifest.n_episodes != len(episodes):
            return ()
        return tuple(dict.fromkeys(record.policy for record in _sorted_records(manifest)))
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return ()
    matched = [record.policy for record in manifest.episodes if record.file == relative]
    return tuple(dict.fromkeys(matched))


def _manifest_near(path: Path) -> Path | None:
    if path.is_dir():
        candidate = path / _MANIFEST_NAME
        return candidate if candidate.is_file() else None
    candidate = path.parent / _MANIFEST_NAME
    if candidate.is_file():
        return candidate
    return None


def _split_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    _validate_ratios(ratios)
    raw = [n * float(ratio) for ratio in ratios]
    sizes = [int(math.floor(value + 1e-12)) for value in raw]
    leftover = n - sum(sizes)
    if leftover < 0:
        raise ValueError("split sizes exceeded the corpus")
    order = sorted(range(3), key=lambda index: (-(raw[index] - sizes[index]), index))
    for index in order[:leftover]:
        sizes[index] += 1
    return sizes[0], sizes[1], sizes[2]


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise ValueError("ratios must be (train, val, test)")
    for ratio in ratios:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("ratios must be real numbers")
        if not math.isfinite(float(ratio)) or float(ratio) < 0.0:
            raise ValueError("ratios must be finite and >= 0")
    if abs(sum(float(ratio) for ratio in ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")


def _trajectory_sort_key(trajectory: Trajectory) -> tuple[str, int, int, str]:
    return (
        trajectory.env_id,
        trajectory.seed,
        trajectory.horizon,
        trajectory.model_dump_json(),
    )


def _require_formats(formats: Sequence[FormatName]) -> tuple[FormatName, ...]:
    if len(formats) < 1:
        raise ValueError("formats must include jsonl, json, or both")
    selected: list[FormatName] = []
    for name in formats:
        if name not in FORMATS:
            raise ValueError("formats must include jsonl, json, or both")
        if name not in selected:
            selected.append(name)
    return tuple(selected)


def _require_output_dir(path: str | Path) -> Path:
    directory = Path(path)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"output path is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _require_env_id(env_id: object) -> str:
    if not isinstance(env_id, str) or not env_id:
        raise ValueError("env_id must be a non-empty string")
    return env_id


def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _format_tokens(values: Sequence[str]) -> str:
    return " ".join(values) if values else "none"


def _format_ints(values: Sequence[int]) -> str:
    if not values:
        return "none"
    if len(values) <= 16:
        return " ".join(str(value) for value in values)
    head = " ".join(str(value) for value in values[:8])
    return f"{head} ... ({len(values)} values)"
