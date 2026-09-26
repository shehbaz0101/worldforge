"""Filesystem sandbox for user-supplied corpus, checkpoint, and output paths.

The allowed root is ``WORLDFORGE_DATA_ROOT`` when that variable is set,
and the current working directory otherwise. ``worldforge`` commands
that take ``--data-root`` set the variable for that process. Relative
paths resolve against the root. Absolute paths are accepted only when
they resolve inside it. ``..`` and symlinks are resolved before the
check, so a path that leaves the root is rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "WORLDFORGE_DATA_ROOT"


class PathSandboxError(ValueError):
    """A user path resolved outside the allowed root."""


def configure_data_root(value: str | Path) -> Path:
    """Resolve ``value`` to an existing directory and return that path."""

    raw = str(value).strip()
    if not raw:
        raise PathSandboxError("data root must not be empty")
    if "\x00" in raw:
        raise PathSandboxError("data root contains a null byte")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise PathSandboxError(f"data root could not be resolved: {raw}") from exc
    if not resolved.is_dir():
        raise PathSandboxError(f"data root does not exist or is not a directory: {resolved}")
    return resolved


def data_root() -> Path:
    """Return the allowed root for this process.

    An unset or blank ``WORLDFORGE_DATA_ROOT`` means the current directory.
    """

    raw = os.environ.get(DATA_ROOT_ENV)
    if raw is None or not raw.strip():
        return Path.cwd().resolve()
    return configure_data_root(raw)


def resolve_user_path(value: str | Path, *, label: str, root: Path | None = None) -> Path:
    """Resolve ``value`` and require it to stay inside ``root``.

    ``root`` defaults to :func:`data_root`. The returned path is absolute.
    The target does not have to exist yet, so an output path can be created
    after the check. A symlink that points outside the root is rejected.
    """

    raw = str(value).strip()
    if not raw:
        raise PathSandboxError(f"{label} must not be empty")
    if "\x00" in raw:
        raise PathSandboxError(f"{label} contains a null byte")
    base = data_root() if root is None else root.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise PathSandboxError(f"{label} could not be resolved: {raw}") from exc
    if not _is_inside(resolved, base):
        raise PathSandboxError(
            f"{label} escapes the data root {base} "
            f"(path traversal or absolute path outside the sandbox): {raw}"
        )
    return resolved


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
