"""Validation helpers for filesystem and subprocess boundaries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union


PathLike = Union[str, Path]
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def reject_control_characters(value: str, field: str) -> str:
    """Reject control characters that can corrupt commands, paths, or logs."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} contains control characters")
    return value


def validate_filename(
    value: str,
    *,
    suffix: Optional[str] = None,
    field: str = "filename",
) -> str:
    """Validate a single filename without directory components."""
    reject_control_characters(value, field)
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field} must not contain directory components")
    if not SAFE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field} contains unsupported characters")
    if suffix and not value.endswith(suffix):
        raise ValueError(f"{field} must end with {suffix}")
    return value


def validate_identifier(value: str, *, field: str = "identifier") -> str:
    """Validate identifiers used to select repository resources."""
    reject_control_characters(value, field)
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def resolve_within(
    root: PathLike,
    candidate: PathLike,
    *,
    must_exist: bool = False,
    file_only: bool = False,
    directory_only: bool = False,
) -> Path:
    """Resolve a path and require it to remain beneath a trusted root."""
    trusted_root = Path(root).expanduser().resolve()
    raw_candidate = Path(candidate).expanduser()
    resolved = (
        raw_candidate.resolve()
        if raw_candidate.is_absolute()
        else (trusted_root / raw_candidate).resolve()
    )
    try:
        resolved.relative_to(trusted_root)
    except ValueError as error:
        raise ValueError(f"Path escapes trusted directory {trusted_root}: {candidate}") from error

    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    if file_only and not resolved.is_file():
        raise ValueError(f"Expected a file: {resolved}")
    if directory_only and not resolved.is_dir():
        raise ValueError(f"Expected a directory: {resolved}")
    return resolved


def resolve_within_any(
    roots: Iterable[PathLike],
    candidate: PathLike,
    **kwargs,
) -> Path:
    """Resolve a path beneath one of several trusted roots."""
    errors = []
    for root in roots:
        try:
            return resolve_within(root, candidate, **kwargs)
        except (FileNotFoundError, ValueError) as error:
            errors.append(error)
    raise ValueError(f"Path is outside all trusted directories: {candidate}") from errors[-1]


def validate_cli_path(value: str, *, field: str) -> Path:
    """Validate CLI path syntax before enforcing its trusted directory."""
    reject_control_characters(value, field)
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{field} must not contain '..'")
    return path


def sanitize_log_value(value: object) -> str:
    """Keep one logical line per externally influenced log value."""
    return str(value).replace("\r", r"\r").replace("\n", r"\n")
