"""Utilities for deterministic JSON serialization and hashing."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_serializable(value: Any) -> Any:
    """Convert Python objects into JSON-serializable primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {key: to_serializable(field_value) for key, field_value in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_serializable(field_value) for key, field_value in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_serializable(value.to_dict())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    """Serialize any supported value to a deterministic JSON string."""
    separators = (",", ":") if indent is None else None
    return json.dumps(
        to_serializable(value),
        sort_keys=True,
        ensure_ascii=True,
        separators=separators,
        indent=indent,
    )


def digest(value: Any) -> str:
    """Return a SHA256 digest of deterministic JSON encoding."""
    encoded = json_dumps(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
