"""State conventions for immutable, serializable game states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Self

from .serialize import digest, to_serializable


@dataclass(frozen=True)
class State:
    """Base immutable state object with serialization helpers."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return to_serializable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a state instance from serialized data."""
        return cls(**data)  # type: ignore[misc]

    def state_digest(self) -> str:
        """Return a deterministic digest for logging/replay."""
        return digest(self.to_dict())
