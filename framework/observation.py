"""Observation objects delivered to agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Self

from .serialize import digest, to_serializable


@dataclass(frozen=True)
class Observation:
    """Base observation with deterministic serialization helpers."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return to_serializable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build an observation from serialized data."""
        return cls(**data)  # type: ignore[misc]

    def observation_digest(self) -> str:
        """Return a deterministic digest for event logs."""
        return digest(self.to_dict())
