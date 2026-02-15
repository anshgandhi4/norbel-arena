"""Base move abstractions used by all games."""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, is_dataclass
from typing import Any, ClassVar, Mapping, Self

from .serialize import to_serializable


class Move(ABC):
    """Base class for a typed command emitted by an agent."""

    move_type: ClassVar[str] = "Move"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the move."""
        if is_dataclass(self):
            payload = {key: to_serializable(value) for key, value in asdict(self).items()}
        else:
            payload = {
                key: to_serializable(value)
                for key, value in vars(self).items()
                if not key.startswith("_")
            }
        payload["type"] = self.move_type
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build the move from a dictionary payload."""
        kwargs = {key: value for key, value in data.items() if key not in {"type", "move_type"}}
        return cls(**kwargs)  # type: ignore[misc, call-arg]
