"""Event schema and JSONL logging utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Iterable, Mapping

from .serialize import json_dumps, to_serializable


class EventType(str, Enum):
    """Standard event types emitted by the runner."""

    MATCH_START = "match_start"
    TURN = "turn"
    ILLEGAL_MOVE = "illegal_move"
    AGENT_ERROR = "agent_error"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class MatchEvent:
    """Single replay event emitted during match execution."""

    event_type: EventType
    game_id: str
    turn: int
    timestamp_ms: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable event data."""
        return {
            "event_type": self.event_type.value,
            "game_id": self.game_id,
            "turn": self.turn,
            "timestamp_ms": self.timestamp_ms,
            "payload": to_serializable(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MatchEvent":
        """Build an event from a dictionary payload."""
        return cls(
            event_type=EventType(str(data["event_type"])),
            game_id=str(data["game_id"]),
            turn=int(data["turn"]),
            timestamp_ms=int(data["timestamp_ms"]),
            payload=dict(data.get("payload", {})),
        )

    @classmethod
    def create(cls, event_type: EventType, game_id: str, turn: int, payload: dict[str, Any]) -> "MatchEvent":
        """Construct an event with the current wall-clock timestamp."""
        return cls(
            event_type=event_type,
            game_id=game_id,
            turn=turn,
            timestamp_ms=int(time() * 1000),
            payload=payload,
        )


def write_jsonl(path: str | Path, events: Iterable[MatchEvent]) -> None:
    """Persist events as JSONL to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json_dumps(event.to_dict()))
            handle.write("\n")
