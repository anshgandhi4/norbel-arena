"""Match result models and termination metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Self

from .serialize import to_serializable


class TerminationReason(str, Enum):
    """Standardized reasons for match termination."""

    NORMAL_WIN = "normal_win"
    ILLEGAL_MOVE_FORFEIT = "illegal_move_forfeit"
    TIMEOUT_FORFEIT = "timeout_forfeit"
    MAX_TURNS = "max_turns"
    AGENT_EXCEPTION = "agent_exception"
    DRAW = "draw"


@dataclass(frozen=True)
class MatchResult:
    """Structured outcome for a completed match."""

    game_id: str
    game_name: str
    seed: int
    winner: str | None
    termination_reason: TerminationReason
    scores: dict[str, float] = field(default_factory=dict)
    turns: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    details: str | None = None
    final_state_digest: str | None = None
    event_count: int = 0
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result object."""
        return {
            "game_id": self.game_id,
            "game_name": self.game_name,
            "seed": self.seed,
            "winner": self.winner,
            "termination_reason": self.termination_reason.value,
            "scores": to_serializable(self.scores),
            "turns": self.turns,
            "stats": to_serializable(self.stats),
            "details": self.details,
            "final_state_digest": self.final_state_digest,
            "event_count": self.event_count,
            "log_path": self.log_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build result from serialized data."""
        return cls(
            game_id=str(data["game_id"]),
            game_name=str(data["game_name"]),
            seed=int(data["seed"]),
            winner=data.get("winner"),
            termination_reason=TerminationReason(str(data["termination_reason"])),
            scores=dict(data.get("scores", {})),
            turns=int(data.get("turns", 0)),
            stats=dict(data.get("stats", {})),
            details=data.get("details"),
            final_state_digest=data.get("final_state_digest"),
            event_count=int(data.get("event_count", 0)),
            log_path=data.get("log_path"),
        )
