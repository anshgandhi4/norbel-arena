"""Structured exceptions used across the arena framework."""

from __future__ import annotations

from typing import Any


class ArenaError(Exception):
    """Base class for framework-level exceptions."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable error payload."""
        return {"type": self.__class__.__name__, "message": str(self)}


class MatchConfigurationError(ArenaError):
    """Raised when a match is configured incorrectly."""


class IllegalMoveError(ArenaError):
    """Raised when an agent proposes an illegal move."""

    def __init__(self, player_id: str, move: Any, reason: str | None = None):
        self.player_id = player_id
        self.move = move
        self.reason = reason
        message = f"Illegal move by {player_id}"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update({"player_id": self.player_id, "move": getattr(self.move, "to_dict", lambda: self.move)()})
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


class AgentExecutionError(ArenaError):
    """Raised when an agent fails to produce a move."""

    def __init__(self, player_id: str, message: str):
        self.player_id = player_id
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["player_id"] = self.player_id
        return payload


class AgentTimeoutError(AgentExecutionError):
    """Raised when an agent exceeds the configured move-time limit."""
