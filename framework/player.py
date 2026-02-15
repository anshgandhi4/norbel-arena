"""Agent interface (formerly Player) used by the match runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .events import MatchEvent
from .move import Move
from .observation import Observation
from .result import MatchResult


class Agent(ABC):
    """Base interface for autonomous, scripted, or human-controlled players."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def reset(
        self,
        game_id: str,
        player_id: str,
        role: str | None,
        seed: int,
        config: dict[str, Any] | None,
    ) -> None:
        """Reset internal state before a new match."""

    @abstractmethod
    def act(self, observation: Observation, legal_moves_spec: Any) -> Move:
        """Return the next move given an observation and legal move information."""

    def on_illegal_move(self, error: Exception, observation: Observation) -> None:
        """Optional callback for illegal move feedback."""

    def on_game_end(self, result: MatchResult, history: Sequence[MatchEvent]) -> None:
        """Optional callback invoked when the match ends."""

    def debug_context(self) -> Mapping[str, Any] | None:
        """Optional diagnostics payload for logging around errors/illegal moves."""
        return None
