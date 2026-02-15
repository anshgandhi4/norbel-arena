"""Core game interface for deterministic state-based games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Sequence, TypeVar

from .move import Move
from .observation import Observation
from .result import MatchResult

PlayerId = str
StateT = TypeVar("StateT")
MoveT = TypeVar("MoveT", bound=Move)
ObservationT = TypeVar("ObservationT", bound=Observation)
LegalMovesSpec = Sequence[MoveT] | Mapping[str, Any]


class Game(ABC, Generic[StateT, MoveT, ObservationT]):
    """Abstract interface that every game implementation must satisfy."""

    game_name: str = "game"

    @abstractmethod
    def new_game(self, seed: int, config: dict[str, Any] | None = None) -> StateT:
        """Create a fresh state for a seeded match."""

    @abstractmethod
    def player_ids(self, state: StateT) -> Sequence[PlayerId]:
        """Return all player IDs expected for the state."""

    @abstractmethod
    def role_for_player(self, state: StateT, player_id: PlayerId) -> str | None:
        """Return a role label for a player (for prompts/metadata)."""

    @abstractmethod
    def current_player(self, state: StateT) -> PlayerId:
        """Return the player ID whose turn it is."""

    @abstractmethod
    def legal_moves(self, state: StateT, player_id: PlayerId) -> LegalMovesSpec:
        """Return legal moves or a legal-move specification for a player."""

    @abstractmethod
    def is_legal(self, state: StateT, player_id: PlayerId, move: MoveT) -> tuple[bool, str | None]:
        """Return whether a move is legal and an optional reason when illegal."""

    @abstractmethod
    def apply_move(self, state: StateT, player_id: PlayerId, move: MoveT) -> StateT:
        """Apply a legal move and return the next state."""

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool:
        """Return whether the state is terminal."""

    @abstractmethod
    def outcome(self, state: StateT) -> MatchResult:
        """Return a structured match result for a terminal state."""

    @abstractmethod
    def observation(self, state: StateT, player_id: PlayerId) -> ObservationT:
        """Return a player-specific observation (partial view)."""

    @abstractmethod
    def render(self, state: StateT, player_id: PlayerId | None = None) -> str:
        """Render the state for debugging and replay tooling."""

    def parse_move(self, data: Mapping[str, Any]) -> MoveT:
        """Parse a move payload produced by external agents."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement parse_move().")

    def forfeit_winner(self, state: StateT, offending_player_id: PlayerId, reason: str) -> str | None:
        """Return winner ID for runner-enforced forfeits. Default: no winner."""
        return None
