"""Observation model for Codenames partial observability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from framework.observation import Observation

from .codenames_state import CardType, Phase, Role, Team


@dataclass(frozen=True)
class CodenamesObservation(Observation):
    """Player-specific Codenames view."""

    player_id: str
    team: Team
    role: Role
    board_words: tuple[str, ...]
    revealed: tuple[bool, ...]
    revealed_colors: tuple[CardType | None, ...]
    turn_team: Team
    phase: Phase
    current_clue: tuple[str, int] | None
    guesses_remaining: int
    remaining_targets: dict[str, int]
    turn_index: int
    last_move: dict[str, Any] | None
    assignments: tuple[CardType, ...] | None = None
