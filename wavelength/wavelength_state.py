"""State and enums for the Wavelength game."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from framework.state import State

PSYCHIC_PLAYER_ID = "PSYCHIC"
GUESSER_ONE_PLAYER_ID = "GUESSER_ONE"
GUESSER_TWO_PLAYER_ID = "GUESSER_TWO"
GUESSER_PLAYER_IDS: tuple[str, str] = (GUESSER_ONE_PLAYER_ID, GUESSER_TWO_PLAYER_ID)


class Role(str, Enum):
    """Wavelength roles."""

    PSYCHIC = "PSYCHIC"
    GUESSER = "GUESSER"


class Phase(str, Enum):
    """Wavelength turn phases."""

    CHOOSE_NUMBER = "CHOOSE_NUMBER"
    CATEGORY_PROMPT = "CATEGORY_PROMPT"
    PSYCHIC_RESPONSE = "PSYCHIC_RESPONSE"
    NUMERIC_GUESS = "NUMERIC_GUESS"
    FINAL_ESTIMATE = "FINAL_ESTIMATE"


PLAYER_ROLES: dict[str, Role] = {
    PSYCHIC_PLAYER_ID: Role.PSYCHIC,
    GUESSER_ONE_PLAYER_ID: Role.GUESSER,
    GUESSER_TWO_PLAYER_ID: Role.GUESSER,
}


@dataclass(frozen=True)
class WavelengthState(State):
    """Immutable Wavelength state."""

    seed: int
    max_rounds: int
    phase: Phase
    active_player: str
    active_guesser: str | None
    target_number: int | None
    pending_exchange: dict[str, Any] | None
    history: tuple[dict[str, Any], ...]
    guess_counts: dict[str, int]
    final_estimates: dict[str, int | None]
    winner: str | None = None
    termination_reason: str | None = None
    turn_index: int = 0
    last_move: dict[str, Any] | None = None

    def current_round(self) -> int:
        """Return the 1-based round number currently in play."""
        highest = max(self.guess_counts.values(), default=0)
        return min(highest + 1, self.max_rounds)

    def guess_error(self, player_id: str) -> int | None:
        """Return absolute error for final estimate when available."""
        if self.target_number is None:
            return None
        estimate = self.final_estimates.get(player_id)
        if estimate is None:
            return None
        return abs(estimate - self.target_number)


def role_for_player(player_id: str) -> Role:
    """Return canonical role for a player ID."""
    if player_id not in PLAYER_ROLES:
        raise ValueError(f"Unknown Wavelength player_id: {player_id!r}")
    return PLAYER_ROLES[player_id]
