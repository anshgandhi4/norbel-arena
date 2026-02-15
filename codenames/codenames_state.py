"""State and enums for Codenames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from framework.state import State


class Team(str, Enum):
    """Codenames teams."""

    RED = "RED"
    BLUE = "BLUE"


class Role(str, Enum):
    """Team roles."""

    SPYMASTER = "SPYMASTER"
    OPERATIVE = "OPERATIVE"


class Phase(str, Enum):
    """Turn phases."""

    SPYMASTER_CLUE = "SPYMASTER_CLUE"
    OPERATIVE_GUESSING = "OPERATIVE_GUESSING"


class CardType(str, Enum):
    """Assignment of each board card."""

    RED = "RED"
    BLUE = "BLUE"
    NEUTRAL = "NEUTRAL"
    ASSASSIN = "ASSASSIN"


TEAM_PLAYER_IDS: dict[tuple[Team, str], str] = {
    (Team.RED, "SPYMASTER"): "RED_SPYMASTER",
    (Team.RED, "OPERATIVE"): "RED_OPERATIVE",
    (Team.BLUE, "SPYMASTER"): "BLUE_SPYMASTER",
    (Team.BLUE, "OPERATIVE"): "BLUE_OPERATIVE",
}

PLAYER_TO_TEAM_ROLE: dict[str, tuple[Team, Role]] = {
    player_id: (team, Role(role))
    for (team, role), player_id in TEAM_PLAYER_IDS.items()
}


def player_for(team: Team, role: Role | str) -> str:
    """Return the canonical player ID for a team/role seat."""
    role_name = role.value if isinstance(role, Role) else role
    return TEAM_PLAYER_IDS[(team, role_name)]


def team_role_for_player(player_id: str) -> tuple[Team, Role]:
    """Return team + role for player ID."""
    if player_id not in PLAYER_TO_TEAM_ROLE:
        raise ValueError(f"Unknown Codenames player_id: {player_id!r}")
    return PLAYER_TO_TEAM_ROLE[player_id]


@dataclass(frozen=True)
class CodenamesState(State):
    """Immutable Codenames state."""

    seed: int
    board_words: tuple[str, ...]
    assignments: tuple[CardType, ...]
    revealed: tuple[bool, ...]
    turn_team: Team
    phase: Phase
    current_clue: tuple[str, int] | None
    guesses_remaining: int
    winner: Team | None = None
    termination_reason: str | None = None
    turn_index: int = 0
    last_move: dict[str, Any] | None = None
    red_target_count: int = 0
    blue_target_count: int = 0

    def team_words_remaining(self, team: Team) -> int:
        """Count unrevealed words assigned to `team`."""
        target = CardType.RED if team is Team.RED else CardType.BLUE
        return sum(
            1
            for assignment, revealed in zip(self.assignments, self.revealed, strict=True)
            if assignment is target and not revealed
        )

    def revealed_counts(self) -> dict[str, int]:
        """Return number of revealed cards for each assignment type."""
        counts = {
            CardType.RED.value: 0,
            CardType.BLUE.value: 0,
            CardType.NEUTRAL.value: 0,
            CardType.ASSASSIN.value: 0,
        }
        for assignment, revealed in zip(self.assignments, self.revealed, strict=True):
            if revealed:
                counts[assignment.value] += 1
        return counts
