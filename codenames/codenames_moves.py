"""Move definitions for Codenames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from framework.move import Move


class MoveType(str, Enum):
    """Supported move discriminators."""

    GIVE_CLUE = "GiveClue"
    GUESS = "Guess"
    END_TURN = "EndTurn"
    RESIGN = "Resign"


@dataclass(frozen=True)
class GiveClue(Move):
    """Spymaster move providing a clue and guess count."""

    clue: str
    count: int
    move_type = MoveType.GIVE_CLUE.value

    def __post_init__(self) -> None:
        clue = self.clue.strip()
        if not clue:
            raise ValueError("Clue must be non-empty.")
        if self.count < 0:
            raise ValueError("Clue count must be >= 0.")
        object.__setattr__(self, "clue", clue)


@dataclass(frozen=True)
class Guess(Move):
    """Operative guess for a board index."""

    index: int
    move_type = MoveType.GUESS.value

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Guess index must be >= 0.")


@dataclass(frozen=True)
class EndTurn(Move):
    """Operative move to stop guessing early."""

    move_type = MoveType.END_TURN.value


@dataclass(frozen=True)
class Resign(Move):
    """Any-seat move to resign the game."""

    move_type = MoveType.RESIGN.value


def move_from_dict(data: Mapping[str, Any]) -> Move:
    """Parse a Codenames move from JSON payload."""
    move_type = data.get("type") or data.get("move_type")
    if move_type == MoveType.GIVE_CLUE.value:
        return GiveClue.from_dict(data)
    if move_type == MoveType.GUESS.value:
        if "index" not in data and "word_index" in data:
            translated = dict(data)
            translated["index"] = translated["word_index"]
            return Guess.from_dict(translated)
        return Guess.from_dict(data)
    if move_type == MoveType.END_TURN.value:
        return EndTurn()
    if move_type == MoveType.RESIGN.value:
        return Resign()
    raise ValueError(f"Unknown Codenames move type: {move_type!r}")
