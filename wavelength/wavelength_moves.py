"""Move definitions for Wavelength."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from framework.move import Move


class MoveType(str, Enum):
    """Supported Wavelength move discriminators."""

    CHOOSE_NUMBER = "ChooseNumber"
    ASK_CATEGORY = "AskCategory"
    GIVE_ANSWER = "GiveAnswer"
    SUBMIT_GUESS = "SubmitGuess"
    SUBMIT_FINAL_ESTIMATE = "SubmitFinalEstimate"


@dataclass(frozen=True)
class ChooseNumber(Move):
    """Psychic chooses the hidden target number."""

    value: int
    move_type = MoveType.CHOOSE_NUMBER.value

    def __post_init__(self) -> None:
        if self.value < 1 or self.value > 100:
            raise ValueError("ChooseNumber.value must be between 1 and 100.")


@dataclass(frozen=True)
class AskCategory(Move):
    """Guesser asks for a subjective category scale."""

    category: str
    move_type = MoveType.ASK_CATEGORY.value

    def __post_init__(self) -> None:
        normalized = self.category.strip()
        if not normalized:
            raise ValueError("AskCategory.category must be non-empty.")
        object.__setattr__(self, "category", normalized)


@dataclass(frozen=True)
class GiveAnswer(Move):
    """Psychic provides a subjective anchor item for the category."""

    answer: str
    move_type = MoveType.GIVE_ANSWER.value

    def __post_init__(self) -> None:
        normalized = self.answer.strip()
        if not normalized:
            raise ValueError("GiveAnswer.answer must be non-empty.")
        object.__setattr__(self, "answer", normalized)


@dataclass(frozen=True)
class SubmitGuess(Move):
    """Guesser submits a numeric probe guess."""

    value: int
    move_type = MoveType.SUBMIT_GUESS.value

    def __post_init__(self) -> None:
        if self.value < 1 or self.value > 100:
            raise ValueError("SubmitGuess.value must be between 1 and 100.")


@dataclass(frozen=True)
class SubmitFinalEstimate(Move):
    """Guesser submits their final estimate for the hidden number."""

    value: int
    move_type = MoveType.SUBMIT_FINAL_ESTIMATE.value

    def __post_init__(self) -> None:
        if self.value < 1 or self.value > 100:
            raise ValueError("SubmitFinalEstimate.value must be between 1 and 100.")


def move_from_dict(data: Mapping[str, Any]) -> Move:
    """Parse a Wavelength move from JSON payload."""
    move_type = data.get("type") or data.get("move_type")
    if move_type == MoveType.CHOOSE_NUMBER.value:
        return ChooseNumber.from_dict(data)
    if move_type == MoveType.ASK_CATEGORY.value:
        if "category" not in data and "prompt" in data:
            translated = dict(data)
            translated["category"] = translated["prompt"]
            return AskCategory.from_dict(translated)
        return AskCategory.from_dict(data)
    if move_type == MoveType.GIVE_ANSWER.value:
        return GiveAnswer.from_dict(data)
    if move_type == MoveType.SUBMIT_GUESS.value:
        if "value" not in data and "guess" in data:
            translated = dict(data)
            translated["value"] = translated["guess"]
            return SubmitGuess.from_dict(translated)
        return SubmitGuess.from_dict(data)
    if move_type == MoveType.SUBMIT_FINAL_ESTIMATE.value:
        if "value" not in data and "estimate" in data:
            translated = dict(data)
            translated["value"] = translated["estimate"]
            return SubmitFinalEstimate.from_dict(translated)
        return SubmitFinalEstimate.from_dict(data)
    raise ValueError(f"Unknown Wavelength move type: {move_type!r}")
