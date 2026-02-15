"""Observation model for Wavelength partial observability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from framework.observation import Observation

from .wavelength_state import Phase, Role


@dataclass(frozen=True)
class WavelengthObservation(Observation):
    """Player-specific Wavelength view."""

    player_id: str
    role: Role
    phase: Phase
    current_player: str
    current_round: int
    max_rounds: int
    turn_index: int
    target_number: int | None
    history: tuple[dict[str, Any], ...]
    pending_exchange: dict[str, Any] | None
    guess_counts: dict[str, int]
    final_estimates: dict[str, int | None]
    winner: str | None
    termination_reason: str | None
    last_move: dict[str, Any] | None
