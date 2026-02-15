"""Pydantic request/response schemas for the match API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PlayerType = Literal["human", "random"]


class CreateMatchRequest(BaseModel):
    """Request body for creating a new match session."""

    seed: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    players: dict[str, PlayerType | dict[str, Any]] = Field(default_factory=dict)
    human_player_id: str | None = None


class SubmitMoveRequest(BaseModel):
    """Request body for submitting a move."""

    player_id: str
    move: dict[str, Any]
