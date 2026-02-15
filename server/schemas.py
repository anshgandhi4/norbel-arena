"""Pydantic request/response schemas for the match API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateMatchRequest(BaseModel):
    """Request body for creating a new match session."""

    game: str = "codenames"
    seed: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    players: dict[str, str | dict[str, Any]] = Field(default_factory=dict)
    human_player_id: str | None = None
    viewer_player_id: str | None = None


class SubmitMoveRequest(BaseModel):
    """Request body for submitting a move."""

    player_id: str
    move: dict[str, Any]
