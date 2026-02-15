"""Marker agent used for human-controlled seats."""

from __future__ import annotations

from typing import Any

from ..errors import AgentExecutionError
from ..move import Move
from ..player import Agent


class HumanAgent(Agent):
    """
    Placeholder agent for human-controlled seats.

    This agent should not be called directly in automated runners; it exists so
    seat configuration can explicitly declare a human participant.
    """

    def act(self, observation: Any, legal_moves_spec: Any) -> Move:
        raise AgentExecutionError(self.agent_id, "HumanAgent requires external move input.")
