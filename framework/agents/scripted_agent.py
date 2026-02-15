"""Scripted agent scaffold."""

from __future__ import annotations

from typing import Any, Callable

from ..move import Move
from ..player import Agent


class ScriptedAgent(Agent):
    """Runs a user-provided policy callable."""

    def __init__(self, agent_id: str, policy: Callable[[Any, Any], Move] | None = None):
        super().__init__(agent_id=agent_id)
        self.policy = policy

    def act(self, observation: Any, legal_moves_spec: Any) -> Move:
        """Delegate to the configured scripted policy."""
        if self.policy is None:
            raise NotImplementedError("ScriptedAgent requires a policy(observation, legal_moves_spec) callable.")
        return self.policy(observation, legal_moves_spec)
