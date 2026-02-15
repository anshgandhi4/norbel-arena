"""Random baseline agent."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Any, Mapping

from ..errors import AgentExecutionError
from ..move import Move
from ..player import Agent


class RandomAgent(Agent):
    """Chooses uniformly from enumerable legal moves."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)
        self._rng = random.Random()

    def reset(
        self,
        game_id: str,
        player_id: str,
        role: str | None,
        seed: int,
        config: dict[str, Any] | None,
    ) -> None:
        """Reset deterministic RNG state per match and seat."""
        material = f"{seed}:{game_id}:{self.agent_id}:{player_id}".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big", signed=False)
        self._rng.seed(derived_seed)

    def act(self, observation: Any, legal_moves_spec: Any) -> Move:
        """Pick a random legal move."""
        if isinstance(legal_moves_spec, Sequence) and not isinstance(legal_moves_spec, (str, bytes)):
            options = list(legal_moves_spec)
            if not options:
                raise AgentExecutionError(self.agent_id, "No legal moves available.")
            return self._rng.choice(options)

        if isinstance(legal_moves_spec, Mapping):
            sample_moves = legal_moves_spec.get("sample_moves")
            if isinstance(sample_moves, Sequence) and not isinstance(sample_moves, (str, bytes)):
                options = [move for move in sample_moves if isinstance(move, Move)]
                if options:
                    return self._rng.choice(options)
            raise AgentExecutionError(self.agent_id, "Non-enumerable legal move spec lacks Move samples.")

        raise AgentExecutionError(self.agent_id, "Unsupported legal move specification format.")
