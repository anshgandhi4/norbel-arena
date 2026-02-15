"""Optional CLI-driven human controller."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..errors import AgentExecutionError
from ..move import Move
from ..player import Agent
from ..serialize import json_dumps


class HumanCLIController(Agent):
    """Prompts for move selection in a terminal."""

    def act(self, observation: Any, legal_moves_spec: Any) -> Move:
        """Select a move by index when legal moves are enumerable."""
        print("\n=== Observation ===")
        print(json_dumps(observation.to_dict() if hasattr(observation, "to_dict") else observation, indent=2))

        if isinstance(legal_moves_spec, Sequence) and not isinstance(legal_moves_spec, (str, bytes)):
            options = list(legal_moves_spec)
            if not options:
                raise AgentExecutionError(self.agent_id, "No legal moves available.")
            print("\n=== Legal Moves ===")
            for index, move in enumerate(options):
                payload = move.to_dict() if hasattr(move, "to_dict") else move
                print(f"[{index}] {json_dumps(payload)}")
            while True:
                raw = input("Choose move index: ").strip()
                try:
                    choice = int(raw)
                except ValueError:
                    print("Expected an integer index.")
                    continue
                if 0 <= choice < len(options):
                    return options[choice]
                print("Index out of range.")

        raise AgentExecutionError(
            self.agent_id,
            "HumanCLIController only supports enumerable legal move lists.",
        )
