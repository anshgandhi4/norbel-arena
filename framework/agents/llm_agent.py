"""Provider-agnostic LLM agent skeleton with strict JSON move output."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Protocol

from ..errors import AgentExecutionError
from ..move import Move
from ..player import Agent
from ..serialize import json_dumps


class LLMClient(Protocol):
    """Minimal protocol for LLM API adapters."""

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Return a model response for a prompt."""


class StubLLMClient:
    """Stub client that intentionally raises until replaced."""

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise NotImplementedError("StubLLMClient has no backend. Provide a concrete LLMClient implementation.")


class LLMAgent(Agent):
    """LLM-backed agent that emits structured JSON moves."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        move_parser: Callable[[Mapping[str, Any]], Move],
        *,
        system_prompt: str | None = None,
        max_retries: int = 2,
    ):
        super().__init__(agent_id=agent_id)
        self.llm_client = llm_client
        self.move_parser = move_parser
        self.system_prompt = system_prompt
        self.max_retries = max_retries

    def act(self, observation: Any, legal_moves_spec: Any) -> Move:
        """Prompt the model and parse strict JSON into a move."""
        prompt = self._build_prompt(observation, legal_moves_spec)
        raw = self.llm_client.complete(prompt, system_prompt=self.system_prompt)
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                payload = self._extract_json_object(raw)
                return self.move_parser(payload)
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.max_retries:
                    break
                raw = self.llm_client.complete(
                    self._build_repair_prompt(raw_response=raw, error=last_error),
                    system_prompt=self.system_prompt,
                )

        raise AgentExecutionError(self.agent_id, f"LLM could not produce a valid move JSON: {last_error}")

    def _build_prompt(self, observation: Any, legal_moves_spec: Any) -> str:
        observation_payload = observation.to_dict() if hasattr(observation, "to_dict") else observation
        legal_payload = self._serialize_legal_moves(legal_moves_spec)
        return (
            "You are an autonomous game agent.\n"
            "Output exactly one JSON object representing a move and nothing else.\n"
            "Do not use markdown fences.\n\n"
            "Observation:\n"
            f"{json_dumps(observation_payload, indent=2)}\n\n"
            "Legal move specification:\n"
            f"{json_dumps(legal_payload, indent=2)}\n\n"
            "Required output schema examples:\n"
            '{ "type": "GiveClue", "clue": "animal", "count": 2 }\n'
            '{ "type": "Guess", "index": 13 }\n'
            '{ "type": "EndTurn" }\n'
        )

    def _build_repair_prompt(self, *, raw_response: str, error: str) -> str:
        return (
            "Repair the response into valid JSON for exactly one move.\n"
            "Return only the JSON object, no markdown.\n"
            f"Error: {error}\n"
            f"Original response:\n{raw_response}\n"
        )

    def _extract_json_object(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("No JSON object found in LLM output.")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("LLM output JSON must be an object.")
        return parsed

    def _serialize_legal_moves(self, legal_moves_spec: Any) -> Any:
        if isinstance(legal_moves_spec, list):
            return [
                move.to_dict() if hasattr(move, "to_dict") else move
                for move in legal_moves_spec
            ]
        if isinstance(legal_moves_spec, tuple):
            return [
                move.to_dict() if hasattr(move, "to_dict") else move
                for move in legal_moves_spec
            ]
        if isinstance(legal_moves_spec, dict):
            serialized = dict(legal_moves_spec)
            sample_moves = serialized.get("sample_moves")
            if isinstance(sample_moves, list):
                serialized["sample_moves"] = [
                    move.to_dict() if hasattr(move, "to_dict") else move for move in sample_moves
                ]
            return serialized
        return legal_moves_spec
