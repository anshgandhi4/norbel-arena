"""Provider-agnostic LLM agent skeleton with strict JSON move output."""

from __future__ import annotations

import json
import random
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
        self._last_debug_context: dict[str, Any] | None = None

    def reset(
        self,
        game_id: str,
        player_id: str,
        role: str | None,
        seed: int,
        config: dict[str, Any] | None,
    ) -> None:
        """Clear per-match debug context."""
        self._last_debug_context = None

    def debug_context(self) -> Mapping[str, Any] | None:
        """Return diagnostics for the most recent action."""
        if self._last_debug_context is None:
            return None
        return dict(self._last_debug_context)

    def act(self, observation: Any, legal_moves_spec: Any) -> Move:
        """Prompt the model and parse strict JSON into a move."""
        prompt = self._build_prompt(observation, legal_moves_spec)
        context: dict[str, Any] = {
            "agent_id": self.agent_id,
            "system_prompt": self.system_prompt,
            "initial_prompt": prompt,
            "repair_prompts": [],
            "raw_responses": [],
            "parse_errors": [],
        }
        self._last_debug_context = context
        raw: str | None = None
        try:
            raw = self.llm_client.complete(prompt, system_prompt=self.system_prompt)
            context["raw_responses"].append(raw)
        except Exception as exc:
            last_error = f"LLM request failed: {exc}"
            context["parse_errors"].append({"attempt": 1, "error": last_error})
            fallback_move = self._codenames_guess_fallback_move(
                observation=observation,
                legal_moves_spec=legal_moves_spec,
                error=last_error,
                context=context,
            )
            if fallback_move is not None:
                return fallback_move
            raise AgentExecutionError(self.agent_id, f"LLM could not produce a valid move JSON: {last_error}") from exc

        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                if raw is None:
                    raise ValueError("LLM returned no response.")
                payload = self._extract_json_object(raw)
                context["selected_move_payload"] = payload
                return self.move_parser(payload)
            except Exception as exc:
                last_error = str(exc)
                context["parse_errors"].append({"attempt": attempt + 1, "error": last_error})
                if attempt >= self.max_retries:
                    break
                repair_prompt = self._build_repair_prompt(raw_response=raw, error=last_error)
                context["repair_prompts"].append(repair_prompt)
                try:
                    raw = self.llm_client.complete(repair_prompt, system_prompt=self.system_prompt)
                    context["raw_responses"].append(raw)
                except Exception as repair_exc:
                    last_error = f"LLM repair request failed: {repair_exc}"
                    context["parse_errors"].append({"attempt": attempt + 2, "error": last_error})
                    break

        fallback_move = self._codenames_guess_fallback_move(
            observation=observation,
            legal_moves_spec=legal_moves_spec,
            error=last_error,
            context=context,
        )
        if fallback_move is not None:
            return fallback_move
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

    def _codenames_guess_fallback_move(
        self,
        *,
        observation: Any,
        legal_moves_spec: Any,
        error: str | None,
        context: dict[str, Any],
    ) -> Move | None:
        """
        For Codenames operative turns, fall back to a legal random Guess when the LLM fails.

        This prevents immediate forfeits caused purely by malformed model output.
        """
        phase = getattr(observation, "phase", None)
        phase_value = getattr(phase, "value", phase)
        if str(phase_value) != "OPERATIVE_GUESSING":
            return None
        if not hasattr(observation, "board_words") or not hasattr(observation, "revealed"):
            return None

        guess_indices = self._extract_guess_indices(legal_moves_spec)
        if not guess_indices:
            revealed = getattr(observation, "revealed", ())
            if isinstance(revealed, (list, tuple)):
                guess_indices = [index for index, is_revealed in enumerate(revealed) if not is_revealed]
        if not guess_indices:
            return None

        selected_index = random.choice(sorted(set(guess_indices)))
        fallback_payload = {"type": "Guess", "index": selected_index}
        context["fallback_move_payload"] = fallback_payload
        context["fallback_reason"] = error
        context["selected_move_payload"] = fallback_payload
        try:
            return self.move_parser(fallback_payload)
        except Exception as exc:
            context["fallback_error"] = str(exc)
            return None

    def _extract_guess_indices(self, legal_moves_spec: Any) -> list[int]:
        indices: list[int] = []
        if isinstance(legal_moves_spec, (list, tuple)):
            for move in legal_moves_spec:
                payload: Any
                if hasattr(move, "to_dict"):
                    payload = move.to_dict()
                else:
                    payload = move
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") != "Guess":
                    continue
                raw_index = payload.get("index")
                if isinstance(raw_index, int):
                    indices.append(raw_index)
            return indices

        if isinstance(legal_moves_spec, dict):
            allowed = legal_moves_spec.get("allowed")
            if isinstance(allowed, dict):
                guess = allowed.get("Guess")
                if isinstance(guess, dict):
                    raw_indices = guess.get("indices")
                    if isinstance(raw_indices, list):
                        for raw_index in raw_indices:
                            if isinstance(raw_index, int):
                                indices.append(raw_index)
            if indices:
                return indices

            sample_moves = legal_moves_spec.get("sample_moves")
            if isinstance(sample_moves, list):
                for move in sample_moves:
                    payload: Any
                    if hasattr(move, "to_dict"):
                        payload = move.to_dict()
                    else:
                        payload = move
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("type") != "Guess":
                        continue
                    raw_index = payload.get("index")
                    if isinstance(raw_index, int):
                        indices.append(raw_index)
        return indices
