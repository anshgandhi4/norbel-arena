"""Unit tests for LLMAgent JSON parsing and fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codenames.codenames_game import CodenamesGame
from codenames.codenames_moves import GiveClue, Guess
from framework.agents.llm_agent import LLMAgent
from framework.errors import AgentExecutionError


@dataclass
class _StaticResponseClient:
    """LLM client stub that returns predefined responses in order."""

    responses: list[str]

    def __post_init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:  # noqa: ARG002
        self.calls += 1
        if self.calls <= len(self.responses):
            return self.responses[self.calls - 1]
        return self.responses[-1]


@dataclass
class _FailingClient:
    """LLM client stub that always raises."""

    message: str = "provider error"

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:  # noqa: ARG002
        raise RuntimeError(self.message)


def _operative_turn_state() -> tuple[CodenamesGame, object]:
    game = CodenamesGame()
    state = game.new_game(seed=67, config={"starting_team": "RED"})
    state = game.apply_move(state, "RED_SPYMASTER", GiveClue(clue="alpha", count=1))
    return game, state


def test_llm_agent_falls_back_to_legal_guess_after_parse_retries_exhausted() -> None:
    game, state = _operative_turn_state()
    observation = game.observation(state, "RED_OPERATIVE")
    legal_spec = game.legal_moves(state, "RED_OPERATIVE")
    client = _StaticResponseClient(responses=["not-json", "still not-json", "also not-json"])
    agent = LLMAgent(
        agent_id="llm-red-operative",
        llm_client=client,
        move_parser=game.parse_move,
        max_retries=2,
    )

    move = agent.act(observation, legal_spec)
    assert isinstance(move, Guess)
    legal, reason = game.is_legal(state, "RED_OPERATIVE", move)
    assert legal, reason

    context = agent.debug_context()
    assert isinstance(context, dict)
    assert context.get("fallback_move_payload", {}).get("type") == "Guess"
    assert client.calls == 3


def test_llm_agent_falls_back_to_legal_guess_when_provider_call_fails() -> None:
    game, state = _operative_turn_state()
    observation = game.observation(state, "RED_OPERATIVE")
    legal_spec = game.legal_moves(state, "RED_OPERATIVE")
    agent = LLMAgent(
        agent_id="llm-red-operative",
        llm_client=_FailingClient(message="request timeout"),
        move_parser=game.parse_move,
        max_retries=1,
    )

    move = agent.act(observation, legal_spec)
    assert isinstance(move, Guess)
    legal, reason = game.is_legal(state, "RED_OPERATIVE", move)
    assert legal, reason


def test_llm_agent_does_not_guess_fallback_for_non_operative_phase() -> None:
    game = CodenamesGame()
    state = game.new_game(seed=67, config={"starting_team": "RED"})
    observation = game.observation(state, "RED_SPYMASTER")
    legal_spec = game.legal_moves(state, "RED_SPYMASTER")
    agent = LLMAgent(
        agent_id="llm-red-spymaster",
        llm_client=_StaticResponseClient(responses=["not-json", "still not-json"]),
        move_parser=game.parse_move,
        max_retries=1,
    )

    with pytest.raises(AgentExecutionError):
        agent.act(observation, legal_spec)
