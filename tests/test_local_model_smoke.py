"""Opt-in live smoke test for local model JSON move responses.

This test is skipped by default. Run it explicitly when debugging local model output:

RUN_LOCAL_MODEL_SMOKE=1 \
LOCAL_SMOKE_MODEL='meta-llama/Llama-3.1-8B-Instruct' \
LOCAL_SMOKE_BACKEND='transformers' \
.venv/bin/pytest -q tests/test_local_model_smoke.py -s
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from codenames.codenames_game import CodenamesGame
from framework.serialize import to_serializable
from server.agent_factory import create_agent_for_player


def _enabled() -> bool:
    value = os.getenv("RUN_LOCAL_MODEL_SMOKE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw.strip())


def _build_config(role: str) -> dict[str, Any]:
    role_prefix = role.upper()
    player_type = os.getenv(f"LOCAL_SMOKE_{role_prefix}_TYPE", os.getenv("LOCAL_SMOKE_TYPE", "local")).strip().lower()
    model = os.getenv(
        f"LOCAL_SMOKE_{role_prefix}_MODEL",
        os.getenv("LOCAL_SMOKE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    ).strip()
    backend = os.getenv(
        f"LOCAL_SMOKE_{role_prefix}_BACKEND",
        os.getenv("LOCAL_SMOKE_BACKEND", "transformers"),
    ).strip()
    base_url = os.getenv(
        f"LOCAL_SMOKE_{role_prefix}_BASE_URL",
        os.getenv("LOCAL_SMOKE_BASE_URL"),
    )
    system_prompt = os.getenv(
        f"LOCAL_SMOKE_{role_prefix}_SYSTEM_PROMPT",
        os.getenv("LOCAL_SMOKE_SYSTEM_PROMPT"),
    )
    max_retries = _env_int("LOCAL_SMOKE_MAX_RETRIES", 2)
    temperature = _env_float("LOCAL_SMOKE_TEMPERATURE", 0.0)
    max_tokens_raw = os.getenv(
        f"LOCAL_SMOKE_{role_prefix}_MAX_TOKENS",
        os.getenv("LOCAL_SMOKE_MAX_TOKENS"),
    )

    config: dict[str, Any] = {
        "type": player_type,
        "model": model,
        "max_retries": max_retries,
        "temperature": temperature,
    }
    if max_tokens_raw is not None and max_tokens_raw.strip():
        config["max_tokens"] = int(max_tokens_raw.strip())
    if player_type in {"local", "nemotron"}:
        config["backend"] = backend
        if base_url:
            config["base_url"] = base_url
    if system_prompt:
        config["system_prompt"] = system_prompt
    return config


def _debug_excerpt(agent: Any) -> str:
    if not hasattr(agent, "debug_context") or not callable(agent.debug_context):
        return "debug_context unavailable"
    try:
        context = agent.debug_context()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"debug_context error: {exc}"
    if context is None:
        return "debug_context is None"
    payload = to_serializable(context)
    if not isinstance(payload, dict):
        return repr(payload)
    excerpt = {
        "agent_id": payload.get("agent_id"),
        "system_prompt": payload.get("system_prompt"),
        "parse_errors": payload.get("parse_errors"),
        "selected_move_payload": payload.get("selected_move_payload"),
        "raw_responses": payload.get("raw_responses"),
    }
    return json.dumps(excerpt, indent=2, sort_keys=True, ensure_ascii=True)


def _act_or_fail(*, game: CodenamesGame, state: Any, player_id: str, agent: Any) -> Any:
    observation = game.observation(state, player_id)
    legal_spec = game.legal_moves(state, player_id)
    try:
        move = agent.act(observation, legal_spec)
    except Exception as exc:
        pytest.fail(
            "Local smoke pipeline failed before producing a move.\n"
            f"player_id={player_id}\n"
            f"error={exc!r}\n"
            f"debug_context=\n{_debug_excerpt(agent)}"
        )

    legal, reason = game.is_legal(state, player_id, move)
    if not legal:
        move_payload = move.to_dict() if hasattr(move, "to_dict") else to_serializable(move)
        pytest.fail(
            "Local smoke pipeline produced an illegal move.\n"
            f"player_id={player_id}\n"
            f"move={json.dumps(move_payload, sort_keys=True)}\n"
            f"reason={reason}\n"
            f"debug_context=\n{_debug_excerpt(agent)}"
        )
    return move


def test_local_model_pipeline_smoke_two_phase_json_contract() -> None:
    """
    Smoke-test the same path used in live matches:
    create_agent_for_player -> LocalLLMAgent -> LLMAgent JSON parse/repair -> Codenames parse+legality.
    """
    if not _enabled():
        pytest.skip("Set RUN_LOCAL_MODEL_SMOKE=1 to run live local model smoke test.")

    seed = _env_int("LOCAL_SMOKE_SEED", 67)
    starting_team = os.getenv("LOCAL_SMOKE_STARTING_TEAM", "RED").strip().upper()
    if starting_team not in {"RED", "BLUE"}:
        pytest.fail(f"Invalid LOCAL_SMOKE_STARTING_TEAM={starting_team!r}. Use RED or BLUE.")

    game = CodenamesGame()
    state = game.new_game(seed=seed, config={"starting_team": starting_team})

    team = state.turn_team.value
    spymaster_player_id = f"{team}_SPYMASTER"
    operative_player_id = f"{team}_OPERATIVE"

    spy_config = _build_config("SPYMASTER")
    spymaster_agent = create_agent_for_player(
        player_id=spymaster_player_id,
        config=spy_config,
        move_parser=lambda payload, g=game: g.parse_move(payload),
    )
    spymaster_agent.reset(
        "local-smoke",
        spymaster_player_id,
        game.role_for_player(state, spymaster_player_id),
        seed,
        {"starting_team": starting_team},
    )
    spymaster_move = _act_or_fail(
        game=game,
        state=state,
        player_id=spymaster_player_id,
        agent=spymaster_agent,
    )
    state_after_clue = game.apply_move(state, spymaster_player_id, spymaster_move)

    operative_config = _build_config("OPERATIVE")
    operative_agent = create_agent_for_player(
        player_id=operative_player_id,
        config=operative_config,
        move_parser=lambda payload, g=game: g.parse_move(payload),
    )
    operative_agent.reset(
        "local-smoke",
        operative_player_id,
        game.role_for_player(state_after_clue, operative_player_id),
        seed,
        {"starting_team": starting_team},
    )
    operative_move = _act_or_fail(
        game=game,
        state=state_after_clue,
        player_id=operative_player_id,
        agent=operative_agent,
    )
    state_after_guess = game.apply_move(state_after_clue, operative_player_id, operative_move)

    assert state_after_guess.turn_index >= 2
