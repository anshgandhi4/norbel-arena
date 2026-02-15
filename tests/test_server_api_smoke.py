"""Smoke tests for the local FastAPI match API."""

from __future__ import annotations

from fastapi import HTTPException

from codenames.codenames_moves import Guess
from framework.player import Agent
import server.main as main_module
from server.main import get_events, get_observation, get_report_cards, new_match, submit_move
from server.schemas import CreateMatchRequest, SubmitMoveRequest
import server.session as session_module


class _BadPromptAgent(Agent):
    """Deterministic agent that emits an illegal move with prompt diagnostics."""

    def act(self, observation, legal_moves_spec):  # type: ignore[override]
        return Guess(index=0)

    def debug_context(self) -> dict[str, str]:
        return {
            "system_prompt": "You are a test agent.",
            "initial_prompt": "Return an illegal move for test coverage.",
        }


class _FirstLegalAgent(Agent):
    """Deterministic agent for API tests that always picks the first legal move."""

    def act(self, observation, legal_moves_spec):  # type: ignore[override]
        if isinstance(legal_moves_spec, list) and legal_moves_spec:
            return legal_moves_spec[0]
        if isinstance(legal_moves_spec, dict):
            sample_moves = legal_moves_spec.get("sample_moves")
            if isinstance(sample_moves, list) and sample_moves:
                return sample_moves[0]
        raise RuntimeError("No legal move available for deterministic test agent.")


class _FirstLegalDebugResponseAgent(_FirstLegalAgent):
    """Deterministic legal agent that exposes synthetic model raw responses."""

    def debug_context(self) -> dict[str, object]:
        return {
            "system_prompt": "You are a deterministic debug agent.",
            "raw_responses": ['{"type":"GiveClue","clue":"alpha","count":1}'],
            "parse_errors": [],
        }


def _patch_network_agents_with_deterministic_stub(monkeypatch) -> None:
    original_factory = session_module.create_agent_for_player

    def _patched_factory(*, player_id, config, move_parser):
        player_type = str(config.get("type", "")).lower()
        if player_type == "badprompt":
            return _BadPromptAgent(agent_id=f"badprompt-{player_id.lower()}")
        if player_type in {"openai", "anthropic", "perplexity", "local", "nemotron", "random"}:
            return _FirstLegalAgent(agent_id=f"stub-{player_id.lower()}")
        return original_factory(player_id=player_id, config=config, move_parser=move_parser)

    monkeypatch.setattr(session_module, "create_agent_for_player", _patched_factory)


def _new_match(payload: dict) -> dict:
    return new_match(CreateMatchRequest.model_validate(payload))


def _submit(match_id: str, payload: dict) -> dict:
    return submit_move(match_id, SubmitMoveRequest.model_validate(payload))


def _observe(match_id: str, player_id: str, turn: int | None = None) -> dict:
    return get_observation(match_id=match_id, player_id=player_id, turn=turn)


def _events(match_id: str) -> list[dict]:
    return get_events(match_id=match_id, format="array")


def _expect_http_error(fn, expected_status: int) -> str:
    try:
        fn()
    except HTTPException as exc:
        assert exc.status_code == expected_status
        return str(exc.detail)
    raise AssertionError("Expected HTTPException to be raised.")


def _wavelength_reported_match_count(payload: dict) -> int:
    games = payload.get("games", {})
    if not isinstance(games, dict):
        return 0
    bucket = games.get("wavelength:guesser_eval", {})
    if not isinstance(bucket, dict):
        return 0

    total = 0
    for stats in bucket.values():
        if isinstance(stats, dict):
            total += int(stats.get("matches", 0))
    return total


def test_match_api_flow_create_observe_move_events(monkeypatch) -> None:
    _patch_network_agents_with_deterministic_stub(monkeypatch)
    created = _new_match(
        {
            "seed": 123,
            "config": {"starting_team": "RED", "evaluation_mode": "spymaster_eval"},
            "players": {
                "RED_SPYMASTER": "human",
                "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                "BLUE_SPYMASTER": "random",
                "BLUE_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
            },
            "human_player_id": "RED_SPYMASTER",
        }
    )

    match_id = created["match_id"]
    player_id = created["player_id"]
    initial_turn = created["observation"]["turn_index"]

    observed = _observe(match_id=match_id, player_id=player_id)
    assert observed["match_id"] == match_id
    assert observed["player_id"] == player_id
    assert observed["meta"]["evaluation_mode"] == "spymaster_eval"

    moved = _submit(
        match_id,
        {
            "player_id": player_id,
            "move": {"type": "GiveClue", "clue": "animal", "count": 1},
        },
    )
    assert moved["observation"]["turn_index"] > initial_turn

    payload = _events(match_id)
    assert isinstance(payload, list)
    assert len(payload) >= 2


def test_match_api_uses_time_seed_when_seed_is_omitted(monkeypatch) -> None:
    _patch_network_agents_with_deterministic_stub(monkeypatch)
    fake_time_ns = 1_730_000_000_123_456_789
    monkeypatch.setattr(main_module.time, "time_ns", lambda: fake_time_ns)

    created = _new_match(
        {
            "config": {"starting_team": "RED", "evaluation_mode": "spymaster_eval"},
            "players": {
                "RED_SPYMASTER": "human",
                "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                "BLUE_SPYMASTER": "random",
                "BLUE_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
            },
            "human_player_id": "RED_SPYMASTER",
        }
    )

    expected_seed = int(fake_time_ns & 0x7FFFFFFF) or 1
    assert created["meta"]["seed"] == expected_seed


def test_all_ai_match_replay_and_report_cards(monkeypatch) -> None:
    _patch_network_agents_with_deterministic_stub(monkeypatch)
    created = _new_match(
        {
            "seed": 456,
            "config": {"starting_team": "RED", "board_size": 9, "evaluation_mode": "operator_eval"},
            "players": {
                "RED_SPYMASTER": {"type": "openai", "model": "gpt-4o-mini"},
                "RED_OPERATIVE": {"type": "anthropic", "model": "claude-sonnet-4-20250514"},
                "BLUE_SPYMASTER": {"type": "openai", "model": "gpt-4o-mini"},
                "BLUE_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
            },
            "viewer_player_id": "RED_OPERATIVE",
        }
    )

    match_id = created["match_id"]
    player_id = created["player_id"]
    assert created["meta"]["terminal"] is True
    assert created["meta"]["is_live"] is True
    max_turn = int(created["meta"]["max_turn"])
    assert max_turn >= 1

    replay_start_payload = _observe(match_id=match_id, player_id=player_id, turn=0)
    assert replay_start_payload["meta"]["replay_turn"] == 0
    assert replay_start_payload["meta"]["is_live"] is False

    replay_end_payload = _observe(match_id=match_id, player_id=player_id, turn=max_turn)
    assert replay_end_payload["meta"]["replay_turn"] == max_turn

    report_payload = get_report_cards()
    assert "games" in report_payload
    assert "codenames:operator_eval" in report_payload["games"]
    assert isinstance(report_payload["games"]["codenames:operator_eval"], dict)


def test_terminal_error_logs_prompt_context(monkeypatch) -> None:
    _patch_network_agents_with_deterministic_stub(monkeypatch)

    created = _new_match(
        {
            "seed": 789,
            "config": {"starting_team": "RED", "evaluation_mode": "spymaster_eval"},
            "players": {
                "RED_SPYMASTER": {"type": "badprompt"},
                "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                "BLUE_SPYMASTER": "random",
                "BLUE_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
            },
            "viewer_player_id": "BLUE_OPERATIVE",
        }
    )
    assert created["meta"]["terminal"] is True

    match_id = created["match_id"]
    payload = _events(match_id)

    illegal_events = [event for event in payload if event["event_type"] == "illegal_move"]
    assert illegal_events, payload
    prompt_context = illegal_events[0]["payload"].get("prompt_context")
    assert isinstance(prompt_context, dict)
    assert prompt_context["initial_prompt"] == "Return an illegal move for test coverage."

    terminal_events = [event for event in payload if event["event_type"] == "terminal"]
    assert terminal_events, payload
    terminal_prompt = terminal_events[0]["payload"].get("prompt_context")
    assert isinstance(terminal_prompt, dict)
    assert terminal_prompt["initial_prompt"] == "Return an illegal move for test coverage."


def test_turn_event_logs_model_response_when_agent_provides_raw_output(monkeypatch) -> None:
    original_factory = session_module.create_agent_for_player

    def _patched_factory(*, player_id, config, move_parser):
        player_type = str(config.get("type", "")).lower()
        if player_type == "debugresponse":
            return _FirstLegalDebugResponseAgent(agent_id=f"debugresponse-{player_id.lower()}")
        if player_type in {"openai", "anthropic", "perplexity", "local", "nemotron", "random"}:
            return _FirstLegalAgent(agent_id=f"stub-{player_id.lower()}")
        return original_factory(player_id=player_id, config=config, move_parser=move_parser)

    monkeypatch.setattr(session_module, "create_agent_for_player", _patched_factory)

    created = _new_match(
        {
            "seed": 987,
            "config": {"starting_team": "RED", "evaluation_mode": "spymaster_eval"},
            "players": {
                "RED_SPYMASTER": {"type": "debugresponse"},
                "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                "BLUE_SPYMASTER": "random",
                "BLUE_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
            },
            "viewer_player_id": "RED_OPERATIVE",
        }
    )

    payload = _events(created["match_id"])
    debug_turns = [
        event
        for event in payload
        if event["event_type"] == "turn" and event["payload"].get("player_id") == "RED_SPYMASTER"
    ]
    assert debug_turns, payload
    turn_payload = debug_turns[0]["payload"]
    assert turn_payload.get("model_response") == '{"type":"GiveClue","clue":"alpha","count":1}'
    assert turn_payload.get("model_response_history") == ['{"type":"GiveClue","clue":"alpha","count":1}']


def test_operator_eval_rejects_same_operator_models() -> None:
    message = _expect_http_error(
        lambda: _new_match(
            {
                "seed": 101,
                "config": {"evaluation_mode": "operator_eval"},
                "players": {
                    "RED_SPYMASTER": {"type": "openai", "model": "gpt-4o-mini"},
                    "BLUE_SPYMASTER": {"type": "openai", "model": "gpt-4o-mini"},
                    "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                    "BLUE_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                },
            }
        ),
        expected_status=400,
    )
    assert "Operators must use different models" in message


def test_operator_eval_rejects_non_identical_spymasters() -> None:
    message = _expect_http_error(
        lambda: _new_match(
            {
                "seed": 102,
                "config": {"evaluation_mode": "operator_eval"},
                "players": {
                    "RED_SPYMASTER": {"type": "openai", "model": "gpt-4o-mini"},
                    "BLUE_SPYMASTER": {"type": "anthropic", "model": "claude-sonnet-4-20250514"},
                    "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                    "BLUE_OPERATIVE": {"type": "perplexity", "model": "sonar"},
                },
            }
        ),
        expected_status=400,
    )
    assert "same LLM signature" in message


def test_spymaster_eval_rejects_non_identical_operators() -> None:
    message = _expect_http_error(
        lambda: _new_match(
            {
                "seed": 103,
                "config": {"evaluation_mode": "spymaster_eval"},
                "players": {
                    "RED_SPYMASTER": "random",
                    "BLUE_SPYMASTER": "random",
                    "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                    "BLUE_OPERATIVE": {"type": "anthropic", "model": "claude-sonnet-4-20250514"},
                },
            }
        ),
        expected_status=400,
    )
    assert "same LLM signature" in message


def test_spymaster_eval_rejects_same_spymaster_models() -> None:
    message = _expect_http_error(
        lambda: _new_match(
            {
                "seed": 104,
                "config": {"evaluation_mode": "spymaster_eval"},
                "players": {
                    "RED_SPYMASTER": {"type": "openai", "model": "gpt-4o-mini"},
                    "BLUE_SPYMASTER": {"type": "openai", "model": "gpt-4o-mini"},
                    "RED_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                    "BLUE_OPERATIVE": {"type": "openai", "model": "gpt-4o-mini"},
                },
            }
        ),
        expected_status=400,
    )
    assert "Spymasters must use different models" in message


def test_wavelength_eval_rejects_same_guesser_models() -> None:
    message = _expect_http_error(
        lambda: _new_match(
            {
                "game": "wavelength",
                "seed": 105,
                "config": {"evaluation_mode": "guesser_eval"},
                "players": {
                    "PSYCHIC": {"type": "openai", "model": "gpt-4o-mini"},
                    "GUESSER_ONE": {"type": "openai", "model": "gpt-4o-mini"},
                    "GUESSER_TWO": {"type": "openai", "model": "gpt-4o-mini"},
                },
            }
        ),
        expected_status=400,
    )
    assert "Guessers must use different models" in message


def test_wavelength_match_records_only_guesser_report_cards(monkeypatch) -> None:
    _patch_network_agents_with_deterministic_stub(monkeypatch)
    created = _new_match(
        {
            "game": "wavelength",
            "seed": 106,
            "config": {"evaluation_mode": "guesser_eval"},
            "players": {
                "PSYCHIC": {"type": "openai", "model": "gpt-4o-mini"},
                "GUESSER_ONE": {"type": "openai", "model": "gpt-4o-mini"},
                "GUESSER_TWO": {"type": "anthropic", "model": "claude-sonnet-4-20250514"},
            },
            "viewer_player_id": "GUESSER_ONE",
        }
    )
    assert created["meta"]["game"] == "wavelength"
    assert created["meta"]["terminal"] is True

    report_payload = get_report_cards()
    assert "wavelength:guesser_eval" in report_payload["games"]

    bucket = report_payload["games"]["wavelength:guesser_eval"]
    assert any(label.endswith("[GUESSER]") for label in bucket.keys())
    assert not any(label.endswith("[PSYCHIC]") for label in bucket.keys())


def test_wavelength_incomplete_match_does_not_increment_report_cards(monkeypatch) -> None:
    original_factory = session_module.create_agent_for_player

    def _patched_factory(*, player_id, config, move_parser):
        player_type = str(config.get("type", "")).lower()
        if (
            player_id == "GUESSER_ONE"
            and player_type in {"openai", "anthropic", "perplexity", "local", "nemotron"}
        ):
            return _BadPromptAgent(agent_id=f"badprompt-{player_id.lower()}")
        if player_type in {"openai", "anthropic", "perplexity", "local", "nemotron", "random"}:
            return _FirstLegalAgent(agent_id=f"stub-{player_id.lower()}")
        return original_factory(player_id=player_id, config=config, move_parser=move_parser)

    monkeypatch.setattr(session_module, "create_agent_for_player", _patched_factory)

    before = _wavelength_reported_match_count(get_report_cards())
    created = _new_match(
        {
            "game": "wavelength",
            "seed": 107,
            "config": {"evaluation_mode": "guesser_eval"},
            "players": {
                "PSYCHIC": {"type": "openai", "model": "gpt-4o-mini"},
                "GUESSER_ONE": {"type": "openai", "model": "gpt-4o-mini"},
                "GUESSER_TWO": {"type": "anthropic", "model": "claude-sonnet-4-20250514"},
            },
            "viewer_player_id": "GUESSER_TWO",
        }
    )
    assert created["meta"]["terminal"] is True

    after = _wavelength_reported_match_count(get_report_cards())
    assert after == before
