"""In-memory match session management for stepwise human + AI play."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from codenames.codenames_game import CodenamesGame
from codenames.codenames_moves import EndTurn, GiveClue, Guess
from codenames.codenames_state import CodenamesState, Team
from framework.events import EventType, MatchEvent
from framework.game import Game
from framework.result import MatchResult, TerminationReason
from framework.serialize import to_serializable
from server.agent_factory import create_agent_for_player, normalize_player_config, player_label
from server.report_cards import ReportCardStore
from wavelength.wavelength_game import WavelengthGame
from wavelength.wavelength_state import GUESSER_ONE_PLAYER_ID, GUESSER_TWO_PLAYER_ID

GAME_CODENAMES = "codenames"
GAME_WAVELENGTH = "wavelength"
SUPPORTED_GAMES = {GAME_CODENAMES, GAME_WAVELENGTH}

EVALUATION_MODE_OPERATOR = "operator_eval"
EVALUATION_MODE_SPYMASTER = "spymaster_eval"
EVALUATION_MODE_GUESSER = "guesser_eval"
SUPPORTED_EVALUATION_MODES: dict[str, set[str]] = {
    GAME_CODENAMES: {EVALUATION_MODE_OPERATOR, EVALUATION_MODE_SPYMASTER},
    GAME_WAVELENGTH: {EVALUATION_MODE_GUESSER},
}

MODEL_BACKED_PLAYER_TYPES = {"openai", "anthropic", "perplexity", "local", "nemotron"}
LOCAL_LIKE_PLAYER_TYPES = {"local", "nemotron"}


def _normalize_game(game: str | None) -> str:
    normalized = (game or GAME_CODENAMES).strip().lower()
    if normalized not in SUPPORTED_GAMES:
        raise ValueError(f"Unsupported game '{game}'. Supported games: {sorted(SUPPORTED_GAMES)}")
    return normalized


def _build_game(game_name: str) -> Game[Any, Any, Any]:
    if game_name == GAME_CODENAMES:
        return CodenamesGame()
    if game_name == GAME_WAVELENGTH:
        return WavelengthGame()
    raise ValueError(f"Unsupported game '{game_name}'.")


def _default_evaluation_mode(game_name: str) -> str:
    if game_name == GAME_CODENAMES:
        return EVALUATION_MODE_OPERATOR
    return EVALUATION_MODE_GUESSER


def _normalize_evaluation_mode(game_name: str, config: dict[str, Any] | None) -> str:
    raw = (config or {}).get("evaluation_mode", _default_evaluation_mode(game_name))
    mode = str(raw).strip().lower()
    supported = SUPPORTED_EVALUATION_MODES[game_name]
    if mode not in supported:
        raise ValueError(
            "config.evaluation_mode must be one of "
            f"{sorted(supported)} for game '{game_name}'; received {raw!r}."
        )
    return mode


def _team_from_string(value: str | None) -> Team | None:
    if value is None:
        return None
    try:
        return Team(value)
    except ValueError:
        return None


def _state_digest(state: Any) -> str | None:
    if hasattr(state, "state_digest") and callable(state.state_digest):
        return str(state.state_digest())
    return None


def _move_summary(game_name: str, player_id: str, move_payload: dict[str, Any], state_before: Any, state_after: Any) -> str:
    if game_name == GAME_CODENAMES:
        move_type = str(move_payload.get("type", "Move"))
        if move_type == "GiveClue":
            clue = move_payload.get("clue")
            count = move_payload.get("count")
            return f"{player_id} gave clue '{clue}' ({count})."
        if move_type == "Guess":
            index = int(move_payload.get("index", -1))
            if 0 <= index < len(state_after.board_words):
                word = state_after.board_words[index]
                card = state_after.assignments[index].value
                return f"{player_id} guessed {word} ({card})."
            return f"{player_id} guessed index {index}."
        if move_type == "EndTurn":
            return f"{player_id} ended turn."
        return f"{player_id} played {move_type}."

    move_type = str(move_payload.get("type", "Move"))
    if move_type == "ChooseNumber":
        return f"{player_id} locked in the hidden target number."
    if move_type == "AskCategory":
        return f"{player_id} asked category '{move_payload.get('category')}'."
    if move_type == "GiveAnswer":
        return f"{player_id} answered '{move_payload.get('answer')}'."
    if move_type == "SubmitGuess":
        guess = move_payload.get("value")
        return f"{player_id} guessed {guess}."
    if move_type == "SubmitFinalEstimate":
        return f"{player_id} submitted a final estimate."
    return f"{player_id} played {move_type}."


def _serialize_generic_legal_spec(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        return {
            "enumerated": [
                move.to_dict() if hasattr(move, "to_dict") else to_serializable(move)
                for move in raw
            ],
            "type": "enumerated",
            "allowed": {},
        }

    if isinstance(raw, dict):
        serializable: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "sample_moves" and isinstance(value, list):
                serializable[key] = [
                    move.to_dict() if hasattr(move, "to_dict") else to_serializable(move)
                    for move in value
                ]
            else:
                serializable[key] = to_serializable(value)
        serializable.setdefault("type", "mask")
        serializable.setdefault("allowed", {})
        return serializable

    return {"enumerated": [], "type": "mask", "allowed": {}}


def _serialize_legal_spec(game_name: str, game: Game[Any, Any, Any], state: Any, player_id: str) -> dict[str, Any]:
    raw = game.legal_moves(state, player_id)
    if game_name != GAME_CODENAMES:
        return _serialize_generic_legal_spec(raw)

    if isinstance(raw, list):
        enumerated = [
            move.to_dict() if hasattr(move, "to_dict") else to_serializable(move)
            for move in raw
        ]
        guess_indices: list[int] = []
        can_end_turn = False
        can_give_clue = False
        for move in raw:
            if isinstance(move, Guess):
                guess_indices.append(move.index)
            elif isinstance(move, EndTurn):
                can_end_turn = True
            elif isinstance(move, GiveClue):
                can_give_clue = True
        return {
            "enumerated": enumerated,
            "type": "mask",
            "allowed": {
                "Guess": {"indices": guess_indices},
                "EndTurn": can_end_turn,
                "GiveClue": can_give_clue,
            },
        }

    if isinstance(raw, dict):
        serializable: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "sample_moves" and isinstance(value, list):
                serializable[key] = [
                    move.to_dict() if hasattr(move, "to_dict") else to_serializable(move)
                    for move in value
                ]
            else:
                serializable[key] = to_serializable(value)
        clue_count_max = state.team_words_remaining(state.turn_team)
        serializable["type"] = "mask"
        serializable["allowed"] = {
            "GiveClue": {"count_min": 0, "count_max": int(clue_count_max)},
        }
        return serializable

    return {"enumerated": [], "type": "mask", "allowed": {}}


def _last_summary_for_turn(session: "MatchSession", replay_turn: int) -> str | None:
    for event in reversed(session.events):
        if event.event_type == EventType.TURN and event.turn <= replay_turn:
            return str(event.payload.get("summary", ""))
    return None


def _serialize_codenames_observation(*, session: "MatchSession", state: Any, player_id: str, replay_turn: int) -> dict[str, Any]:
    obs = session.game.observation(state, player_id)
    board = []
    for index, word in enumerate(obs.board_words):
        revealed = bool(obs.revealed[index])
        revealed_color = obs.revealed_colors[index].value if obs.revealed_colors[index] is not None else None
        assignment = None
        if obs.assignments is not None:
            assignment = obs.assignments[index].value
        board.append(
            {
                "index": index,
                "word": word,
                "revealed": revealed,
                "revealed_color": revealed_color,
                "assignment": assignment,
            }
        )

    clue = None
    if obs.current_clue is not None:
        clue = {"clue": obs.current_clue[0], "count": obs.current_clue[1]}

    return {
        "game": GAME_CODENAMES,
        "player_id": obs.player_id,
        "role": obs.role.value,
        "team": obs.team.value,
        "turn_team": obs.turn_team.value,
        "phase": obs.phase.value,
        "board": board,
        "current_clue": clue,
        "guesses_remaining": obs.guesses_remaining,
        "public_counts": {
            "RED_LEFT": int(obs.remaining_targets.get("RED", 0)),
            "BLUE_LEFT": int(obs.remaining_targets.get("BLUE", 0)),
        },
        "last_move_summary": _last_summary_for_turn(session, replay_turn),
        "turn_index": obs.turn_index,
        "last_move": obs.last_move,
    }


def _serialize_wavelength_observation(*, session: "MatchSession", state: Any, player_id: str, replay_turn: int) -> dict[str, Any]:
    obs = session.game.observation(state, player_id)
    return {
        "game": GAME_WAVELENGTH,
        "player_id": obs.player_id,
        "role": obs.role.value,
        "phase": obs.phase.value,
        "current_player": obs.current_player,
        "current_round": obs.current_round,
        "max_rounds": obs.max_rounds,
        "target_number": obs.target_number,
        "history": to_serializable(obs.history),
        "pending_exchange": to_serializable(obs.pending_exchange),
        "guess_counts": to_serializable(obs.guess_counts),
        "final_estimates": to_serializable(obs.final_estimates),
        "winner": obs.winner,
        "termination_reason": obs.termination_reason,
        "last_move_summary": _last_summary_for_turn(session, replay_turn),
        "turn_index": obs.turn_index,
        "last_move": to_serializable(obs.last_move),
    }


def _serialize_observation(*, session: "MatchSession", state: Any, player_id: str, replay_turn: int) -> dict[str, Any]:
    if session.game_name == GAME_CODENAMES:
        return _serialize_codenames_observation(
            session=session,
            state=state,
            player_id=player_id,
            replay_turn=replay_turn,
        )
    return _serialize_wavelength_observation(
        session=session,
        state=state,
        player_id=player_id,
        replay_turn=replay_turn,
    )


def _agent_prompt_context(agent: Any) -> dict[str, Any] | None:
    if not hasattr(agent, "debug_context") or not callable(agent.debug_context):
        return None
    try:
        context = agent.debug_context()
    except Exception:
        return None
    if context is None:
        return None
    return to_serializable(context)


def _agent_response_payload(prompt_context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract compact model-response diagnostics from agent prompt context."""
    if not isinstance(prompt_context, dict):
        return {}
    raw_responses = prompt_context.get("raw_responses")
    if not isinstance(raw_responses, list) or not raw_responses:
        return {}
    serializable_history = to_serializable(raw_responses)
    if not isinstance(serializable_history, list) or not serializable_history:
        return {}
    last_response = serializable_history[-1]
    response_text = last_response if isinstance(last_response, str) else str(last_response)
    return {
        "model_response": response_text,
        "model_response_history": serializable_history,
    }


def _llm_signature(player_config: dict[str, Any]) -> tuple[str, ...] | None:
    player_type = str(player_config.get("type", "")).strip().lower()
    if player_type not in MODEL_BACKED_PLAYER_TYPES:
        return None

    model = str(player_config.get("model", "")).strip().lower()
    if not model:
        return None

    if player_type in LOCAL_LIKE_PLAYER_TYPES:
        default_backend = "transformers"
        backend = str(player_config.get("backend", default_backend)).strip().lower() or default_backend
        return player_type, backend, model
    return player_type, model


def _validate_codenames_evaluation_mode_constraints(
    *,
    evaluation_mode: str,
    player_configs: dict[str, dict[str, Any]],
) -> None:
    red_spymaster_sig = _llm_signature(player_configs["RED_SPYMASTER"])
    blue_spymaster_sig = _llm_signature(player_configs["BLUE_SPYMASTER"])
    red_operative_sig = _llm_signature(player_configs["RED_OPERATIVE"])
    blue_operative_sig = _llm_signature(player_configs["BLUE_OPERATIVE"])

    if evaluation_mode == EVALUATION_MODE_OPERATOR:
        if red_spymaster_sig is None or blue_spymaster_sig is None:
            raise ValueError("Humans cannot be used as a comparison for model evaluation")
        if red_spymaster_sig != blue_spymaster_sig:
            raise ValueError("Both spymasters must use the same LLM signature in operator_eval mode")
        if red_operative_sig == blue_operative_sig:
            raise ValueError("Operators must use different models in operator_eval mode")
        return

    if evaluation_mode == EVALUATION_MODE_SPYMASTER:
        if red_operative_sig is None or blue_operative_sig is None:
            raise ValueError("Humans cannot be used as a comparison for model evaluation")
        if red_operative_sig != blue_operative_sig:
            raise ValueError("Both operatives must use the same LLM signature in spymaster_eval mode")
        if red_spymaster_sig is not None and blue_spymaster_sig is not None and red_spymaster_sig == blue_spymaster_sig:
            raise ValueError("Spymasters must use different models in spymaster_eval mode")
        return

    raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")


def _validate_wavelength_evaluation_mode_constraints(
    *,
    evaluation_mode: str,
    player_configs: dict[str, dict[str, Any]],
) -> None:
    if evaluation_mode != EVALUATION_MODE_GUESSER:
        raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")

    guesser_one_sig = _llm_signature(player_configs[GUESSER_ONE_PLAYER_ID])
    guesser_two_sig = _llm_signature(player_configs[GUESSER_TWO_PLAYER_ID])
    if guesser_one_sig is None or guesser_two_sig is None:
        raise ValueError("Humans cannot be used as a comparison for model evaluation")
    if guesser_one_sig == guesser_two_sig:
        raise ValueError("Guessers must use different models in guesser_eval mode")


def _validate_evaluation_mode_constraints(
    *,
    game_name: str,
    evaluation_mode: str,
    player_configs: dict[str, dict[str, Any]],
) -> None:
    if game_name == GAME_CODENAMES:
        _validate_codenames_evaluation_mode_constraints(
            evaluation_mode=evaluation_mode,
            player_configs=player_configs,
        )
        return
    _validate_wavelength_evaluation_mode_constraints(
        evaluation_mode=evaluation_mode,
        player_configs=player_configs,
    )


def _default_view_player_id(
    *,
    game_name: str,
    player_configs: Mapping[str, Any],
    human_players: set[str],
    human_player_id: str | None,
    viewer_player_id: str | None,
) -> str:
    if viewer_player_id is not None:
        return viewer_player_id
    if game_name == GAME_CODENAMES and "RED_SPYMASTER" in player_configs:
        return "RED_SPYMASTER"
    if game_name == GAME_WAVELENGTH and GUESSER_ONE_PLAYER_ID in player_configs:
        return GUESSER_ONE_PLAYER_ID
    if human_player_id is not None:
        return human_player_id
    if human_players:
        return sorted(human_players)[0]
    return sorted(player_configs.keys())[0]


def _ranked_player_ids(game_name: str, player_roles: Mapping[str, str | None]) -> set[str]:
    if game_name != GAME_WAVELENGTH:
        return set(player_roles.keys())
    return {
        player_id
        for player_id, role in player_roles.items()
        if (role or "").upper() != "PSYCHIC"
    }


def _wavelength_match_completed_rounds(*, state: Any, config: Mapping[str, Any]) -> bool:
    raw_max_rounds = config.get("max_rounds", 3)
    try:
        max_rounds = max(1, int(raw_max_rounds))
    except (TypeError, ValueError):
        max_rounds = 3

    guess_counts = getattr(state, "guess_counts", None)
    if not isinstance(guess_counts, Mapping):
        return False

    return all(
        int(guess_counts.get(guesser, 0)) >= max_rounds
        for guesser in (GUESSER_ONE_PLAYER_ID, GUESSER_TWO_PLAYER_ID)
    )


def _apply_forfeit_state(*, game_name: str, state: Any, winner: str | None, offending_player_id: str, reason: TerminationReason) -> Any:
    if game_name == GAME_CODENAMES:
        winner_team = _team_from_string(winner)
        return replace(
            state,
            winner=winner_team,
            termination_reason=reason.value,
            turn_index=state.turn_index + 1,
            last_move={"type": "Forfeit", "player_id": offending_player_id},
        )

    return replace(
        state,
        winner=winner,
        termination_reason=reason.value,
        turn_index=state.turn_index + 1,
        last_move={"type": "Forfeit", "player_id": offending_player_id},
    )


@dataclass
class MatchSession:
    """Single in-memory match session."""

    match_id: str
    game_name: str
    seed: int
    config: dict[str, Any]
    game: Game[Any, Any, Any]
    state: Any
    state_history: list[Any]
    player_configs: dict[str, dict[str, Any]]
    player_roles: dict[str, str | None]
    ranked_player_ids: set[str]
    agent_labels: dict[str, str]
    agents: dict[str, Any]
    human_players: set[str]
    default_view_player_id: str
    evaluation_mode: str
    events: list[MatchEvent] = field(default_factory=list)
    result: MatchResult | None = None
    on_result: Callable[["MatchSession", MatchResult], None] | None = None
    _report_recorded: bool = False

    @classmethod
    def create(
        cls,
        *,
        game: str,
        seed: int,
        config: dict[str, Any] | None,
        players: dict[str, Any] | None,
        human_player_id: str | None,
        viewer_player_id: str | None,
        on_result: Callable[["MatchSession", MatchResult], None] | None,
    ) -> "MatchSession":
        game_name = _normalize_game(game)
        config_payload = dict(config or {})
        evaluation_mode = _normalize_evaluation_mode(game_name, config_payload)
        config_payload["evaluation_mode"] = evaluation_mode

        game_impl = _build_game(game_name)
        state = game_impl.new_game(seed=seed, config=config_payload)
        match_id = f"match-{uuid4().hex[:10]}"
        player_ids = list(game_impl.player_ids(state))

        player_configs: dict[str, dict[str, Any]] = {}
        for player_id in player_ids:
            raw = (players or {}).get(player_id, "random")
            player_configs[player_id] = normalize_player_config(raw)

        if human_player_id is not None:
            if human_player_id not in player_configs:
                raise ValueError(f"Unknown human_player_id: {human_player_id}")
            player_configs[human_player_id]["type"] = "human"

        _validate_evaluation_mode_constraints(
            game_name=game_name,
            evaluation_mode=evaluation_mode,
            player_configs=player_configs,
        )

        human_players = {
            player_id
            for player_id, player_config in player_configs.items()
            if str(player_config.get("type", "random")).lower() == "human"
        }

        if viewer_player_id is not None and viewer_player_id not in player_configs:
            raise ValueError(f"Unknown viewer_player_id: {viewer_player_id}")

        default_view_player_id = _default_view_player_id(
            game_name=game_name,
            player_configs=player_configs,
            human_players=human_players,
            human_player_id=human_player_id,
            viewer_player_id=viewer_player_id,
        )

        agents: dict[str, Any] = {}
        agent_labels: dict[str, str] = {}
        player_roles: dict[str, str | None] = {}
        for player_id in player_ids:
            normalized = player_configs[player_id]
            agent_labels[player_id] = player_label(normalized)
            role = game_impl.role_for_player(state, player_id)
            player_roles[player_id] = role
            if player_id in human_players:
                continue
            agent = create_agent_for_player(
                player_id=player_id,
                config=normalized,
                move_parser=lambda payload, g=game_impl: g.parse_move(payload),
            )
            agent.reset(match_id, player_id, role, seed, config_payload)
            agents[player_id] = agent

        ranked_player_ids = _ranked_player_ids(game_name, player_roles)

        session = cls(
            match_id=match_id,
            game_name=game_name,
            seed=seed,
            config=config_payload,
            game=game_impl,
            state=state,
            state_history=[state],
            player_configs=player_configs,
            player_roles=player_roles,
            ranked_player_ids=ranked_player_ids,
            agent_labels=agent_labels,
            agents=agents,
            human_players=human_players,
            default_view_player_id=default_view_player_id,
            evaluation_mode=evaluation_mode,
            events=[],
            result=None,
            on_result=on_result,
        )
        session.events.append(
            MatchEvent.create(
                event_type=EventType.MATCH_START,
                game_id=match_id,
                turn=0,
                payload={
                    "game": game_name,
                    "seed": seed,
                    "config": to_serializable(config_payload),
                    "players": to_serializable(player_configs),
                    "current_player": game_impl.current_player(state),
                    "evaluation_mode": evaluation_mode,
                },
            )
        )
        session.advance_until_human_turn()
        return session

    def current_player(self) -> str:
        """Return current player seat for the live state."""
        return self.game.current_player(self.state)

    def is_terminal(self) -> bool:
        """Return whether live state is terminal."""
        return self.game.is_terminal(self.state)

    def selected_player(self, requested_player_id: str | None = None) -> str:
        """Resolve which player observation should be returned to caller."""
        if requested_player_id is not None:
            if requested_player_id not in self.player_configs:
                raise ValueError(f"Unknown player_id: {requested_player_id}")
            return requested_player_id
        return self.default_view_player_id

    def view(self, player_id: str, *, turn: int | None = None) -> dict[str, Any]:
        """Return observation payload for one player and optional replay turn."""
        if player_id not in self.player_configs:
            raise ValueError(f"Unknown player_id: {player_id}")

        max_turn = len(self.state_history) - 1
        if turn is None:
            replay_turn = max_turn
        else:
            replay_turn = max(0, min(int(turn), max_turn))

        state_for_view = self.state_history[replay_turn]
        is_live = replay_turn == max_turn
        terminal_for_view = self.game.is_terminal(state_for_view)
        current_player = self.game.current_player(state_for_view) if not terminal_for_view else None
        is_human_turn = (
            is_live
            and (not terminal_for_view)
            and current_player == player_id
            and player_id in self.human_players
        )

        observation = _serialize_observation(
            session=self,
            state=state_for_view,
            player_id=player_id,
            replay_turn=replay_turn,
        )
        legal = _serialize_legal_spec(self.game_name, self.game, state_for_view, player_id)

        response = {
            "match_id": self.match_id,
            "player_id": player_id,
            "observation": observation,
            "legal_moves_spec": legal,
            "meta": {
                "game": self.game_name,
                "match_id": self.match_id,
                "seed": self.seed,
                "current_player": current_player,
                "is_human_turn": is_human_turn,
                "terminal": terminal_for_view,
                "human_players": sorted(self.human_players),
                "replay_turn": replay_turn,
                "max_turn": max_turn,
                "is_live": is_live,
                "evaluation_mode": self.evaluation_mode,
                "player_configs": to_serializable(self.player_configs),
            },
            "last_event": self._last_event_for_turn(
                replay_turn,
                viewer_player_id=player_id,
                terminal_for_view=terminal_for_view,
            ),
        }

        if self.result is not None and replay_turn >= self.result.turns:
            response["result"] = self.result.to_dict()
        return response

    def submit_human_move(self, *, player_id: str, move_payload: dict[str, Any]) -> dict[str, Any]:
        """Apply a human move, then auto-play AI turns until next human turn."""
        if self.is_terminal():
            return self.view(player_id)
        if player_id not in self.human_players:
            raise PermissionError(f"Player {player_id} is not configured as human.")
        if self.current_player() != player_id:
            raise ValueError(f"It is not {player_id}'s turn.")

        move = self.game.parse_move(move_payload)
        legal, reason = self.game.is_legal(self.state, player_id, move)
        if not legal:
            self.events.append(
                MatchEvent.create(
                    event_type=EventType.ILLEGAL_MOVE,
                    game_id=self.match_id,
                    turn=self.state.turn_index,
                    payload={"player_id": player_id, "move": to_serializable(move_payload), "reason": reason},
                )
            )
            raise ValueError(reason or "Illegal move.")

        before = self.state
        started = perf_counter()
        self.state = self.game.apply_move(self.state, player_id, move)
        self.state_history.append(self.state)
        duration_ms = (perf_counter() - started) * 1000.0
        summary = _move_summary(self.game_name, player_id, move.to_dict(), before, self.state)
        self.events.append(
            MatchEvent.create(
                event_type=EventType.TURN,
                game_id=self.match_id,
                turn=self.state.turn_index,
                payload={
                    "player_id": player_id,
                    "move": move.to_dict(),
                    "duration_ms": duration_ms,
                    "summary": summary,
                },
            )
        )

        self._finalize_if_terminal()
        if not self.is_terminal():
            self.advance_until_human_turn()
        return self.view(player_id)

    def advance_until_human_turn(self) -> None:
        """Run AI turns until a human turn is reached or game ends."""
        while not self.is_terminal():
            player_id = self.current_player()
            if player_id in self.human_players:
                return

            agent = self.agents[player_id]
            observation = self.game.observation(self.state, player_id)
            legal_spec = self.game.legal_moves(self.state, player_id)
            started = perf_counter()
            try:
                move = agent.act(observation, legal_spec)
            except Exception as exc:
                prompt_context = _agent_prompt_context(agent)
                payload: dict[str, Any] = {
                    "player_id": player_id,
                    "error": str(exc),
                    **_agent_response_payload(prompt_context),
                }
                if prompt_context is not None:
                    payload["prompt_context"] = prompt_context
                self.events.append(
                    MatchEvent.create(
                        event_type=EventType.AGENT_ERROR,
                        game_id=self.match_id,
                        turn=self.state.turn_index,
                        payload=payload,
                    )
                )
                self._forfeit_player(
                    player_id,
                    TerminationReason.AGENT_EXCEPTION,
                    str(exc),
                    prompt_context=prompt_context,
                )
                return

            legal, reason = self.game.is_legal(self.state, player_id, move)
            if not legal:
                prompt_context = _agent_prompt_context(agent)
                payload = {
                    "player_id": player_id,
                    "move": move.to_dict() if hasattr(move, "to_dict") else to_serializable(move),
                    "reason": reason,
                    **_agent_response_payload(prompt_context),
                }
                if prompt_context is not None:
                    payload["prompt_context"] = prompt_context
                self.events.append(
                    MatchEvent.create(
                        event_type=EventType.ILLEGAL_MOVE,
                        game_id=self.match_id,
                        turn=self.state.turn_index,
                        payload=payload,
                    )
                )
                self._forfeit_player(
                    player_id,
                    TerminationReason.ILLEGAL_MOVE_FORFEIT,
                    reason or "illegal move",
                    prompt_context=prompt_context,
                )
                return

            before = self.state
            self.state = self.game.apply_move(self.state, player_id, move)
            self.state_history.append(self.state)
            duration_ms = (perf_counter() - started) * 1000.0
            move_payload = move.to_dict() if hasattr(move, "to_dict") else to_serializable(move)
            summary = _move_summary(self.game_name, player_id, move_payload, before, self.state)
            prompt_context = _agent_prompt_context(agent)
            self.events.append(
                MatchEvent.create(
                    event_type=EventType.TURN,
                    game_id=self.match_id,
                    turn=self.state.turn_index,
                    payload={
                        "player_id": player_id,
                        "move": move_payload,
                        "duration_ms": duration_ms,
                        "summary": summary,
                        **_agent_response_payload(prompt_context),
                    },
                )
            )
            self._finalize_if_terminal()

    def _forfeit_player(
        self,
        player_id: str,
        reason: TerminationReason,
        details: str,
        *,
        prompt_context: dict[str, Any] | None = None,
    ) -> None:
        winner = self.game.forfeit_winner(self.state, player_id, reason.value)
        self.state = _apply_forfeit_state(
            game_name=self.game_name,
            state=self.state,
            winner=winner,
            offending_player_id=player_id,
            reason=reason,
        )
        self.state_history.append(self.state)
        self.result = MatchResult(
            game_id=self.match_id,
            game_name=self.game.game_name,
            seed=self.seed,
            winner=winner,
            termination_reason=reason,
            scores={},
            turns=self.state.turn_index,
            stats={},
            details=details,
            final_state_digest=_state_digest(self.state),
            event_count=len(self.events) + 1,
            log_path=None,
        )
        self.events.append(
            MatchEvent.create(
                event_type=EventType.TERMINAL,
                game_id=self.match_id,
                turn=self.state.turn_index,
                payload={
                    "result": self.result.to_dict(),
                    **_agent_response_payload(prompt_context),
                    **({"prompt_context": prompt_context} if prompt_context is not None else {}),
                },
            )
        )
        self._record_result_if_needed()

    def _finalize_if_terminal(self) -> None:
        if not self.is_terminal() or self.result is not None:
            return

        raw = self.game.outcome(self.state)
        self.result = MatchResult(
            game_id=self.match_id,
            game_name=raw.game_name,
            seed=self.seed,
            winner=raw.winner,
            termination_reason=raw.termination_reason,
            scores=raw.scores,
            turns=self.state.turn_index,
            stats=raw.stats,
            details=raw.details,
            final_state_digest=_state_digest(self.state),
            event_count=len(self.events) + 1,
            log_path=None,
        )
        self.events.append(
            MatchEvent.create(
                event_type=EventType.TERMINAL,
                game_id=self.match_id,
                turn=self.state.turn_index,
                payload={"result": self.result.to_dict()},
            )
        )
        self._record_result_if_needed()

    def _record_result_if_needed(self) -> None:
        if self._report_recorded or self.result is None:
            return
        if self.on_result is not None:
            self.on_result(self, self.result)
        self._report_recorded = True

    def _sanitize_event_for_view(
        self,
        *,
        event_data: dict[str, Any],
        viewer_player_id: str,
        terminal_for_view: bool,
    ) -> dict[str, Any]:
        payload = dict(event_data.get("payload", {}))
        # Keep player-facing summaries compact and avoid exposing model internals.
        payload.pop("prompt_context", None)
        payload.pop("model_response", None)
        payload.pop("model_response_history", None)

        if (
            self.game_name == GAME_WAVELENGTH
            and not terminal_for_view
            and viewer_player_id in {GUESSER_ONE_PLAYER_ID, GUESSER_TWO_PLAYER_ID}
        ):
            move = payload.get("move")
            actor = payload.get("player_id")
            if (
                isinstance(move, dict)
                and move.get("type") == "SubmitFinalEstimate"
                and isinstance(actor, str)
                and actor != viewer_player_id
            ):
                redacted_move = dict(move)
                redacted_move.pop("value", None)
                payload["move"] = redacted_move

        sanitized = dict(event_data)
        sanitized["payload"] = payload
        return sanitized

    def _last_event_for_turn(
        self,
        replay_turn: int,
        *,
        viewer_player_id: str,
        terminal_for_view: bool,
    ) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.turn <= replay_turn:
                return self._sanitize_event_for_view(
                    event_data=event.to_dict(),
                    viewer_player_id=viewer_player_id,
                    terminal_for_view=terminal_for_view,
                )
        if not self.events:
            return None
        return self._sanitize_event_for_view(
            event_data=self.events[0].to_dict(),
            viewer_player_id=viewer_player_id,
            terminal_for_view=terminal_for_view,
        )


class SessionStore:
    """In-memory session dictionary keyed by match ID."""

    def __init__(self, report_store: ReportCardStore | None = None) -> None:
        self._sessions: dict[str, MatchSession] = {}
        default_path = Path(os.getenv("REPORT_CARD_DB_PATH", "server/data/report_cards.json"))
        self.report_store = report_store or ReportCardStore(path=default_path)

    def create_match(
        self,
        *,
        game: str,
        seed: int,
        config: dict[str, Any] | None,
        players: dict[str, Any] | None,
        human_player_id: str | None,
        viewer_player_id: str | None,
    ) -> MatchSession:
        session = MatchSession.create(
            game=game,
            seed=seed,
            config=config,
            players=players,
            human_player_id=human_player_id,
            viewer_player_id=viewer_player_id,
            on_result=self._on_match_result,
        )
        self._sessions[session.match_id] = session
        return session

    def get(self, match_id: str) -> MatchSession:
        if match_id not in self._sessions:
            raise KeyError(match_id)
        return self._sessions[match_id]

    def all_events(self, match_id: str) -> list[dict[str, Any]]:
        session = self.get(match_id)
        return [event.to_dict() for event in session.events]

    def report_cards(self) -> dict[str, Any]:
        return self.report_store.report()

    def _on_match_result(self, session: MatchSession, result: MatchResult) -> None:
        ranked_labels = {
            player_id: label
            for player_id, label in session.agent_labels.items()
            if player_id in session.ranked_player_ids
        }
        if not ranked_labels:
            return
        if session.game_name == GAME_WAVELENGTH and not _wavelength_match_completed_rounds(
            state=session.state,
            config=session.config,
        ):
            return

        self.report_store.record_match(
            game_name=f"{session.game.game_name}:{session.evaluation_mode}",
            agent_labels=ranked_labels,
            winner=result.winner,
            termination_reason=result.termination_reason.value,
            turns=result.turns,
        )
