"""In-memory match session management for stepwise human + AI play."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any
from uuid import uuid4

from codenames.codenames_game import CodenamesGame
from codenames.codenames_moves import EndTurn, GiveClue, Guess, Resign
from codenames.codenames_state import CodenamesState, Team
from framework.agents.random_agent import RandomAgent
from framework.events import EventType, MatchEvent
from framework.result import MatchResult, TerminationReason
from framework.serialize import to_serializable


def _team_from_string(value: str | None) -> Team | None:
    if value is None:
        return None
    try:
        return Team(value)
    except ValueError:
        return None


def _player_type_from_config(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip().lower()
    if isinstance(raw, dict):
        value = raw.get("type")
        if isinstance(value, str):
            return value.strip().lower()
    return "random"


def _move_summary(player_id: str, move_payload: dict[str, Any], state_before: CodenamesState, state_after: CodenamesState) -> str:
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
    if move_type == "Resign":
        return f"{player_id} resigned."
    return f"{player_id} played {move_type}."


def _serialize_legal_spec(game: CodenamesGame, state: CodenamesState, player_id: str) -> dict[str, Any]:
    raw = game.legal_moves(state, player_id)
    resign_legal, _ = game.is_legal(state, player_id, Resign())

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
                "Resign": resign_legal,
            },
        }

    if isinstance(raw, dict):
        serializable = {}
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
            "Resign": resign_legal,
        }
        return serializable

    return {"enumerated": [], "type": "mask", "allowed": {"Resign": resign_legal}}


def _serialize_observation(session: "MatchSession", player_id: str) -> dict[str, Any]:
    obs = session.game.observation(session.state, player_id)
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

    last_summary = None
    for event in reversed(session.events):
        if event.event_type == EventType.TURN:
            last_summary = str(event.payload.get("summary", ""))
            break

    return {
        "game": "codenames",
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
        "last_move_summary": last_summary,
        "turn_index": obs.turn_index,
        "last_move": obs.last_move,
    }


@dataclass
class MatchSession:
    """Single in-memory match session."""

    match_id: str
    seed: int
    config: dict[str, Any]
    game: CodenamesGame
    state: CodenamesState
    player_types: dict[str, str]
    agents: dict[str, RandomAgent]
    human_players: set[str]
    events: list[MatchEvent] = field(default_factory=list)
    result: MatchResult | None = None

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        config: dict[str, Any] | None,
        players: dict[str, Any] | None,
        human_player_id: str | None,
    ) -> "MatchSession":
        game = CodenamesGame()
        state = game.new_game(seed=seed, config=config or {})
        match_id = f"match-{uuid4().hex[:10]}"
        player_ids = list(game.player_ids(state))

        player_types: dict[str, str] = {}
        for player_id in player_ids:
            raw = (players or {}).get(player_id, "random")
            player_types[player_id] = _player_type_from_config(raw)
        if human_player_id is not None:
            player_types[human_player_id] = "human"

        human_players = {player_id for player_id, ptype in player_types.items() if ptype == "human"}
        if not human_players:
            current = game.current_player(state)
            player_types[current] = "human"
            human_players.add(current)

        agents: dict[str, RandomAgent] = {}
        for player_id in player_ids:
            if player_types[player_id] == "human":
                continue
            agent = RandomAgent(agent_id=f"random-{player_id.lower()}")
            role = game.role_for_player(state, player_id)
            agent.reset(match_id, player_id, role, seed, config or {})
            agents[player_id] = agent

        session = cls(
            match_id=match_id,
            seed=seed,
            config=dict(config or {}),
            game=game,
            state=state,
            player_types=player_types,
            agents=agents,
            human_players=human_players,
            events=[],
            result=None,
        )
        session.events.append(
            MatchEvent.create(
                event_type=EventType.MATCH_START,
                game_id=match_id,
                turn=0,
                payload={
                    "seed": seed,
                    "config": to_serializable(config or {}),
                    "players": player_types,
                    "current_player": game.current_player(state),
                },
            )
        )
        session.advance_until_human_turn()
        return session

    def current_player(self) -> str:
        """Return current player seat."""
        return self.game.current_player(self.state)

    def is_terminal(self) -> bool:
        """Return whether session state is terminal."""
        return self.game.is_terminal(self.state)

    def selected_player(self, requested_player_id: str | None = None) -> str:
        """Resolve which player's observation should be returned to caller."""
        if requested_player_id is not None:
            return requested_player_id
        if self.human_players:
            return sorted(self.human_players)[0]
        return self.current_player()

    def view(self, player_id: str) -> dict[str, Any]:
        """Return observation payload for one player."""
        observation = _serialize_observation(self, player_id)
        legal = _serialize_legal_spec(self.game, self.state, player_id)
        meta = {
            "match_id": self.match_id,
            "seed": self.seed,
            "current_player": self.current_player() if not self.is_terminal() else None,
            "is_human_turn": (not self.is_terminal()) and self.current_player() == player_id and player_id in self.human_players,
            "terminal": self.is_terminal(),
            "human_players": sorted(self.human_players),
        }
        response = {
            "match_id": self.match_id,
            "player_id": player_id,
            "observation": observation,
            "legal_moves_spec": legal,
            "meta": meta,
            "last_event": self.events[-1].to_dict() if self.events else None,
        }
        if self.result is not None:
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
        duration_ms = (perf_counter() - started) * 1000.0
        summary = _move_summary(player_id, move.to_dict(), before, self.state)
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
                self.events.append(
                    MatchEvent.create(
                        event_type=EventType.AGENT_ERROR,
                        game_id=self.match_id,
                        turn=self.state.turn_index,
                        payload={"player_id": player_id, "error": str(exc)},
                    )
                )
                self._forfeit_player(player_id, TerminationReason.AGENT_EXCEPTION, str(exc))
                return

            legal, reason = self.game.is_legal(self.state, player_id, move)
            if not legal:
                self.events.append(
                    MatchEvent.create(
                        event_type=EventType.ILLEGAL_MOVE,
                        game_id=self.match_id,
                        turn=self.state.turn_index,
                        payload={
                            "player_id": player_id,
                            "move": move.to_dict() if hasattr(move, "to_dict") else to_serializable(move),
                            "reason": reason,
                        },
                    )
                )
                self._forfeit_player(player_id, TerminationReason.ILLEGAL_MOVE_FORFEIT, reason or "illegal move")
                return

            before = self.state
            self.state = self.game.apply_move(self.state, player_id, move)
            duration_ms = (perf_counter() - started) * 1000.0
            summary = _move_summary(
                player_id,
                move.to_dict() if hasattr(move, "to_dict") else to_serializable(move),
                before,
                self.state,
            )
            self.events.append(
                MatchEvent.create(
                    event_type=EventType.TURN,
                    game_id=self.match_id,
                    turn=self.state.turn_index,
                    payload={
                        "player_id": player_id,
                        "move": move.to_dict() if hasattr(move, "to_dict") else to_serializable(move),
                        "duration_ms": duration_ms,
                        "summary": summary,
                    },
                )
            )

            self._finalize_if_terminal()

    def _forfeit_player(self, player_id: str, reason: TerminationReason, details: str) -> None:
        winner = self.game.forfeit_winner(self.state, player_id, reason.value)
        winner_team = _team_from_string(winner)
        self.state = replace(
            self.state,
            winner=winner_team,
            termination_reason=reason.value,
            turn_index=self.state.turn_index + 1,
            last_move={"type": "Forfeit", "player_id": player_id},
        )
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
            final_state_digest=self.state.state_digest(),
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

    def _finalize_if_terminal(self) -> None:
        if not self.is_terminal():
            return
        if self.result is not None:
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
            final_state_digest=self.state.state_digest(),
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


class SessionStore:
    """In-memory session dictionary keyed by match ID."""

    def __init__(self) -> None:
        self._sessions: dict[str, MatchSession] = {}

    def create_match(
        self,
        *,
        seed: int,
        config: dict[str, Any] | None,
        players: dict[str, Any] | None,
        human_player_id: str | None,
    ) -> MatchSession:
        session = MatchSession.create(
            seed=seed,
            config=config,
            players=players,
            human_player_id=human_player_id,
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
