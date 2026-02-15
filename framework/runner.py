"""Autonomous match runner for state-based games."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .errors import AgentExecutionError, AgentTimeoutError, IllegalMoveError, MatchConfigurationError
from .events import EventType, MatchEvent, write_jsonl
from .game import Game, PlayerId
from .result import MatchResult, TerminationReason
from .serialize import digest, to_serializable


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime configuration for match execution."""

    max_turns: int = 200
    move_timeout_sec: float | None = None
    forfeit_on_illegal: bool = True
    max_illegal_retries: int = 0
    max_turns_policy: str = "draw"
    event_log_dir: str | Path | None = None


@dataclass(frozen=True)
class MatchRun:
    """Complete execution artifact for one match."""

    result: MatchResult
    events: list[MatchEvent]


class MatchRunner:
    """Runs matches to completion with legality checks and event logging."""

    def __init__(self, config: RunnerConfig | None = None):
        self.config = config or RunnerConfig()

    def run_match(
        self,
        game: Game[Any, Any, Any],
        agents: Mapping[PlayerId, Any] | Sequence[Any],
        seed: int,
        game_config: dict[str, Any] | None = None,
        *,
        game_id: str | None = None,
        log_path: str | Path | None = None,
    ) -> MatchRun:
        """Run a full match and return result + event history."""
        config = game_config or {}
        resolved_game_id = game_id or f"{game.game_name}-{seed}-{uuid4().hex[:8]}"
        state = game.new_game(seed=seed, config=config)

        player_ids = list(game.player_ids(state))
        normalized_agents = self._normalize_agents(player_ids=player_ids, agents=agents)
        self._validate_agents(player_ids=player_ids, agents=normalized_agents)

        history: list[MatchEvent] = []
        history.append(
            MatchEvent.create(
                event_type=EventType.MATCH_START,
                game_id=resolved_game_id,
                turn=0,
                payload={
                    "seed": seed,
                    "config": to_serializable(config),
                    "players": player_ids,
                    "initial_state_digest": self._state_digest(state),
                },
            )
        )

        for player_id in player_ids:
            role = game.role_for_player(state, player_id)
            normalized_agents[player_id].reset(resolved_game_id, player_id, role, seed, config)

        illegal_move_counts: dict[str, int] = defaultdict(int)
        move_durations_ms: dict[str, list[float]] = defaultdict(list)
        turn = 0

        while not game.is_terminal(state):
            if turn >= self.config.max_turns:
                result = self._forced_result(
                    game=game,
                    game_id=resolved_game_id,
                    seed=seed,
                    state=state,
                    winner=None,
                    reason=TerminationReason.MAX_TURNS,
                    details=f"Reached max_turns={self.config.max_turns}.",
                    turn=turn,
                    illegal_move_counts=illegal_move_counts,
                    move_durations_ms=move_durations_ms,
                )
                history.append(
                    MatchEvent.create(
                        event_type=EventType.TERMINAL,
                        game_id=resolved_game_id,
                        turn=turn,
                        payload={"result": result.to_dict()},
                    )
                )
                return self._finish(
                    agents=normalized_agents,
                    result=result,
                    history=history,
                    log_path=log_path,
                    game_id=resolved_game_id,
                )

            player_id = game.current_player(state)
            if player_id not in normalized_agents:
                raise MatchConfigurationError(f"Missing agent for current player {player_id!r}.")

            agent = normalized_agents[player_id]
            observation = game.observation(state, player_id)
            legal_moves_spec = game.legal_moves(state, player_id)

            attempt = 0
            move = None
            duration_ms = 0.0
            while True:
                start = perf_counter()
                try:
                    move = agent.act(observation, legal_moves_spec)
                except Exception as exc:  # pragma: no cover - protective path
                    error = AgentExecutionError(player_id, f"Agent act() failed: {exc}")
                    history.append(
                        MatchEvent.create(
                            event_type=EventType.AGENT_ERROR,
                            game_id=resolved_game_id,
                            turn=turn,
                            payload={"player_id": player_id, "error": error.to_dict()},
                        )
                    )
                    winner = game.forfeit_winner(state, player_id, "agent_exception")
                    result = self._forced_result(
                        game=game,
                        game_id=resolved_game_id,
                        seed=seed,
                        state=state,
                        winner=winner,
                        reason=TerminationReason.AGENT_EXCEPTION,
                        details=str(error),
                        turn=turn,
                        illegal_move_counts=illegal_move_counts,
                        move_durations_ms=move_durations_ms,
                    )
                    history.append(
                        MatchEvent.create(
                            event_type=EventType.TERMINAL,
                            game_id=resolved_game_id,
                            turn=turn,
                            payload={"result": result.to_dict()},
                        )
                    )
                    return self._finish(
                        agents=normalized_agents,
                        result=result,
                        history=history,
                        log_path=log_path,
                        game_id=resolved_game_id,
                    )

                duration_ms = (perf_counter() - start) * 1000.0
                move_durations_ms[player_id].append(duration_ms)

                if self.config.move_timeout_sec is not None and duration_ms > self.config.move_timeout_sec * 1000.0:
                    timeout_error = AgentTimeoutError(
                        player_id,
                        f"Move took {duration_ms:.2f}ms > limit {self.config.move_timeout_sec * 1000.0:.2f}ms.",
                    )
                    history.append(
                        MatchEvent.create(
                            event_type=EventType.AGENT_ERROR,
                            game_id=resolved_game_id,
                            turn=turn,
                            payload={"player_id": player_id, "error": timeout_error.to_dict()},
                        )
                    )
                    winner = game.forfeit_winner(state, player_id, "timeout")
                    result = self._forced_result(
                        game=game,
                        game_id=resolved_game_id,
                        seed=seed,
                        state=state,
                        winner=winner,
                        reason=TerminationReason.TIMEOUT_FORFEIT,
                        details=str(timeout_error),
                        turn=turn,
                        illegal_move_counts=illegal_move_counts,
                        move_durations_ms=move_durations_ms,
                    )
                    history.append(
                        MatchEvent.create(
                            event_type=EventType.TERMINAL,
                            game_id=resolved_game_id,
                            turn=turn,
                            payload={"result": result.to_dict()},
                        )
                    )
                    return self._finish(
                        agents=normalized_agents,
                        result=result,
                        history=history,
                        log_path=log_path,
                        game_id=resolved_game_id,
                    )

                legal, reason = game.is_legal(state, player_id, move)
                if legal:
                    break

                illegal_move_counts[player_id] += 1
                error = IllegalMoveError(player_id, move, reason)
                history.append(
                    MatchEvent.create(
                        event_type=EventType.ILLEGAL_MOVE,
                        game_id=resolved_game_id,
                        turn=turn,
                        payload={
                            "player_id": player_id,
                            "move": move.to_dict() if hasattr(move, "to_dict") else to_serializable(move),
                            "reason": reason,
                            "attempt": attempt + 1,
                        },
                    )
                )
                try:
                    agent.on_illegal_move(error, observation)
                except Exception:
                    # Illegal-move callback must not break the runner.
                    pass

                attempt += 1
                if self.config.forfeit_on_illegal or attempt > self.config.max_illegal_retries:
                    winner = game.forfeit_winner(state, player_id, "illegal_move")
                    result = self._forced_result(
                        game=game,
                        game_id=resolved_game_id,
                        seed=seed,
                        state=state,
                        winner=winner,
                        reason=TerminationReason.ILLEGAL_MOVE_FORFEIT,
                        details=str(error),
                        turn=turn,
                        illegal_move_counts=illegal_move_counts,
                        move_durations_ms=move_durations_ms,
                    )
                    history.append(
                        MatchEvent.create(
                            event_type=EventType.TERMINAL,
                            game_id=resolved_game_id,
                            turn=turn,
                            payload={"result": result.to_dict()},
                        )
                    )
                    return self._finish(
                        agents=normalized_agents,
                        result=result,
                        history=history,
                        log_path=log_path,
                        game_id=resolved_game_id,
                    )

            state = game.apply_move(state, player_id, move)
            turn += 1
            history.append(
                MatchEvent.create(
                    event_type=EventType.TURN,
                    game_id=resolved_game_id,
                    turn=turn,
                    payload={
                        "player_id": player_id,
                        "observation_digest": self._observation_digest(observation),
                        "move": move.to_dict() if hasattr(move, "to_dict") else to_serializable(move),
                        "state_digest": self._state_digest(state),
                        "duration_ms": duration_ms,
                    },
                )
            )

        game_result = game.outcome(state)
        result = MatchResult(
            game_id=resolved_game_id,
            game_name=game.game_name,
            seed=seed,
            winner=game_result.winner,
            termination_reason=game_result.termination_reason,
            scores=dict(game_result.scores),
            turns=turn,
            stats=self._merge_stats(game_result.stats, illegal_move_counts, move_durations_ms),
            details=game_result.details,
            final_state_digest=self._state_digest(state),
            event_count=len(history) + 1,
            log_path=None,
        )
        history.append(
            MatchEvent.create(
                event_type=EventType.TERMINAL,
                game_id=resolved_game_id,
                turn=turn,
                payload={"result": result.to_dict()},
            )
        )
        return self._finish(
            agents=normalized_agents,
            result=result,
            history=history,
            log_path=log_path,
            game_id=resolved_game_id,
        )

    def _finish(
        self,
        *,
        agents: Mapping[PlayerId, Any],
        result: MatchResult,
        history: list[MatchEvent],
        log_path: str | Path | None,
        game_id: str,
    ) -> MatchRun:
        resolved_log_path = self._resolve_log_path(log_path=log_path, game_id=game_id)
        final_result = MatchResult(
            game_id=result.game_id,
            game_name=result.game_name,
            seed=result.seed,
            winner=result.winner,
            termination_reason=result.termination_reason,
            scores=result.scores,
            turns=result.turns,
            stats=result.stats,
            details=result.details,
            final_state_digest=result.final_state_digest,
            event_count=len(history),
            log_path=str(resolved_log_path) if resolved_log_path is not None else None,
        )
        if resolved_log_path is not None:
            write_jsonl(resolved_log_path, history)

        for agent in agents.values():
            try:
                agent.on_game_end(final_result, history)
            except Exception:
                # End-of-game hooks are optional and must not crash the runner.
                pass
        return MatchRun(result=final_result, events=history)

    def _resolve_log_path(self, *, log_path: str | Path | None, game_id: str) -> Path | None:
        if log_path is not None:
            return Path(log_path)
        if self.config.event_log_dir is None:
            return None
        return Path(self.config.event_log_dir) / f"{game_id}.jsonl"

    def _forced_result(
        self,
        *,
        game: Game[Any, Any, Any],
        game_id: str,
        seed: int,
        state: Any,
        winner: str | None,
        reason: TerminationReason,
        details: str | None,
        turn: int,
        illegal_move_counts: Mapping[str, int],
        move_durations_ms: Mapping[str, Sequence[float]],
    ) -> MatchResult:
        return MatchResult(
            game_id=game_id,
            game_name=game.game_name,
            seed=seed,
            winner=winner,
            termination_reason=reason,
            scores={},
            turns=turn,
            stats=self._merge_stats({}, illegal_move_counts, move_durations_ms),
            details=details,
            final_state_digest=self._state_digest(state),
            event_count=0,
            log_path=None,
        )

    def _merge_stats(
        self,
        base_stats: Mapping[str, Any] | None,
        illegal_move_counts: Mapping[str, int],
        move_durations_ms: Mapping[str, Sequence[float]],
    ) -> dict[str, Any]:
        merged = dict(base_stats or {})
        merged["illegal_moves"] = {player_id: int(count) for player_id, count in illegal_move_counts.items()}
        merged["move_durations_ms"] = {
            player_id: [float(duration) for duration in durations]
            for player_id, durations in move_durations_ms.items()
        }
        return merged

    def _validate_agents(self, *, player_ids: Sequence[PlayerId], agents: Mapping[PlayerId, Any]) -> None:
        missing = [player_id for player_id in player_ids if player_id not in agents]
        if missing:
            raise MatchConfigurationError(f"Missing agents for player IDs: {missing}")

    def _normalize_agents(
        self,
        *,
        player_ids: Sequence[PlayerId],
        agents: Mapping[PlayerId, Any] | Sequence[Any],
    ) -> dict[PlayerId, Any]:
        if isinstance(agents, Mapping):
            return dict(agents)
        agent_list = list(agents)
        if len(agent_list) != len(player_ids):
            raise MatchConfigurationError(
                f"Expected {len(player_ids)} agents for sequence input, received {len(agent_list)}."
            )
        return {player_id: agent for player_id, agent in zip(player_ids, agent_list, strict=True)}

    def _state_digest(self, state: Any) -> str:
        if hasattr(state, "state_digest") and callable(state.state_digest):
            return str(state.state_digest())
        if hasattr(state, "to_dict") and callable(state.to_dict):
            return digest(state.to_dict())
        return digest(to_serializable(state))

    def _observation_digest(self, observation: Any) -> str:
        if hasattr(observation, "observation_digest") and callable(observation.observation_digest):
            return str(observation.observation_digest())
        if hasattr(observation, "to_dict") and callable(observation.to_dict):
            return digest(observation.to_dict())
        return digest(to_serializable(observation))
