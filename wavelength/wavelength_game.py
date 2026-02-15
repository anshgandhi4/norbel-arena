"""Wavelength game implementation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from framework.game import Game, LegalMovesSpec
from framework.result import MatchResult, TerminationReason

from .wavelength_moves import (
    AskCategory,
    ChooseNumber,
    GiveAnswer,
    MoveType,
    SubmitFinalEstimate,
    SubmitGuess,
    move_from_dict,
)
from .wavelength_observation import WavelengthObservation
from .wavelength_state import (
    GUESSER_ONE_PLAYER_ID,
    GUESSER_PLAYER_IDS,
    GUESSER_TWO_PLAYER_ID,
    PSYCHIC_PLAYER_ID,
    Phase,
    Role,
    WavelengthState,
    role_for_player,
)

DEFAULT_CATEGORY_SAMPLES: tuple[str, ...] = (
    "car brand",
    "vacation destination",
    "movie franchise",
    "coffee shop chain",
    "snack food",
)
DEFAULT_ANSWER_SAMPLES: tuple[str, ...] = (
    "budget sedan",
    "track supercar",
    "mountain cabin",
    "space opera",
    "artisanal espresso",
)
DEFAULT_NUMERIC_SAMPLES: tuple[int, ...] = (1, 15, 35, 50, 65, 85, 100)


class WavelengthGame(Game[WavelengthState, Any, WavelengthObservation]):
    """Three-player Wavelength: one psychic and two guessers."""

    game_name = "wavelength"

    def __init__(self, default_config: dict[str, Any] | None = None):
        self.default_config = default_config or {}

    def new_game(self, seed: int, config: dict[str, Any] | None = None) -> WavelengthState:
        """Create a deterministic initial Wavelength state."""
        cfg = dict(self.default_config)
        cfg.update(config or {})
        max_rounds = int(cfg.get("max_rounds", 3))
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")

        return WavelengthState(
            seed=seed,
            max_rounds=max_rounds,
            phase=Phase.CHOOSE_NUMBER,
            active_player=PSYCHIC_PLAYER_ID,
            active_guesser=GUESSER_ONE_PLAYER_ID,
            target_number=None,
            pending_exchange=None,
            history=tuple(),
            guess_counts={
                GUESSER_ONE_PLAYER_ID: 0,
                GUESSER_TWO_PLAYER_ID: 0,
            },
            final_estimates={
                GUESSER_ONE_PLAYER_ID: None,
                GUESSER_TWO_PLAYER_ID: None,
            },
            winner=None,
            termination_reason=None,
            turn_index=0,
            last_move=None,
        )

    def player_ids(self, state: WavelengthState) -> Sequence[str]:
        """Return canonical Wavelength seats."""
        return (PSYCHIC_PLAYER_ID, GUESSER_ONE_PLAYER_ID, GUESSER_TWO_PLAYER_ID)

    def role_for_player(self, state: WavelengthState, player_id: str) -> str | None:
        """Return role for player ID."""
        return role_for_player(player_id).value

    def current_player(self, state: WavelengthState) -> str:
        """Return active player for the current phase."""
        return state.active_player

    def legal_moves(self, state: WavelengthState, player_id: str) -> LegalMovesSpec:
        """Return legal move list/spec for the active player."""
        if self.is_terminal(state):
            return []
        if player_id != state.active_player:
            return []

        if state.phase is Phase.CHOOSE_NUMBER:
            return self._numeric_move_spec(
                move_type=MoveType.CHOOSE_NUMBER,
                template_field="value",
                sample_moves=[ChooseNumber(value=value) for value in DEFAULT_NUMERIC_SAMPLES],
            )
        if state.phase is Phase.CATEGORY_PROMPT:
            return self._text_move_spec(
                move_type=MoveType.ASK_CATEGORY,
                template_field="category",
                sample_moves=[AskCategory(category=value) for value in DEFAULT_CATEGORY_SAMPLES],
            )
        if state.phase is Phase.PSYCHIC_RESPONSE:
            return self._text_move_spec(
                move_type=MoveType.GIVE_ANSWER,
                template_field="answer",
                sample_moves=[GiveAnswer(answer=value) for value in DEFAULT_ANSWER_SAMPLES],
            )
        if state.phase is Phase.NUMERIC_GUESS:
            return self._numeric_move_spec(
                move_type=MoveType.SUBMIT_GUESS,
                template_field="value",
                sample_moves=[SubmitGuess(value=value) for value in DEFAULT_NUMERIC_SAMPLES],
            )

        return self._numeric_move_spec(
            move_type=MoveType.SUBMIT_FINAL_ESTIMATE,
            template_field="value",
            sample_moves=[SubmitFinalEstimate(value=value) for value in DEFAULT_NUMERIC_SAMPLES],
        )

    def is_legal(self, state: WavelengthState, player_id: str, move: Any) -> tuple[bool, str | None]:
        """Validate legality of move for current state and player."""
        if self.is_terminal(state):
            return False, "Game is already terminal."
        if player_id != state.active_player:
            return False, f"It is not {player_id}'s turn."

        if state.phase is Phase.CHOOSE_NUMBER:
            if player_id != PSYCHIC_PLAYER_ID:
                return False, "Only PSYCHIC can choose the hidden number."
            if not isinstance(move, ChooseNumber):
                return False, "Expected ChooseNumber during CHOOSE_NUMBER phase."
            return True, None

        if state.phase is Phase.CATEGORY_PROMPT:
            if not isinstance(move, AskCategory):
                return False, "Expected AskCategory during CATEGORY_PROMPT phase."
            if len(move.category.strip()) == 0:
                return False, "Category must be non-empty."
            return True, None

        if state.phase is Phase.PSYCHIC_RESPONSE:
            if player_id != PSYCHIC_PLAYER_ID:
                return False, "Only PSYCHIC can provide the category answer."
            if not isinstance(move, GiveAnswer):
                return False, "Expected GiveAnswer during PSYCHIC_RESPONSE phase."
            if state.pending_exchange is None:
                return False, "No pending category prompt to answer."
            return True, None

        if state.phase is Phase.NUMERIC_GUESS:
            if not isinstance(move, SubmitGuess):
                return False, "Expected SubmitGuess during NUMERIC_GUESS phase."
            if state.pending_exchange is None or state.target_number is None:
                return False, "Cannot submit numeric guess without a pending prompt and chosen target number."
            return True, None

        if not isinstance(move, SubmitFinalEstimate):
            return False, "Expected SubmitFinalEstimate during FINAL_ESTIMATE phase."
        if any(state.guess_counts.get(guesser, 0) < state.max_rounds for guesser in GUESSER_PLAYER_IDS):
            return False, "Final estimate is only available after both guessers complete all rounds."
        if player_id not in GUESSER_PLAYER_IDS:
            return False, "Only guessers can submit final estimates."
        if state.final_estimates.get(player_id) is not None:
            return False, "Final estimate already submitted for this player."
        return True, None

    def apply_move(self, state: WavelengthState, player_id: str, move: Any) -> WavelengthState:
        """Apply a legal move and return next immutable state."""
        legal, reason = self.is_legal(state, player_id, move)
        if not legal:
            raise ValueError(f"Illegal move: {reason}")

        if isinstance(move, ChooseNumber):
            return replace(
                state,
                target_number=move.value,
                phase=Phase.CATEGORY_PROMPT,
                active_player=GUESSER_ONE_PLAYER_ID,
                active_guesser=GUESSER_ONE_PLAYER_ID,
                turn_index=state.turn_index + 1,
                last_move=move.to_dict(),
            )

        if isinstance(move, AskCategory):
            assert state.active_guesser is not None
            pending_exchange = {
                "round": state.guess_counts[state.active_guesser] + 1,
                "guesser_id": state.active_guesser,
                "category": move.category,
                "answer": None,
            }
            return replace(
                state,
                pending_exchange=pending_exchange,
                phase=Phase.PSYCHIC_RESPONSE,
                active_player=PSYCHIC_PLAYER_ID,
                turn_index=state.turn_index + 1,
                last_move=move.to_dict(),
            )

        if isinstance(move, GiveAnswer):
            pending_exchange = dict(state.pending_exchange or {})
            pending_exchange["answer"] = move.answer
            return replace(
                state,
                pending_exchange=pending_exchange,
                phase=Phase.NUMERIC_GUESS,
                active_player=state.active_guesser or GUESSER_ONE_PLAYER_ID,
                turn_index=state.turn_index + 1,
                last_move=move.to_dict(),
            )

        if isinstance(move, SubmitGuess):
            if state.target_number is None or state.pending_exchange is None:
                raise ValueError("Numeric guess requires a chosen target number and pending category exchange.")

            active_guesser = state.active_guesser or player_id
            completed_exchange = dict(state.pending_exchange)
            completed_exchange.update(
                {
                    "guess": move.value,
                }
            )

            history = list(state.history)
            history.append(completed_exchange)
            guess_counts = dict(state.guess_counts)
            guess_counts[active_guesser] = guess_counts.get(active_guesser, 0) + 1

            next_phase, next_player, next_guesser = self._next_after_probe(
                guess_counts=guess_counts,
                most_recent_guesser=active_guesser,
                max_rounds=state.max_rounds,
            )

            return replace(
                state,
                history=tuple(history),
                pending_exchange=None,
                guess_counts=guess_counts,
                phase=next_phase,
                active_player=next_player,
                active_guesser=next_guesser,
                turn_index=state.turn_index + 1,
                last_move=move.to_dict(),
            )

        if isinstance(move, SubmitFinalEstimate):
            final_estimates = dict(state.final_estimates)
            final_estimates[player_id] = move.value
            redacted_last_move = {
                "type": MoveType.SUBMIT_FINAL_ESTIMATE.value,
                "player_id": player_id,
            }
            other_guesser = self._other_guesser(player_id)
            if final_estimates.get(other_guesser) is None:
                return replace(
                    state,
                    final_estimates=final_estimates,
                    phase=Phase.FINAL_ESTIMATE,
                    active_player=other_guesser,
                    active_guesser=None,
                    turn_index=state.turn_index + 1,
                    last_move=redacted_last_move,
                )

            winner, reason = self._resolve_final_winner(target=state.target_number, final_estimates=final_estimates)
            return replace(
                state,
                final_estimates=final_estimates,
                winner=winner,
                termination_reason=reason,
                turn_index=state.turn_index + 1,
                last_move=redacted_last_move,
            )

        raise ValueError(f"Unsupported move type: {type(move)!r}")

    def is_terminal(self, state: WavelengthState) -> bool:
        """Return whether the state has reached a terminal condition."""
        return state.winner is not None or state.termination_reason is not None

    def outcome(self, state: WavelengthState) -> MatchResult:
        """Build structured outcome from terminal (or adjudicated) state."""
        scores: dict[str, float] = {}
        target = state.target_number
        for guesser_id in GUESSER_PLAYER_IDS:
            estimate = state.final_estimates.get(guesser_id)
            if estimate is None or target is None:
                continue
            scores[guesser_id] = float(max(0, 100 - abs(estimate - target)))

        if state.winner is not None:
            reason = TerminationReason.NORMAL_WIN
        else:
            reason = TerminationReason.DRAW

        return MatchResult(
            game_id="",
            game_name=self.game_name,
            seed=state.seed,
            winner=state.winner,
            termination_reason=reason,
            scores=scores,
            turns=state.turn_index,
            stats={
                "target_number": state.target_number,
                "guess_counts": dict(state.guess_counts),
                "history_length": len(state.history),
                "final_estimates": dict(state.final_estimates),
            },
            details=state.termination_reason,
            final_state_digest=state.state_digest(),
            event_count=0,
            log_path=None,
        )

    def observation(self, state: WavelengthState, player_id: str) -> WavelengthObservation:
        """Return partial observation for seat."""
        role = role_for_player(player_id)
        target_number = state.target_number if role is Role.PSYCHIC else None

        final_estimates: dict[str, int | None] = dict(state.final_estimates)
        if role is Role.GUESSER and not self.is_terminal(state):
            for guesser in GUESSER_PLAYER_IDS:
                if guesser != player_id:
                    final_estimates[guesser] = None

        last_move = state.last_move
        if (
            role is Role.GUESSER
            and not self.is_terminal(state)
            and isinstance(last_move, dict)
            and last_move.get("type") == MoveType.SUBMIT_FINAL_ESTIMATE.value
            and last_move.get("player_id") != player_id
        ):
            last_move = {
                "type": MoveType.SUBMIT_FINAL_ESTIMATE.value,
                "player_id": last_move.get("player_id"),
            }

        return WavelengthObservation(
            player_id=player_id,
            role=role,
            phase=state.phase,
            current_player=state.active_player,
            current_round=state.current_round(),
            max_rounds=state.max_rounds,
            turn_index=state.turn_index,
            target_number=target_number,
            history=state.history,
            pending_exchange=state.pending_exchange,
            guess_counts=dict(state.guess_counts),
            final_estimates=final_estimates,
            winner=state.winner,
            termination_reason=state.termination_reason,
            last_move=last_move,
        )

    def render(self, state: WavelengthState, player_id: str | None = None) -> str:
        """Render state for debugging."""
        can_see_target = player_id == PSYCHIC_PLAYER_ID
        target = state.target_number if can_see_target else "?"
        header = (
            f"phase={state.phase.value} active={state.active_player} round={state.current_round()}/{state.max_rounds} "
            f"target={target} winner={state.winner} reason={state.termination_reason}"
        )
        return header + "\n" + "\n".join(
            [
                f"history={len(state.history)} entries",
                f"pending={state.pending_exchange}",
                f"guess_counts={state.guess_counts}",
                f"final_estimates={state.final_estimates}",
            ]
        )

    def parse_move(self, data: Mapping[str, Any]) -> Any:
        """Parse move payload into move object."""
        return move_from_dict(data)

    def forfeit_winner(self, state: WavelengthState, offending_player_id: str, reason: str) -> str | None:
        """Opponent guesser wins if one guesser forfeits; psychic forfeits yield no winner."""
        if offending_player_id == GUESSER_ONE_PLAYER_ID:
            return GUESSER_TWO_PLAYER_ID
        if offending_player_id == GUESSER_TWO_PLAYER_ID:
            return GUESSER_ONE_PLAYER_ID
        return None

    def _next_after_probe(
        self,
        *,
        guess_counts: Mapping[str, int],
        most_recent_guesser: str,
        max_rounds: int,
    ) -> tuple[Phase, str, str | None]:
        if all(guess_counts.get(guesser, 0) >= max_rounds for guesser in GUESSER_PLAYER_IDS):
            return (Phase.FINAL_ESTIMATE, GUESSER_ONE_PLAYER_ID, None)

        next_guesser = self._other_guesser(most_recent_guesser)
        if guess_counts.get(next_guesser, 0) >= max_rounds:
            next_guesser = most_recent_guesser
        return (Phase.CATEGORY_PROMPT, next_guesser, next_guesser)

    def _resolve_final_winner(
        self,
        *,
        target: int | None,
        final_estimates: Mapping[str, int | None],
    ) -> tuple[str | None, str]:
        if target is None:
            return None, "missing_target"

        guess_one = final_estimates.get(GUESSER_ONE_PLAYER_ID)
        guess_two = final_estimates.get(GUESSER_TWO_PLAYER_ID)
        if guess_one is None or guess_two is None:
            return None, "incomplete_final_estimates"

        error_one = abs(guess_one - target)
        error_two = abs(guess_two - target)
        if error_one < error_two:
            return GUESSER_ONE_PLAYER_ID, "closest_final_estimate"
        if error_two < error_one:
            return GUESSER_TWO_PLAYER_ID, "closest_final_estimate"
        return None, "closest_tie"

    def _other_guesser(self, player_id: str) -> str:
        if player_id == GUESSER_ONE_PLAYER_ID:
            return GUESSER_TWO_PLAYER_ID
        if player_id == GUESSER_TWO_PLAYER_ID:
            return GUESSER_ONE_PLAYER_ID
        raise ValueError(f"Player is not a guesser: {player_id!r}")

    def _numeric_move_spec(
        self,
        *,
        move_type: MoveType,
        template_field: str,
        sample_moves: Sequence[Any],
    ) -> dict[str, Any]:
        return {
            "enumerable": False,
            "phase": move_type.value,
            "template": {"type": move_type.value, template_field: "int[1..100]"},
            "allowed": {move_type.value: {"min": 1, "max": 100}},
            "sample_moves": list(sample_moves),
        }

    def _text_move_spec(
        self,
        *,
        move_type: MoveType,
        template_field: str,
        sample_moves: Sequence[Any],
    ) -> dict[str, Any]:
        return {
            "enumerable": False,
            "phase": move_type.value,
            "template": {"type": move_type.value, template_field: "non-empty string"},
            "allowed": {move_type.value: True},
            "sample_moves": list(sample_moves),
        }
