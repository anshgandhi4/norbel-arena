"""Rule-level tests for Wavelength win conditions and observability."""

from __future__ import annotations

from dataclasses import replace

from wavelength.wavelength_game import WavelengthGame
from wavelength.wavelength_moves import (
    AskCategory,
    ChooseNumber,
    GiveAnswer,
    SubmitFinalEstimate,
    SubmitGuess,
)
from wavelength.wavelength_state import (
    GUESSER_ONE_PLAYER_ID,
    GUESSER_TWO_PLAYER_ID,
    PSYCHIC_PLAYER_ID,
    Phase,
)


def test_closest_final_estimate_wins() -> None:
    game = WavelengthGame()
    state = game.new_game(seed=7, config={"max_rounds": 3})

    state = game.apply_move(state, PSYCHIC_PLAYER_ID, ChooseNumber(value=73))

    for _ in range(3):
        state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, AskCategory(category="car brand"))
        state = game.apply_move(state, PSYCHIC_PLAYER_ID, GiveAnswer(answer="track supercar"))
        state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, SubmitGuess(value=70))

        state = game.apply_move(state, GUESSER_TWO_PLAYER_ID, AskCategory(category="snack food"))
        state = game.apply_move(state, PSYCHIC_PLAYER_ID, GiveAnswer(answer="plain cracker"))
        state = game.apply_move(state, GUESSER_TWO_PLAYER_ID, SubmitGuess(value=35))

    state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, SubmitFinalEstimate(value=72))
    state = game.apply_move(state, GUESSER_TWO_PLAYER_ID, SubmitFinalEstimate(value=41))

    assert game.is_terminal(state)
    assert state.winner == GUESSER_ONE_PLAYER_ID
    assert state.termination_reason == "closest_final_estimate"


def test_psychic_sees_target_number_but_guesser_does_not() -> None:
    game = WavelengthGame()
    state = game.new_game(seed=11, config={})
    state = game.apply_move(state, PSYCHIC_PLAYER_ID, ChooseNumber(value=88))

    psychic_obs = game.observation(state, PSYCHIC_PLAYER_ID)
    guesser_obs = game.observation(state, GUESSER_ONE_PLAYER_ID)

    assert psychic_obs.target_number == 88
    assert guesser_obs.target_number is None


def test_probe_history_omits_target_delta_fields() -> None:
    game = WavelengthGame()
    state = game.new_game(seed=12, config={"max_rounds": 1})
    state = game.apply_move(state, PSYCHIC_PLAYER_ID, ChooseNumber(value=88))
    state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, AskCategory(category="weather"))
    state = game.apply_move(state, PSYCHIC_PLAYER_ID, GiveAnswer(answer="stormy"))
    state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, SubmitGuess(value=81))

    latest_history = state.history[-1]
    assert latest_history["guess"] == 81
    assert "delta" not in latest_history
    assert "abs_error" not in latest_history
    assert "direction" not in latest_history

    guesser_obs = game.observation(state, GUESSER_TWO_PLAYER_ID)
    latest_visible_history = guesser_obs.history[-1]
    assert "delta" not in latest_visible_history
    assert "abs_error" not in latest_visible_history
    assert "direction" not in latest_visible_history


def test_final_estimates_are_hidden_from_other_guesser_until_terminal() -> None:
    game = WavelengthGame()
    state = game.new_game(seed=17, config={"max_rounds": 1})
    state = game.apply_move(state, PSYCHIC_PLAYER_ID, ChooseNumber(value=42))

    state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, AskCategory(category="weather"))
    state = game.apply_move(state, PSYCHIC_PLAYER_ID, GiveAnswer(answer="stormy"))
    state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, SubmitGuess(value=38))
    state = game.apply_move(state, GUESSER_TWO_PLAYER_ID, AskCategory(category="sports car"))
    state = game.apply_move(state, PSYCHIC_PLAYER_ID, GiveAnswer(answer="balanced coupe"))
    state = game.apply_move(state, GUESSER_TWO_PLAYER_ID, SubmitGuess(value=55))

    state = game.apply_move(state, GUESSER_ONE_PLAYER_ID, SubmitFinalEstimate(value=41))
    guesser_two_obs = game.observation(state, GUESSER_TWO_PLAYER_ID)

    assert guesser_two_obs.final_estimates[GUESSER_ONE_PLAYER_ID] is None
    assert guesser_two_obs.final_estimates[GUESSER_TWO_PLAYER_ID] is None
    assert guesser_two_obs.last_move is not None
    assert guesser_two_obs.last_move.get("type") == "SubmitFinalEstimate"
    assert "value" not in guesser_two_obs.last_move


def test_final_estimate_requires_all_rounds_complete() -> None:
    game = WavelengthGame()
    state = game.new_game(seed=23, config={"max_rounds": 3})
    state = game.apply_move(state, PSYCHIC_PLAYER_ID, ChooseNumber(value=58))

    # Simulate a malformed transition into FINAL_ESTIMATE before all probes are complete.
    state = replace(
        state,
        phase=Phase.FINAL_ESTIMATE,
        active_player=GUESSER_ONE_PLAYER_ID,
        active_guesser=None,
    )
    legal, reason = game.is_legal(
        state,
        GUESSER_ONE_PLAYER_ID,
        SubmitFinalEstimate(value=59),
    )

    assert legal is False
    assert reason == "Final estimate is only available after both guessers complete all rounds."
