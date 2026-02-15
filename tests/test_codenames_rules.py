"""Rule-level tests for Codenames win conditions and observability."""

from __future__ import annotations

from codenames.codenames_game import CodenamesGame
from codenames.codenames_moves import Guess
from codenames.codenames_state import CardType, CodenamesState, Phase, Role, Team, player_for


def _base_state() -> CodenamesState:
    return CodenamesState(
        seed=1,
        board_words=("alpha", "beta", "gamma", "delta"),
        assignments=(CardType.RED, CardType.BLUE, CardType.NEUTRAL, CardType.ASSASSIN),
        revealed=(False, False, False, False),
        turn_team=Team.RED,
        phase=Phase.OPERATIVE_GUESSING,
        current_clue=("test", 1),
        guesses_remaining=2,
        winner=None,
        termination_reason=None,
        turn_index=0,
        last_move=None,
        red_target_count=1,
        blue_target_count=1,
    )


def test_assassin_guess_ends_game_immediately() -> None:
    game = CodenamesGame()
    state = _base_state()
    move = Guess(index=3)
    next_state = game.apply_move(state, player_for(Team.RED, Role.OPERATIVE), move)

    assert next_state.winner is Team.BLUE
    assert next_state.termination_reason == "assassin"
    assert game.is_terminal(next_state)


def test_revealing_all_team_words_triggers_win() -> None:
    game = CodenamesGame()
    state = _base_state()
    move = Guess(index=0)
    next_state = game.apply_move(state, player_for(Team.RED, Role.OPERATIVE), move)

    assert next_state.winner is Team.RED
    assert next_state.termination_reason == "all_words_revealed"
    assert game.is_terminal(next_state)


def test_spymaster_observation_has_key_and_operative_does_not() -> None:
    game = CodenamesGame()
    state = _base_state()
    spy_obs = game.observation(state, player_for(Team.RED, Role.SPYMASTER))
    operative_obs = game.observation(state, player_for(Team.RED, Role.OPERATIVE))

    assert spy_obs.assignments is not None
    assert operative_obs.assignments is None
