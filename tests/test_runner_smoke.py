"""Smoke tests for the generic runner + codenames integration."""

from __future__ import annotations

from codenames.codenames_game import CodenamesGame
from codenames.codenames_state import Role, Team, player_for
from framework.agents.random_agent import RandomAgent
from framework.arena import Arena
from framework.runner import MatchRunner, RunnerConfig
from wavelength.wavelength_game import WavelengthGame
from wavelength.wavelength_state import GUESSER_ONE_PLAYER_ID, GUESSER_TWO_PLAYER_ID, PSYCHIC_PLAYER_ID


def _build_random_agents() -> dict[str, RandomAgent]:
    return {
        player_for(Team.RED, Role.SPYMASTER): RandomAgent("random-red-spymaster"),
        player_for(Team.RED, Role.OPERATIVE): RandomAgent("random-red-operative"),
        player_for(Team.BLUE, Role.SPYMASTER): RandomAgent("random-blue-spymaster"),
        player_for(Team.BLUE, Role.OPERATIVE): RandomAgent("random-blue-operative"),
    }


def test_three_seeded_matches_complete_without_crashes() -> None:
    runner = MatchRunner(RunnerConfig(max_turns=200))
    arena = Arena(runner=runner)

    summary = arena.run_series(
        game_factory=CodenamesGame,
        agents_factory=lambda _: _build_random_agents(),
        seeds=[11, 12, 13],
    )

    assert len(summary.results) == 3
    for result in summary.results:
        assert result.game_name == "codenames"
        assert result.termination_reason is not None


def _build_wavelength_random_agents() -> dict[str, RandomAgent]:
    return {
        PSYCHIC_PLAYER_ID: RandomAgent("random-psychic"),
        GUESSER_ONE_PLAYER_ID: RandomAgent("random-guesser-one"),
        GUESSER_TWO_PLAYER_ID: RandomAgent("random-guesser-two"),
    }


def test_three_seeded_wavelength_matches_complete_without_crashes() -> None:
    runner = MatchRunner(RunnerConfig(max_turns=200))
    arena = Arena(runner=runner)

    summary = arena.run_series(
        game_factory=WavelengthGame,
        agents_factory=lambda _: _build_wavelength_random_agents(),
        seeds=[21, 22, 23],
    )

    assert len(summary.results) == 3
    for result in summary.results:
        assert result.game_name == "wavelength"
        assert result.termination_reason is not None
