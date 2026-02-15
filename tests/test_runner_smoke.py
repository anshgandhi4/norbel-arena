"""Smoke tests for the generic runner + codenames integration."""

from __future__ import annotations

from codenames.codenames_game import CodenamesGame
from codenames.codenames_state import Role, Team, player_for
from framework.agents.random_agent import RandomAgent
from framework.arena import Arena
from framework.runner import MatchRunner, RunnerConfig


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
