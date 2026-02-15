"""Arena orchestration for single matches, series, and round-robins."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .game import Game
from .result import MatchResult
from .runner import MatchRunner
from .serialize import to_serializable


@dataclass(frozen=True)
class ArenaSummary:
    """Aggregated output from a set of matches."""

    results: list[MatchResult]
    wins: dict[str, int]
    win_rates: dict[str, float]
    draws: int
    ratings: dict[str, float] = field(default_factory=dict)
    competitor_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "results": [result.to_dict() for result in self.results],
            "wins": dict(self.wins),
            "win_rates": dict(self.win_rates),
            "draws": self.draws,
            "ratings": dict(self.ratings),
            "competitor_stats": to_serializable(self.competitor_stats),
        }


class Arena:
    """High-level interface for running many matches."""

    def __init__(self, runner: MatchRunner | None = None):
        self.runner = runner or MatchRunner()
        self.last_events: list[Any] = []

    def run_match(
        self,
        game: Game[Any, Any, Any],
        agents: Mapping[str, Any] | Sequence[Any],
        seed: int,
        game_config: dict[str, Any] | None = None,
        *,
        game_id: str | None = None,
        log_path: str | Path | None = None,
    ) -> MatchResult:
        """Run one match and return a `MatchResult`."""
        run = self.runner.run_match(
            game=game,
            agents=agents,
            seed=seed,
            game_config=game_config,
            game_id=game_id,
            log_path=log_path,
        )
        self.last_events = run.events
        return run.result

    def run_series(
        self,
        game_factory: Callable[[], Game[Any, Any, Any]],
        agents_factory: (
            Mapping[str, Any]
            | Sequence[Any]
            | Callable[[int], Mapping[str, Any] | Sequence[Any]]
        ),
        seeds: Sequence[int],
        game_config: dict[str, Any] | None = None,
    ) -> ArenaSummary:
        """Run a seeded series and aggregate win rates."""
        results: list[MatchResult] = []
        for seed in seeds:
            game = game_factory()
            agents = agents_factory(seed) if callable(agents_factory) else agents_factory
            result = self.run_match(game=game, agents=agents, seed=seed, game_config=game_config)
            results.append(result)
        return self._summarize_results(results)

    def run_round_robin(
        self,
        game_factory: Callable[[], Game[Any, Any, Any]],
        competitors: Mapping[str, Callable[[str], Any]],
        seeds: Sequence[int],
        game_config: dict[str, Any] | None = None,
    ) -> ArenaSummary:
        """
        Run a simple round-robin.

        Competitor factories are assigned to RED/BLUE seats using player IDs when
        possible (fallback: split sorted players in half).
        """
        if not seeds:
            return ArenaSummary(results=[], wins={}, win_rates={}, draws=0, ratings={}, competitor_stats={})

        sample_game = game_factory()
        sample_state = sample_game.new_game(seed=seeds[0], config=game_config or {})
        player_ids = list(sample_game.player_ids(sample_state))
        red_slots, blue_slots = self._infer_team_slots(player_ids)

        results: list[MatchResult] = []
        competitor_stats = {
            name: {"wins": 0, "losses": 0, "draws": 0}
            for name in competitors.keys()
        }
        ratings = {name: 1000.0 for name in competitors.keys()}

        for first, second in combinations(competitors.keys(), 2):
            seatings = [(first, second), (second, first)]
            for red_competitor, blue_competitor in seatings:
                for seed in seeds:
                    game = game_factory()
                    agents = self._build_match_agents(
                        player_ids=player_ids,
                        red_slots=red_slots,
                        blue_slots=blue_slots,
                        red_competitor=red_competitor,
                        blue_competitor=blue_competitor,
                        competitors=competitors,
                    )
                    result = self.run_match(
                        game=game,
                        agents=agents,
                        seed=seed,
                        game_config=game_config,
                    )
                    results.append(result)

                    mapped_winner = self._map_winner_to_competitor(
                        winner=result.winner,
                        red_competitor=red_competitor,
                        blue_competitor=blue_competitor,
                    )
                    if mapped_winner is None:
                        competitor_stats[red_competitor]["draws"] += 1
                        competitor_stats[blue_competitor]["draws"] += 1
                        self._update_elo(ratings, red_competitor, blue_competitor, score_red=0.5)
                    elif mapped_winner == red_competitor:
                        competitor_stats[red_competitor]["wins"] += 1
                        competitor_stats[blue_competitor]["losses"] += 1
                        self._update_elo(ratings, red_competitor, blue_competitor, score_red=1.0)
                    else:
                        competitor_stats[blue_competitor]["wins"] += 1
                        competitor_stats[red_competitor]["losses"] += 1
                        self._update_elo(ratings, red_competitor, blue_competitor, score_red=0.0)

        summary = self._summarize_results(results)
        return ArenaSummary(
            results=summary.results,
            wins=summary.wins,
            win_rates=summary.win_rates,
            draws=summary.draws,
            ratings=ratings,
            competitor_stats=competitor_stats,
        )

    def _build_match_agents(
        self,
        *,
        player_ids: Sequence[str],
        red_slots: Sequence[str],
        blue_slots: Sequence[str],
        red_competitor: str,
        blue_competitor: str,
        competitors: Mapping[str, Callable[[str], Any]],
    ) -> dict[str, Any]:
        agents: dict[str, Any] = {}
        for player_id in player_ids:
            if player_id in red_slots:
                agents[player_id] = competitors[red_competitor](player_id)
            else:
                agents[player_id] = competitors[blue_competitor](player_id)
        return agents

    def _map_winner_to_competitor(
        self,
        *,
        winner: str | None,
        red_competitor: str,
        blue_competitor: str,
    ) -> str | None:
        if winner is None:
            return None
        value = winner.upper()
        if value == "RED":
            return red_competitor
        if value == "BLUE":
            return blue_competitor
        return None

    def _infer_team_slots(self, player_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        red_slots = [player_id for player_id in player_ids if "RED" in player_id.upper()]
        blue_slots = [player_id for player_id in player_ids if "BLUE" in player_id.upper()]
        if red_slots and blue_slots:
            return red_slots, blue_slots

        ordered = sorted(player_ids)
        split = len(ordered) // 2
        return ordered[:split], ordered[split:]

    def _summarize_results(self, results: Sequence[MatchResult]) -> ArenaSummary:
        wins: dict[str, int] = {}
        draws = 0
        for result in results:
            if result.winner is None:
                draws += 1
                continue
            wins[result.winner] = wins.get(result.winner, 0) + 1
        total = len(results) if results else 1
        win_rates = {winner: count / total for winner, count in wins.items()}
        return ArenaSummary(
            results=list(results),
            wins=wins,
            win_rates=win_rates,
            draws=draws,
            ratings={},
            competitor_stats={},
        )

    def _update_elo(self, ratings: dict[str, float], red: str, blue: str, score_red: float, k: float = 16.0) -> None:
        red_rating = ratings[red]
        blue_rating = ratings[blue]
        expected_red = 1.0 / (1.0 + 10.0 ** ((blue_rating - red_rating) / 400.0))
        expected_blue = 1.0 - expected_red
        ratings[red] = red_rating + k * (score_red - expected_red)
        ratings[blue] = blue_rating + k * ((1.0 - score_red) - expected_blue)
