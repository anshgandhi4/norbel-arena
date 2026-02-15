"""Persistent JSON report-card store for agent performance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping

ELO_BASE_RATING = 1000.0
ELO_K_FACTOR = 16.0


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _default_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _utc_now_iso(),
        "games": {},
    }


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _normalize_rating(value: Any, *, default: float = ELO_BASE_RATING) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _role_for_player_id(player_id: str) -> str:
    upper = player_id.upper()
    if upper.endswith("SPYMASTER"):
        return "SPYMASTER"
    if upper.endswith("OPERATIVE"):
        return "OPERATIVE"
    if "GUESSER" in upper:
        return "GUESSER"
    if "PSYCHIC" in upper:
        return "PSYCHIC"
    return "UNKNOWN"


@dataclass
class ReportCardStore:
    """Simple JSON file-backed report-card database."""

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(_default_payload())

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            raw = _default_payload()
        if "games" not in raw or not isinstance(raw["games"], dict):
            raw["games"] = {}
        return raw

    def _write(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _utc_now_iso()
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def record_match(
        self,
        *,
        game_name: str,
        agent_labels: Mapping[str, str],
        winner: str | None,
        termination_reason: str,
        turns: int,
    ) -> None:
        """Record one completed match into the persistent report card."""
        payload = self._read()
        games = payload.setdefault("games", {})
        game_data: dict[str, Any] = games.setdefault(game_name, {})

        winning_players: set[str] = set()
        winner_token = (winner or "").upper()
        if winner is None:
            winning_players = set()
        elif winner in agent_labels:
            winning_players = {winner}
        elif winner_token in {"RED", "BLUE"}:
            winning_players = {player_id for player_id in agent_labels if winner_token in player_id.upper()}
        else:
            winning_players = set()

        role_labels_by_player: dict[str, str] = {}
        for player_id, label in agent_labels.items():
            role = _role_for_player_id(player_id)
            role_label = f"{label} [{role}]"
            entry: dict[str, Any] = game_data.setdefault(
                role_label,
                {
                    "matches": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "total_turns": 0,
                    "terminations": {},
                    "seats": {},
                    "last_played": None,
                    "elo": ELO_BASE_RATING,
                },
            )
            role_labels_by_player[player_id] = role_label

            entry["matches"] += 1
            entry["total_turns"] += int(turns)
            seat_counts = entry.setdefault("seats", {})
            seat_counts[player_id] = int(seat_counts.get(player_id, 0)) + 1

            if winner is None or not winning_players:
                entry["draws"] += 1
            elif player_id in winning_players:
                entry["wins"] += 1
            else:
                entry["losses"] += 1

            terminations = entry.setdefault("terminations", {})
            terminations[termination_reason] = int(terminations.get(termination_reason, 0)) + 1
            entry["last_played"] = _utc_now_iso()

        if game_name.startswith("wavelength:"):
            self._update_wavelength_elo(
                game_data=game_data,
                role_labels_by_player=role_labels_by_player,
                winning_players=winning_players,
            )
        elif game_name.startswith("codenames:"):
            self._update_codenames_elo(
                game_data=game_data,
                role_labels_by_player=role_labels_by_player,
                winning_players=winning_players,
            )

        self._write(payload)

    def _update_codenames_elo(
        self,
        *,
        game_data: dict[str, Any],
        role_labels_by_player: Mapping[str, str],
        winning_players: set[str],
    ) -> None:
        """Update Elo for red-vs-blue Codenames seats by role."""
        self._update_two_player_role_elo(
            role="SPYMASTER",
            game_data=game_data,
            role_labels_by_player=role_labels_by_player,
            winning_players=winning_players,
        )
        self._update_two_player_role_elo(
            role="OPERATIVE",
            game_data=game_data,
            role_labels_by_player=role_labels_by_player,
            winning_players=winning_players,
        )

    def _update_wavelength_elo(
        self,
        *,
        game_data: dict[str, Any],
        role_labels_by_player: Mapping[str, str],
        winning_players: set[str],
    ) -> None:
        """Update Elo for one wavelength guesser-vs-guesser match."""
        if len(role_labels_by_player) != 2:
            return

        player_ids = list(role_labels_by_player.keys())
        player_a, player_b = player_ids[0], player_ids[1]
        label_a = role_labels_by_player[player_a]
        label_b = role_labels_by_player[player_b]
        if label_a == label_b:
            return

        entry_a = game_data.get(label_a)
        entry_b = game_data.get(label_b)
        if not isinstance(entry_a, dict) or not isinstance(entry_b, dict):
            return

        rating_a = _normalize_rating(entry_a.get("elo"))
        rating_b = _normalize_rating(entry_b.get("elo"))

        if not winning_players:
            score_a = 0.5
            score_b = 0.5
        elif player_a in winning_players and player_b in winning_players:
            score_a = 0.5
            score_b = 0.5
        elif player_a in winning_players:
            score_a = 1.0
            score_b = 0.0
        elif player_b in winning_players:
            score_a = 0.0
            score_b = 1.0
        else:
            score_a = 0.5
            score_b = 0.5

        expected_a = _elo_expected(rating_a, rating_b)
        expected_b = _elo_expected(rating_b, rating_a)
        entry_a["elo"] = rating_a + ELO_K_FACTOR * (score_a - expected_a)
        entry_b["elo"] = rating_b + ELO_K_FACTOR * (score_b - expected_b)

    def _update_two_player_role_elo(
        self,
        *,
        role: str,
        game_data: dict[str, Any],
        role_labels_by_player: Mapping[str, str],
        winning_players: set[str],
    ) -> None:
        """Update Elo between two opponents occupying the same role."""
        role_suffix = f"[{role}]"
        role_player_ids = [
            player_id
            for player_id, role_label in role_labels_by_player.items()
            if role_label.endswith(role_suffix)
        ]
        if len(role_player_ids) != 2:
            return

        player_a, player_b = role_player_ids
        label_a = role_labels_by_player[player_a]
        label_b = role_labels_by_player[player_b]
        if label_a == label_b:
            return

        entry_a = game_data.get(label_a)
        entry_b = game_data.get(label_b)
        if not isinstance(entry_a, dict) or not isinstance(entry_b, dict):
            return

        rating_a = _normalize_rating(entry_a.get("elo"))
        rating_b = _normalize_rating(entry_b.get("elo"))
        if not winning_players:
            score_a = 0.5
            score_b = 0.5
        elif player_a in winning_players and player_b in winning_players:
            score_a = 0.5
            score_b = 0.5
        elif player_a in winning_players:
            score_a = 1.0
            score_b = 0.0
        elif player_b in winning_players:
            score_a = 0.0
            score_b = 1.0
        else:
            score_a = 0.5
            score_b = 0.5

        expected_a = _elo_expected(rating_a, rating_b)
        expected_b = _elo_expected(rating_b, rating_a)
        entry_a["elo"] = rating_a + ELO_K_FACTOR * (score_a - expected_a)
        entry_b["elo"] = rating_b + ELO_K_FACTOR * (score_b - expected_b)

    def report(self) -> dict[str, Any]:
        """Return report cards enriched with derived metrics."""
        payload = self._read()
        report = {
            "version": payload.get("version", 1),
            "updated_at": payload.get("updated_at", _utc_now_iso()),
            "games": {},
        }

        games = payload.get("games", {})
        if not isinstance(games, dict):
            return report

        for game_name, game_data in games.items():
            if not isinstance(game_data, dict):
                continue
            report["games"][game_name] = {}
            for label, stats in game_data.items():
                if not isinstance(stats, dict):
                    continue
                matches = max(1, int(stats.get("matches", 0)))
                wins = int(stats.get("wins", 0))
                total_turns = int(stats.get("total_turns", 0))
                enriched = dict(stats)
                enriched["win_rate"] = wins / matches
                enriched["avg_turns"] = total_turns / matches
                report["games"][game_name][label] = enriched
        return report
