"""Unit tests for report-card aggregation."""

from __future__ import annotations

from pathlib import Path

from server.report_cards import ReportCardStore


def _store(tmp_path: Path) -> ReportCardStore:
    return ReportCardStore(path=tmp_path / "report_cards.json")


def test_record_match_splits_entries_by_role(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent_labels = {
        "RED_SPYMASTER": "openai:gpt-4o-mini",
        "BLUE_SPYMASTER": "openai:gpt-4o-mini",
        "RED_OPERATIVE": "anthropic:claude-sonnet-4-20250514",
        "BLUE_OPERATIVE": "perplexity:sonar",
    }

    store.record_match(
        game_name="codenames:operator_eval",
        agent_labels=agent_labels,
        winner="RED",
        termination_reason="normal_win",
        turns=11,
    )

    report = store.report()
    game = report["games"]["codenames:operator_eval"]

    spymaster_entry = game["openai:gpt-4o-mini [SPYMASTER]"]
    assert spymaster_entry["matches"] == 2
    assert spymaster_entry["wins"] == 1
    assert spymaster_entry["losses"] == 1
    assert spymaster_entry["draws"] == 0
    assert spymaster_entry["seats"]["RED_SPYMASTER"] == 1
    assert spymaster_entry["seats"]["BLUE_SPYMASTER"] == 1

    red_operative_entry = game["anthropic:claude-sonnet-4-20250514 [OPERATIVE]"]
    assert red_operative_entry["matches"] == 1
    assert red_operative_entry["wins"] == 1
    assert red_operative_entry["losses"] == 0

    blue_operative_entry = game["perplexity:sonar [OPERATIVE]"]
    assert blue_operative_entry["matches"] == 1
    assert blue_operative_entry["wins"] == 0
    assert blue_operative_entry["losses"] == 1


def test_record_match_separates_buckets_by_evaluation_mode(tmp_path: Path) -> None:
    store = _store(tmp_path)
    operator_eval_labels = {
        "RED_SPYMASTER": "openai:gpt-4o-mini",
        "BLUE_SPYMASTER": "openai:gpt-4o-mini",
        "RED_OPERATIVE": "anthropic:claude-sonnet-4-20250514",
        "BLUE_OPERATIVE": "perplexity:sonar",
    }
    spymaster_eval_labels = {
        "RED_SPYMASTER": "local:llama3.1",
        "BLUE_SPYMASTER": "openai:gpt-4o-mini",
        "RED_OPERATIVE": "openai:gpt-4o-mini",
        "BLUE_OPERATIVE": "openai:gpt-4o-mini",
    }

    store.record_match(
        game_name="codenames:operator_eval",
        agent_labels=operator_eval_labels,
        winner="RED",
        termination_reason="normal_win",
        turns=10,
    )
    store.record_match(
        game_name="codenames:spymaster_eval",
        agent_labels=spymaster_eval_labels,
        winner="BLUE",
        termination_reason="normal_win",
        turns=12,
    )

    report = store.report()
    assert "codenames:operator_eval" in report["games"]
    assert "codenames:spymaster_eval" in report["games"]

    operator_game = report["games"]["codenames:operator_eval"]
    spymaster_game = report["games"]["codenames:spymaster_eval"]
    assert "openai:gpt-4o-mini [SPYMASTER]" in operator_game
    assert "openai:gpt-4o-mini [SPYMASTER]" in spymaster_game
    assert "local:llama3.1 [SPYMASTER]" in spymaster_game
    assert "local:llama3.1 [SPYMASTER]" not in operator_game

    assert operator_game["openai:gpt-4o-mini [SPYMASTER]"]["matches"] == 2
    assert spymaster_game["openai:gpt-4o-mini [SPYMASTER]"]["matches"] == 1


def test_record_match_supports_wavelength_guesser_role(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labels = {
        "GUESSER_ONE": "openai:gpt-4o-mini",
        "GUESSER_TWO": "anthropic:claude-sonnet-4-20250514",
    }

    store.record_match(
        game_name="wavelength:guesser_eval",
        agent_labels=labels,
        winner="GUESSER_TWO",
        termination_reason="normal_win",
        turns=18,
    )

    report = store.report()
    game = report["games"]["wavelength:guesser_eval"]

    guesser_one = game["openai:gpt-4o-mini [GUESSER]"]
    assert guesser_one["matches"] == 1
    assert guesser_one["wins"] == 0
    assert guesser_one["losses"] == 1

    guesser_two = game["anthropic:claude-sonnet-4-20250514 [GUESSER]"]
    assert guesser_two["matches"] == 1
    assert guesser_two["wins"] == 1
    assert guesser_two["losses"] == 0
    assert isinstance(guesser_one["elo"], float)
    assert isinstance(guesser_two["elo"], float)
    assert guesser_two["elo"] > guesser_one["elo"]


def test_wavelength_elo_updates_across_multiple_matches(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labels = {
        "GUESSER_ONE": "openai:gpt-4o-mini",
        "GUESSER_TWO": "anthropic:claude-sonnet-4-20250514",
    }

    store.record_match(
        game_name="wavelength:guesser_eval",
        agent_labels=labels,
        winner="GUESSER_TWO",
        termination_reason="normal_win",
        turns=20,
    )
    first = store.report()["games"]["wavelength:guesser_eval"]
    first_g1 = first["openai:gpt-4o-mini [GUESSER]"]["elo"]
    first_g2 = first["anthropic:claude-sonnet-4-20250514 [GUESSER]"]["elo"]

    store.record_match(
        game_name="wavelength:guesser_eval",
        agent_labels=labels,
        winner="GUESSER_TWO",
        termination_reason="normal_win",
        turns=21,
    )
    second = store.report()["games"]["wavelength:guesser_eval"]
    second_g1 = second["openai:gpt-4o-mini [GUESSER]"]["elo"]
    second_g2 = second["anthropic:claude-sonnet-4-20250514 [GUESSER]"]["elo"]

    assert second_g2 > first_g2
    assert second_g1 < first_g1


def test_codenames_elo_updates_for_spymaster_and_operative_roles(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labels = {
        "RED_SPYMASTER": "openai:gpt-4o-mini",
        "BLUE_SPYMASTER": "anthropic:claude-sonnet-4-20250514",
        "RED_OPERATIVE": "perplexity:sonar",
        "BLUE_OPERATIVE": "local:llama3.1",
    }

    store.record_match(
        game_name="codenames:operator_eval",
        agent_labels=labels,
        winner="RED",
        termination_reason="normal_win",
        turns=13,
    )

    game = store.report()["games"]["codenames:operator_eval"]
    red_spy = game["openai:gpt-4o-mini [SPYMASTER]"]["elo"]
    blue_spy = game["anthropic:claude-sonnet-4-20250514 [SPYMASTER]"]["elo"]
    red_op = game["perplexity:sonar [OPERATIVE]"]["elo"]
    blue_op = game["local:llama3.1 [OPERATIVE]"]["elo"]

    assert red_spy > blue_spy
    assert red_op > blue_op
