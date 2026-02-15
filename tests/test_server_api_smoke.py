"""Smoke tests for the local FastAPI match API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.main import app


def test_match_api_flow_create_observe_move_events() -> None:
    client = TestClient(app)
    create = client.post(
        "/api/match/new",
        json={
            "seed": 123,
            "config": {"starting_team": "RED"},
            "players": {
                "RED_SPYMASTER": "human",
                "RED_OPERATIVE": "random",
                "BLUE_SPYMASTER": "random",
                "BLUE_OPERATIVE": "random",
            },
            "human_player_id": "RED_SPYMASTER",
        },
    )
    assert create.status_code == 200, create.text
    created = create.json()
    match_id = created["match_id"]
    player_id = created["player_id"]
    initial_turn = created["observation"]["turn_index"]

    observe = client.get(f"/api/match/{match_id}/observation", params={"player_id": player_id})
    assert observe.status_code == 200, observe.text
    observed = observe.json()
    assert observed["match_id"] == match_id
    assert observed["player_id"] == player_id

    move = client.post(
        f"/api/match/{match_id}/move",
        json={
            "player_id": player_id,
            "move": {"type": "GiveClue", "clue": "animal", "count": 1},
        },
    )
    assert move.status_code == 200, move.text
    moved = move.json()
    assert moved["observation"]["turn_index"] > initial_turn

    events = client.get(f"/api/match/{match_id}/events")
    assert events.status_code == 200, events.text
    payload = events.json()
    assert isinstance(payload, list)
    assert len(payload) >= 2
