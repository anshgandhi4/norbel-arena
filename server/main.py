"""FastAPI server exposing a local match API for human and/or AI play."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from framework.serialize import json_dumps
from server.schemas import CreateMatchRequest, SubmitMoveRequest
from server.session import SessionStore

app = FastAPI(title="State Games Local API", version="0.1.0")
store = SessionStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Healthcheck endpoint."""
    return {"status": "ok"}


def _time_based_seed() -> int:
    """Generate a positive time-derived seed when client does not provide one."""
    seed = int(time.time_ns() & 0x7FFFFFFF)
    return seed if seed != 0 else 1


@app.post("/api/match/new")
def new_match(request: CreateMatchRequest) -> dict:
    """Create a new in-memory match session."""
    seed = request.seed if request.seed is not None else _time_based_seed()
    try:
        session = store.create_match(
            game=request.game,
            seed=seed,
            config=request.config,
            players=request.players,
            human_player_id=request.human_player_id,
            viewer_player_id=request.viewer_player_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    player_id = session.selected_player(request.viewer_player_id or request.human_player_id)
    return session.view(player_id)


@app.get("/api/match/{match_id}/observation")
def get_observation(
    match_id: str,
    player_id: str = Query(...),
    turn: int | None = Query(default=None, ge=0),
) -> dict:
    """Get latest observation and legal moves for one player."""
    try:
        session = store.get(match_id)
        return session.view(player_id, turn=turn)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown match_id: {match_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/match/{match_id}/move")
def submit_move(match_id: str, request: SubmitMoveRequest) -> dict:
    """Submit a strict JSON move for a human player and advance session."""
    try:
        session = store.get(match_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown match_id: {match_id}") from exc

    try:
        return session.submit_human_move(player_id=request.player_id, move_payload=request.move)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        # Include refreshed state for convenient UI recovery.
        payload = session.view(request.player_id)
        payload["error"] = str(exc)
        raise HTTPException(status_code=400, detail=payload) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/match/{match_id}/events", response_model=None)
def get_events(match_id: str, format: str = Query(default="array")) -> Any:
    """Return full event history as array (default) or JSONL text."""
    try:
        events = store.all_events(match_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown match_id: {match_id}") from exc

    if format == "jsonl":
        text = "\n".join(json_dumps(event) for event in events)
        return PlainTextResponse(content=text, media_type="application/jsonl")
    return events


@app.get("/api/report-cards")
def get_report_cards() -> dict[str, Any]:
    """Return persisted report cards for all games and agents."""
    return store.report_cards()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
