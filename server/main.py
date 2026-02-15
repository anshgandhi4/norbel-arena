"""FastAPI server exposing a local match API for human-in-the-loop play."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from framework.serialize import json_dumps
from server.schemas import CreateMatchRequest, SubmitMoveRequest
from server.session import SessionStore

app = FastAPI(title="Codenames Local API", version="0.1.0")
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


@app.post("/api/match/new")
def new_match(request: CreateMatchRequest) -> dict:
    """Create a new in-memory match session."""
    try:
        session = store.create_match(
            seed=request.seed,
            config=request.config,
            players=request.players,
            human_player_id=request.human_player_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    player_id = session.selected_player(request.human_player_id)
    return session.view(player_id)


@app.get("/api/match/{match_id}/observation")
def get_observation(match_id: str, player_id: str = Query(...)) -> dict:
    """Get latest observation and legal moves for one player."""
    try:
        session = store.get(match_id)
        return session.view(player_id)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
