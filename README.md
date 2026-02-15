# Codenames Arena (Framework + UI)

This repo contains:
- A game framework for state-based games
- A Codenames implementation with partial observability
- A local FastAPI backend for stepwise human + AI matches
- A React frontend for playing as one seat against AI seats

## Run Backend

```bash
.venv/bin/uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /api/match/new`
- `GET /api/match/{match_id}/observation?player_id=...`
- `POST /api/match/{match_id}/move`
- `GET /api/match/{match_id}/events`

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend defaults to `http://localhost:8000` for API calls.
Override with:

```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```

## Play Flow

1. Open the frontend in your browser.
2. Choose a human seat (`RED_OPERATIVE`, `RED_SPYMASTER`, `BLUE_OPERATIVE`, or `BLUE_SPYMASTER`).
3. Create a match.
4. Use controls to submit strict JSON-equivalent moves:
   - GiveClue: `{ "type": "GiveClue", "clue": "animal", "count": 2 }`
   - Guess: `{ "type": "Guess", "index": 13 }`
   - EndTurn: `{ "type": "EndTurn" }`
   - Resign: `{ "type": "Resign" }`

## Tests

Run all tests:

```bash
.venv/bin/pytest -q
```

Includes API smoke coverage (`tests/test_server_api_smoke.py`) for:
- create match
- fetch observation
- submit legal move
- fetch events

## Notes

- Match sessions are stored in-memory and keyed by `match_id`.
- Observations are server-shaped per player role. Operative views do not receive hidden assignments.
- `.env` can contain API keys for LLM agents and is loaded by the agents utilities.

## TODO:

1. Remove human requirement for game. 
2. Make the front end users be able to choose their game setup (which agents in which roles. human players if any)
3. Report card for each agenty (persist in a JSON DB for now)
4. Allow the usexr to run a game and replay through it, clicking forward and backward across the different moves made
5. tic tac toe, 20 questions, wavelength