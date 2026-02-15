# Codenames Arena (Framework + UI)

This repo contains:
- A game framework for state-based games
- A Codenames implementation with partial observability
- A local FastAPI backend for configurable human/AI match sessions
- A React frontend for setup, live play, replay, and report cards

## Run Backend

```bash
.venv/bin/uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /api/match/new`
- `GET /api/match/{match_id}/observation?player_id=...`
- `POST /api/match/{match_id}/move`
- `GET /api/match/{match_id}/events`
- `GET /api/report-cards`

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
2. Configure each seat (`RED_SPYMASTER`, `RED_OPERATIVE`, `BLUE_SPYMASTER`, `BLUE_OPERATIVE`) as:
   - `human`
   - `random`
   - `openai`
   - `anthropic`
   - `perplexity`
   - `local`
   - `nemotron`
3. Choose a viewer perspective and create a match.
4. If your viewer seat is `human`, use controls to submit strict JSON-equivalent moves:
   - GiveClue: `{ "type": "GiveClue", "clue": "animal", "count": 2 }`
   - Guess: `{ "type": "Guess", "index": 13 }`
   - EndTurn: `{ "type": "EndTurn" }`
5. Use replay controls to move backward/forward through turns.
6. Inspect report cards for aggregate performance by agent label and game.

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
- run all-AI match to terminal
- retrieve persisted report cards

## Notes

- Match sessions are stored in-memory and keyed by `match_id`.
- Sessions also store immutable `state_history` snapshots so observations can be requested for prior turns (`turn` query param) for replay.
- Observations are server-shaped per player role. Operative views do not receive hidden assignments.
- Human players are optional. You can run full AI-vs-AI matches and watch/replay from any seat perspective.
- Report cards persist to `server/data/report_cards.json`.
- Override report-card path with `REPORT_CARD_DB_PATH=/path/to/report_cards.json`.
- `.env` can contain API keys for LLM agents and is loaded by the agents utilities.
- For `local`/`nemotron` with `backend="transformers"`, install local runtime deps (including `protobuf`, `sentencepiece`, and `tiktoken`) before running matches.
- `nvidia/llama-3.1-nemotron-70b-instruct` is intended for served/OpenAI-compatible inference (`backend="openai_compat"`), while `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` is the default for in-process `backend="transformers"`.
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` may require `mamba-ssm`/`causal-conv1d` for in-process Transformers loading; on CPU-only environments this is commonly unavailable, so prefer `backend="openai_compat"` for Nemotron.
