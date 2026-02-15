"""Tests for provider clients and local engine wiring."""

from __future__ import annotations

import os

from framework.agents import env_utils
from framework.agents.provider_clients import (
    AnthropicMessagesClient,
    LocalLLMInferenceEngine,
    OpenAIChatClient,
)


def test_load_dotenv_sets_missing_vars(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=test-key\n# comment\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_utils._DOTENV_LOADED = False  # reset module guard for test
    env_utils.load_dotenv(dotenv)
    assert os.getenv("OPENAI_API_KEY") == "test-key"


def test_openai_client_extracts_content(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    def fake_post_json(url, payload, headers, timeout_sec=60.0):  # noqa: ANN001
        return {"choices": [{"message": {"content": '{"type":"EndTurn"}'}}]}

    monkeypatch.setattr("framework.agents.provider_clients.post_json", fake_post_json)
    client = OpenAIChatClient(model="gpt-4o-mini")
    assert client.complete("hello").strip() == '{"type":"EndTurn"}'


def test_anthropic_client_extracts_text_blocks(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    def fake_post_json(url, payload, headers, timeout_sec=60.0):  # noqa: ANN001
        return {"content": [{"type": "text", "text": '{"type":"Guess","index":1}'}]}

    monkeypatch.setattr("framework.agents.provider_clients.post_json", fake_post_json)
    client = AnthropicMessagesClient(model="claude")
    assert client.complete("hello").strip() == '{"type":"Guess","index":1}'


def test_local_engine_uses_openai_compat_backend(monkeypatch) -> None:
    def fake_post_json(url, payload, headers, timeout_sec=60.0):  # noqa: ANN001
        return {"choices": [{"message": {"content": '{"type":"GiveClue","clue":"x","count":1}'}}]}

    monkeypatch.setattr("framework.agents.provider_clients.post_json", fake_post_json)
    engine = LocalLLMInferenceEngine(
        model="nemotron",
        backend="openai_compat",
        base_url="http://127.0.0.1:9999/v1",
    )
    out = engine.complete("move")
    assert '"type":"GiveClue"' in out
