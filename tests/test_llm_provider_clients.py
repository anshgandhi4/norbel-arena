"""Tests for provider clients and local engine wiring."""

from __future__ import annotations

import os
import sys
import types

import pytest

from framework.agents import env_utils
from framework.agents import provider_clients as provider_clients_module
from framework.agents.provider_clients import (
    AnthropicMessagesClient,
    DEFAULT_ANTHROPIC_MODEL,
    LocalLLMInferenceEngine,
    OpenAIChatClient,
    _complete_with_transformers,
    default_nemotron_engine,
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


def test_anthropic_client_uses_v1_messages_path_and_aliases_retired_model(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    captured: dict[str, object] = {}

    def fake_post_json(url, payload, headers, timeout_sec=60.0):  # noqa: ANN001
        captured["url"] = url
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": '{"type":"EndTurn"}'}]}

    monkeypatch.setattr("framework.agents.provider_clients.post_json", fake_post_json)
    client = AnthropicMessagesClient(base_url="https://api.anthropic.com")
    out = client.complete("hello")
    assert out.strip() == '{"type":"EndTurn"}'
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert isinstance(captured["payload"], dict)
    assert captured["payload"]["model"] == DEFAULT_ANTHROPIC_MODEL


def test_anthropic_client_falls_back_when_model_missing(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    calls: list[dict[str, object]] = []

    def fake_post_json(url, payload, headers, timeout_sec=60.0):  # noqa: ANN001
        calls.append({"url": url, "payload": dict(payload)})
        if len(calls) == 1:
            raise RuntimeError(
                'HTTP 404 from https://api.anthropic.com/v1/messages: '
                '{"type":"error","error":{"type":"not_found_error","message":"model: missing-model"}}'
            )
        return {"content": [{"type": "text", "text": '{"type":"EndTurn"}'}]}

    monkeypatch.setattr("framework.agents.provider_clients.post_json", fake_post_json)
    client = AnthropicMessagesClient(model="missing-model")
    out = client.complete("hello")
    assert out.strip() == '{"type":"EndTurn"}'
    assert len(calls) == 2
    assert calls[0]["payload"]["model"] == "missing-model"
    assert calls[1]["payload"]["model"] == DEFAULT_ANTHROPIC_MODEL


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


def test_default_nemotron_engine_accepts_explicit_overrides() -> None:
    engine = default_nemotron_engine(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        backend="openai_compat",
        base_url="http://127.0.0.1:9000/v1",
        temperature=0.2,
        max_tokens=256,
    )
    assert engine.model == "nvidia/llama-3.1-nemotron-70b-instruct"
    assert engine.backend == "openai_compat"
    assert engine.base_url == "http://127.0.0.1:9000/v1"
    assert engine.temperature == 0.2
    assert engine.max_tokens == 256


def test_default_nemotron_engine_uses_4b_default_for_transformers(monkeypatch) -> None:
    monkeypatch.delenv("NEMOTRON_MODEL", raising=False)
    monkeypatch.delenv("NEMOTRON_LOCAL_BACKEND", raising=False)
    engine = default_nemotron_engine(backend="transformers")
    assert engine.model == "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"


def test_local_engine_uses_transformers_backend(monkeypatch) -> None:
    def fake_complete_with_transformers(*, model, prompt, system_prompt, temperature, max_tokens):  # noqa: ANN001
        assert model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
        assert prompt == "move"
        assert system_prompt == "sys"
        assert temperature == 0.0
        assert max_tokens == 128
        return '{"type":"EndTurn"}'

    monkeypatch.setattr(
        "framework.agents.provider_clients._complete_with_transformers",
        fake_complete_with_transformers,
    )
    engine = LocalLLMInferenceEngine(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        backend="transformers",
        max_tokens=128,
    )
    out = engine.complete("move", system_prompt="sys")
    assert out.strip() == '{"type":"EndTurn"}'


def test_transformers_backend_surfaces_actionable_protobuf_error(monkeypatch) -> None:
    def fake_complete_with_transformers(*, model, prompt, system_prompt, temperature, max_tokens):  # noqa: ANN001
        raise RuntimeError(
            "\n requires the protobuf library but it was not found in your environment. "
            "Check out the instructions on the installation page.\n"
        )

    monkeypatch.setattr(
        "framework.agents.provider_clients._complete_with_transformers",
        fake_complete_with_transformers,
    )
    engine = LocalLLMInferenceEngine(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        backend="transformers",
    )
    with pytest.raises(RuntimeError, match="pip install protobuf"):
        engine.complete("move")


def test_transformers_backend_surfaces_actionable_tokenizer_dependency_error(monkeypatch) -> None:
    def fake_complete_with_transformers(*, model, prompt, system_prompt, temperature, max_tokens):  # noqa: ANN001
        raise RuntimeError(
            "Couldn't instantiate the backend tokenizer from one of: "
            "(1) a tokenizers serialization file, "
            "(2) a slow tokenizer instance, or "
            "(3) an equivalent slow tokenizer class. "
            "You need to have sentencepiece or tiktoken installed."
        )

    monkeypatch.setattr(
        "framework.agents.provider_clients._complete_with_transformers",
        fake_complete_with_transformers,
    )
    monkeypatch.setattr(
        "framework.agents.provider_clients._tokenizer_dependency_status",
        lambda: (False, False),
    )
    engine = LocalLLMInferenceEngine(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        backend="transformers",
    )
    with pytest.raises(RuntimeError, match="pip install sentencepiece tiktoken"):
        engine.complete("move")


def test_transformers_backend_reports_real_error_when_tokenizer_deps_are_installed(monkeypatch) -> None:
    def fake_complete_with_transformers(*, model, prompt, system_prompt, temperature, max_tokens):  # noqa: ANN001
        raise RuntimeError(
            "Couldn't instantiate the backend tokenizer from one of: "
            "(1) a tokenizers serialization file, "
            "(2) a slow tokenizer instance, or "
            "(3) an equivalent slow tokenizer class. "
            "You need to have sentencepiece or tiktoken installed."
        )

    monkeypatch.setattr(
        "framework.agents.provider_clients._complete_with_transformers",
        fake_complete_with_transformers,
    )
    monkeypatch.setattr(
        "framework.agents.provider_clients._tokenizer_dependency_status",
        lambda: (True, True),
    )
    engine = LocalLLMInferenceEngine(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        backend="transformers",
    )
    with pytest.raises(RuntimeError, match="model/tokenizer compatibility issue"):
        engine.complete("move")


def test_transformers_backend_surfaces_actionable_mamba_dependency_error(monkeypatch) -> None:
    def fake_complete_with_transformers(*, model, prompt, system_prompt, temperature, max_tokens):  # noqa: ANN001
        raise RuntimeError("mamba-ssm is required by the Mamba model but cannot be imported")

    monkeypatch.setattr(
        "framework.agents.provider_clients._complete_with_transformers",
        fake_complete_with_transformers,
    )
    monkeypatch.setattr(
        "framework.agents.provider_clients._mamba_dependency_status",
        lambda: (False, False),
    )
    engine = LocalLLMInferenceEngine(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        backend="transformers",
    )
    with pytest.raises(RuntimeError, match="backend='openai_compat'"):
        engine.complete("move")


def test_transformers_bundle_retries_with_slow_tokenizer_when_fast_init_needs_extra_deps(monkeypatch) -> None:
    provider_clients_module._TRANSFORMERS_BUNDLE_CACHE.clear()
    tokenizer_calls: list[dict[str, object]] = []
    fake_tokenizer = object()
    fake_model = object()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code=True, **kwargs):  # noqa: ANN001
            tokenizer_calls.append(dict(kwargs))
            if kwargs.get("use_fast", True):
                raise ValueError("You need to have sentencepiece or tiktoken installed.")
            return fake_tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model, **kwargs):  # noqa: ANN001
            return fake_model

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    tokenizer, lm_model = provider_clients_module._transformers_bundle("dummy-model")
    assert tokenizer is fake_tokenizer
    assert lm_model is fake_model
    assert len(tokenizer_calls) == 2
    assert tokenizer_calls[0].get("use_fast") is None
    assert tokenizer_calls[1].get("use_fast") is False


def test_transformers_bundle_reports_non_dependency_failure_after_slow_fallback(monkeypatch) -> None:
    provider_clients_module._TRANSFORMERS_BUNDLE_CACHE.clear()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code=True, **kwargs):  # noqa: ANN001
            if kwargs.get("use_fast", True):
                raise ValueError("You need to have sentencepiece or tiktoken installed.")
            raise ValueError("No slow tokenizer class found for this model.")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model, **kwargs):  # noqa: ANN001
            return object()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "framework.agents.provider_clients._tokenizer_dependency_status",
        lambda: (True, True),
    )

    with pytest.raises(RuntimeError, match="even with `use_fast=False`"):
        provider_clients_module._transformers_bundle("dummy-model-without-slow-tokenizer")


def test_transformers_bundle_nemotron_70b_guides_to_compatible_backend_or_model(monkeypatch) -> None:
    provider_clients_module._TRANSFORMERS_BUNDLE_CACHE.clear()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code=True, **kwargs):  # noqa: ANN001
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: "
                "(1) tokenizers file, (2) slow tokenizer instance, (3) slow tokenizer class. "
                "You need to have sentencepiece or tiktoken installed."
            )

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model, **kwargs):  # noqa: ANN001
            return object()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "framework.agents.provider_clients._tokenizer_dependency_status",
        lambda: (True, True),
    )

    with pytest.raises(RuntimeError, match="openai_compat"):
        provider_clients_module._transformers_bundle("nvidia/llama-3.1-nemotron-70b-instruct")


def test_transformers_bundle_wraps_mamba_dependency_error_with_actionable_guidance(monkeypatch) -> None:
    provider_clients_module._TRANSFORMERS_BUNDLE_CACHE.clear()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code=True, **kwargs):  # noqa: ANN001
            return object()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model, **kwargs):  # noqa: ANN001
            raise RuntimeError("mamba-ssm is required by the Mamba model but cannot be imported")

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "framework.agents.provider_clients._mamba_dependency_status",
        lambda: (False, False),
    )

    with pytest.raises(RuntimeError, match="backend='openai_compat'"):
        provider_clients_module._transformers_bundle("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")


def test_complete_with_transformers_handles_dict_chat_template(monkeypatch) -> None:
    import torch

    captured: dict[str, object] = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors, return_dict):  # noqa: ANN001
            assert tokenize is True
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            assert return_dict is True
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def decode(self, generated_ids, skip_special_tokens=True):  # noqa: ANN001
            assert skip_special_tokens is True
            assert generated_ids.tolist() == [4]
            return '{"type":"EndTurn"}'

    class FakeModel:
        device = torch.device("cpu")

        def parameters(self):  # noqa: ANN201
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, *, input_ids, attention_mask=None, **kwargs):  # noqa: ANN001
            captured["input_ids_type"] = type(input_ids).__name__
            captured["attention_mask_type"] = type(attention_mask).__name__ if attention_mask is not None else None
            captured["kwargs"] = kwargs
            assert hasattr(input_ids, "shape")
            assert input_ids.shape == torch.Size([1, 3])
            assert attention_mask is not None
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    monkeypatch.setattr(
        "framework.agents.provider_clients._transformers_bundle",
        lambda model: (FakeTokenizer(), FakeModel()),
    )
    out = _complete_with_transformers(
        model="dummy",
        prompt="move",
        system_prompt="sys",
        temperature=0.0,
        max_tokens=8,
    )
    assert out.strip() == '{"type":"EndTurn"}'
    assert captured["input_ids_type"] == "Tensor"
    assert captured["attention_mask_type"] == "Tensor"
    assert isinstance(captured["kwargs"], dict)
    assert captured["kwargs"]["max_new_tokens"] == 8


def test_complete_with_transformers_handles_mapping_chat_template(monkeypatch) -> None:
    import torch
    from collections import UserDict

    captured: dict[str, object] = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors, return_dict):  # noqa: ANN001
            assert tokenize is True
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            assert return_dict is True
            return UserDict(
                {
                    "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }
            )

        def decode(self, generated_ids, skip_special_tokens=True):  # noqa: ANN001
            assert skip_special_tokens is True
            assert generated_ids.tolist() == [14]
            return '{"type":"EndTurn"}'

    class FakeModel:
        device = torch.device("cpu")

        def parameters(self):  # noqa: ANN201
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, *, input_ids, attention_mask=None, **kwargs):  # noqa: ANN001
            captured["input_ids_type"] = type(input_ids).__name__
            captured["attention_mask_type"] = type(attention_mask).__name__ if attention_mask is not None else None
            captured["kwargs"] = kwargs
            assert hasattr(input_ids, "shape")
            assert input_ids.shape == torch.Size([1, 3])
            assert attention_mask is not None
            return torch.tensor([[11, 12, 13, 14]], dtype=torch.long)

    monkeypatch.setattr(
        "framework.agents.provider_clients._transformers_bundle",
        lambda model: (FakeTokenizer(), FakeModel()),
    )
    out = _complete_with_transformers(
        model="dummy",
        prompt="move",
        system_prompt="sys",
        temperature=0.0,
        max_tokens=8,
    )
    assert out.strip() == '{"type":"EndTurn"}'
    assert captured["input_ids_type"] == "Tensor"
    assert captured["attention_mask_type"] == "Tensor"
    assert isinstance(captured["kwargs"], dict)
    assert captured["kwargs"]["max_new_tokens"] == 8
