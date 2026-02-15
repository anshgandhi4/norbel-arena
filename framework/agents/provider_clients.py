"""Provider-specific LLM clients and local inference engine backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .env_utils import getenv_any, require_env_any
from .http_utils import post_json


def _extract_openai_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Provider response did not include choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        # Some providers return structured content blocks.
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str):
        raise ValueError("Provider response message content was not a string.")
    return content


@dataclass(frozen=True)
class OpenAIChatClient:
    """OpenAI Chat Completions API client."""

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_tokens: int | None = None
    api_key_env: tuple[str, ...] = ("OPENAI_API_KEY",)

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        api_key = require_env_any(*self.api_key_env)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        response = post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout_sec=self.timeout_sec,
        )
        return _extract_openai_content(response)


@dataclass(frozen=True)
class PerplexityChatClient(OpenAIChatClient):
    """Perplexity API client (OpenAI-compatible chat interface)."""

    model: str = "sonar"
    base_url: str = "https://api.perplexity.ai"
    api_key_env: tuple[str, ...] = ("PERPLEXITY_API_KEY", "PPLX_API_KEY")


@dataclass(frozen=True)
class GrokChatClient(OpenAIChatClient):
    """xAI Grok API client (OpenAI-compatible chat interface)."""

    model: str = "grok-2-latest"
    base_url: str = "https://api.x.ai/v1"
    api_key_env: tuple[str, ...] = ("XAI_API_KEY", "GROK_API_KEY")


@dataclass(frozen=True)
class AnthropicMessagesClient:
    """Anthropic Messages API client."""

    model: str = "claude-3-5-sonnet-latest"
    base_url: str = "https://api.anthropic.com/v1"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 1024
    anthropic_version: str = "2023-06-01"
    api_key_env: tuple[str, ...] = ("ANTHROPIC_API_KEY",)

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        api_key = require_env_any(*self.api_key_env)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        response = post_json(
            url=f"{self.base_url.rstrip('/')}/messages",
            payload=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.anthropic_version,
            },
            timeout_sec=self.timeout_sec,
        )
        content = response.get("content", [])
        if not content:
            raise ValueError("Anthropic response did not include content blocks.")
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "".join(text_parts).strip()
        if not joined:
            raise ValueError("Anthropic response contained no text content.")
        return joined


@dataclass(frozen=True)
class OllamaClient:
    """Local Ollama chat client."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_sec: float = 120.0
    temperature: float = 0.0

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = post_json(
            url=f"{self.base_url.rstrip('/')}/api/chat",
            payload=payload,
            headers={},
            timeout_sec=self.timeout_sec,
        )
        message = response.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response did not include message.content.")
        return content


@dataclass(frozen=True)
class LocalOpenAICompatClient:
    """
    Local OpenAI-compatible client.

    Works with engines such as vLLM, TGI OpenAI gateway, or NVIDIA NIM when served
    on a local OpenAI-compatible endpoint.
    """

    model: str
    base_url: str = "http://127.0.0.1:8000/v1"
    timeout_sec: float = 120.0
    temperature: float = 0.0
    max_tokens: int | None = None
    api_key: str | None = None

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            headers=headers,
            timeout_sec=self.timeout_sec,
        )
        return _extract_openai_content(response)


@dataclass(frozen=True)
class LocalLLMInferenceEngine:
    """
    Local inference engine abstraction for the LLMAgent client protocol.

    `backend`:
    - `ollama`: uses Ollama local server.
    - `openai_compat`: uses local OpenAI-compatible server (vLLM/NIM/etc.).
    """

    model: str
    backend: str = "ollama"
    base_url: str | None = None
    timeout_sec: float = 120.0
    temperature: float = 0.0
    max_tokens: int | None = None

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        backend = self.backend.strip().lower()
        if backend == "ollama":
            client = OllamaClient(
                model=self.model,
                base_url=self.base_url or getenv_any("OLLAMA_BASE_URL", default="http://127.0.0.1:11434") or "http://127.0.0.1:11434",
                timeout_sec=self.timeout_sec,
                temperature=self.temperature,
            )
            return client.complete(prompt, system_prompt=system_prompt)
        if backend == "openai_compat":
            client = LocalOpenAICompatClient(
                model=self.model,
                base_url=self.base_url
                or getenv_any("LOCAL_LLM_BASE_URL", "OPENAI_COMPAT_BASE_URL", default="http://127.0.0.1:8000/v1")
                or "http://127.0.0.1:8000/v1",
                timeout_sec=self.timeout_sec,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=getenv_any("LOCAL_LLM_API_KEY", "OPENAI_COMPAT_API_KEY"),
            )
            return client.complete(prompt, system_prompt=system_prompt)
        raise ValueError(f"Unsupported local backend: {self.backend!r}")


def default_nemotron_engine() -> LocalLLMInferenceEngine:
    """
    Build a local engine configured for NVIDIA Nemotron.

    Environment variables:
    - `NEMOTRON_LOCAL_BACKEND` (`ollama` or `openai_compat`)
    - `NEMOTRON_MODEL`
    - `NEMOTRON_BASE_URL`
    """
    backend = getenv_any("NEMOTRON_LOCAL_BACKEND", default="ollama") or "ollama"
    default_model = "nemotron-mini" if backend == "ollama" else "nvidia/llama-3.1-nemotron-70b-instruct"
    model = getenv_any("NEMOTRON_MODEL", default=default_model) or default_model
    return LocalLLMInferenceEngine(
        model=model,
        backend=backend,
        base_url=getenv_any("NEMOTRON_BASE_URL"),
        timeout_sec=float(getenv_any("NEMOTRON_TIMEOUT_SEC", default="120") or "120"),
        temperature=float(getenv_any("NEMOTRON_TEMPERATURE", default="0") or "0"),
        max_tokens=int(getenv_any("NEMOTRON_MAX_TOKENS", default="1024") or "1024"),
    )
