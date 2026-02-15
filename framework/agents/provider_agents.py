"""Convenience provider-specific agent wrappers around LLMAgent."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..move import Move
from .llm_agent import LLMAgent
from .provider_clients import (
    AnthropicMessagesClient,
    DEFAULT_ANTHROPIC_MODEL,
    LocalLLMInferenceEngine,
    OpenAIChatClient,
    PerplexityChatClient,
    default_nemotron_engine,
)


class OpenAIAgent(LLMAgent):
    """LLM agent backed by OpenAI chat models."""

    def __init__(
        self,
        agent_id: str,
        move_parser: Callable[[Mapping[str, Any]], Move],
        *,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=OpenAIChatClient(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            move_parser=move_parser,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )


class AnthropicAgent(LLMAgent):
    """LLM agent backed by Anthropic Claude models."""

    def __init__(
        self,
        agent_id: str,
        move_parser: Callable[[Mapping[str, Any]], Move],
        *,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        system_prompt: str | None = None,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=AnthropicMessagesClient(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            move_parser=move_parser,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )


class PerplexityAgent(LLMAgent):
    """LLM agent backed by Perplexity models."""

    def __init__(
        self,
        agent_id: str,
        move_parser: Callable[[Mapping[str, Any]], Move],
        *,
        model: str = "sonar",
        system_prompt: str | None = None,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=PerplexityChatClient(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            move_parser=move_parser,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )


class LocalLLMAgent(LLMAgent):
    """LLM agent that uses a local inference backend."""

    def __init__(
        self,
        agent_id: str,
        move_parser: Callable[[Mapping[str, Any]], Move],
        *,
        model: str,
        backend: str = "ollama",
        base_url: str | None = None,
        system_prompt: str | None = None,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=LocalLLMInferenceEngine(
                model=model,
                backend=backend,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            move_parser=move_parser,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )


class NemotronLocalAgent(LLMAgent):
    """
    Local NVIDIA Nemotron agent.

    Defaults are environment-driven via `default_nemotron_engine()`.
    """

    def __init__(
        self,
        agent_id: str,
        move_parser: Callable[[Mapping[str, Any]], Move],
        *,
        model: str | None = None,
        backend: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=default_nemotron_engine(
                model=model,
                backend=backend,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            move_parser=move_parser,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )
