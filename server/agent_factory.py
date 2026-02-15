"""Factory for building agents from session player configuration."""

from __future__ import annotations

from typing import Any, Callable

from framework.agents.provider_agents import (
    AnthropicAgent,
    LocalLLMAgent,
    NemotronLocalAgent,
    OpenAIAgent,
    PerplexityAgent,
)
from framework.agents.provider_clients import DEFAULT_ANTHROPIC_MODEL
from framework.agents.random_agent import RandomAgent
from framework.move import Move


def normalize_player_config(raw: Any) -> dict[str, Any]:
    """Normalize a player configuration into a typed dictionary."""
    if isinstance(raw, str):
        return {"type": raw.strip().lower()}
    if isinstance(raw, dict):
        data = dict(raw)
        player_type = str(data.get("type", "random")).strip().lower()
        data["type"] = player_type
        return data
    return {"type": "random"}


def player_label(config: dict[str, Any]) -> str:
    """Return stable label used in report cards."""
    player_type = str(config.get("type", "random")).lower()
    model = config.get("model")
    if model:
        return f"{player_type}:{model}"
    return player_type


def create_agent_for_player(
    *,
    player_id: str,
    config: dict[str, Any],
    move_parser: Callable[[dict[str, Any]], Move],
) -> Any:
    """Instantiate a concrete agent for one player config."""
    player_type = str(config.get("type", "random")).lower()
    if player_type == "random":
        return RandomAgent(agent_id=f"random-{player_id.lower()}")

    if player_type == "openai":
        return OpenAIAgent(
            agent_id=f"openai-{player_id.lower()}",
            move_parser=move_parser,
            model=str(config.get("model", "gpt-4o-mini")),
            system_prompt=config.get("system_prompt"),
            max_retries=int(config.get("max_retries", 2)),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config["max_tokens"]) if "max_tokens" in config else None,
        )

    if player_type == "anthropic":
        return AnthropicAgent(
            agent_id=f"anthropic-{player_id.lower()}",
            move_parser=move_parser,
            model=str(config.get("model", DEFAULT_ANTHROPIC_MODEL)),
            system_prompt=config.get("system_prompt"),
            max_retries=int(config.get("max_retries", 2)),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 1024)),
        )

    if player_type == "perplexity":
        return PerplexityAgent(
            agent_id=f"perplexity-{player_id.lower()}",
            move_parser=move_parser,
            model=str(config.get("model", "sonar")),
            system_prompt=config.get("system_prompt"),
            max_retries=int(config.get("max_retries", 2)),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config["max_tokens"]) if "max_tokens" in config else None,
        )

    if player_type == "local":
        return LocalLLMAgent(
            agent_id=f"local-{player_id.lower()}",
            move_parser=move_parser,
            model=str(config.get("model", "llama3.1")),
            backend=str(config.get("backend", "transformers")),
            base_url=config.get("base_url"),
            system_prompt=config.get("system_prompt"),
            max_retries=int(config.get("max_retries", 2)),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config["max_tokens"]) if "max_tokens" in config else None,
        )

    if player_type == "nemotron":
        base_url = config.get("base_url")
        return NemotronLocalAgent(
            agent_id=f"nemotron-{player_id.lower()}",
            move_parser=move_parser,
            model=str(config["model"]) if "model" in config else None,
            backend=str(config.get("backend", "transformers")),
            base_url=str(base_url) if base_url is not None else None,
            system_prompt=config.get("system_prompt"),
            max_retries=int(config.get("max_retries", 2)),
            temperature=float(config["temperature"]) if "temperature" in config else None,
            max_tokens=int(config["max_tokens"]) if "max_tokens" in config else None,
        )

    raise ValueError(
        f"Unsupported player type '{player_type}'. "
        "Supported types: human, random, openai, anthropic, perplexity, local, nemotron."
    )
