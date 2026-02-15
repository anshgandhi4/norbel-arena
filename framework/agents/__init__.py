"""Baseline agent implementations."""

from .env_utils import getenv_any, load_dotenv, require_env_any
from .llm_agent import LLMAgent, LLMClient
from .provider_agents import (
    AnthropicAgent,
    LocalLLMAgent,
    NemotronLocalAgent,
    OpenAIAgent,
    PerplexityAgent,
)
from .provider_clients import (
    AnthropicMessagesClient,
    LocalLLMInferenceEngine,
    LocalOpenAICompatClient,
    OllamaClient,
    OpenAIChatClient,
    PerplexityChatClient,
    default_nemotron_engine,
)
from .random_agent import RandomAgent

__all__ = [
    "AnthropicAgent",
    "AnthropicMessagesClient",
    "LLMAgent",
    "LLMClient",
    "LocalLLMAgent",
    "LocalLLMInferenceEngine",
    "LocalOpenAICompatClient",
    "NemotronLocalAgent",
    "OllamaClient",
    "OpenAIAgent",
    "OpenAIChatClient",
    "PerplexityAgent",
    "PerplexityChatClient",
    "RandomAgent",
    "default_nemotron_engine",
    "getenv_any",
    "load_dotenv",
    "require_env_any",
]
