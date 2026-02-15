"""Baseline agent implementations."""

from .env_utils import getenv_any, load_dotenv, require_env_any
from .human_cli_controller import HumanCLIController
from .human_agent import HumanAgent
from .llm_agent import LLMAgent, LLMClient, StubLLMClient
from .provider_agents import (
    AnthropicAgent,
    GrokAgent,
    LocalLLMAgent,
    NemotronLocalAgent,
    OpenAIAgent,
    PerplexityAgent,
)
from .provider_clients import (
    AnthropicMessagesClient,
    GrokChatClient,
    LocalLLMInferenceEngine,
    LocalOpenAICompatClient,
    OllamaClient,
    OpenAIChatClient,
    PerplexityChatClient,
    default_nemotron_engine,
)
from .random_agent import RandomAgent
from .scripted_agent import ScriptedAgent

__all__ = [
    "AnthropicAgent",
    "AnthropicMessagesClient",
    "GrokAgent",
    "GrokChatClient",
    "HumanCLIController",
    "HumanAgent",
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
    "ScriptedAgent",
    "StubLLMClient",
    "default_nemotron_engine",
    "getenv_any",
    "load_dotenv",
    "require_env_any",
]
