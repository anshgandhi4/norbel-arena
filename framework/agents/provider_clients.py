"""Provider-specific LLM clients and local inference engine backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import sys
from typing import Any

from .env_utils import getenv_any, require_env_any
from .http_utils import post_json

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

_TRANSFORMERS_BUNDLE_CACHE: dict[str, tuple[Any, Any]] = {}

_ANTHROPIC_MODEL_ALIASES = {
    # Sonnet 3.5 aliases were retired; map them to a currently supported default.
    "claude-3-5-sonnet-latest": DEFAULT_ANTHROPIC_MODEL,
    "claude-3-5-sonnet-20241022": DEFAULT_ANTHROPIC_MODEL,
}


def _normalize_anthropic_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        return DEFAULT_ANTHROPIC_MODEL
    return _ANTHROPIC_MODEL_ALIASES.get(normalized.lower(), normalized)


def _anthropic_messages_url(base_url: str) -> str:
    normalized = (base_url or "https://api.anthropic.com").rstrip("/")
    if normalized.endswith("/v1/messages"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/messages"
    return f"{normalized}/v1/messages"


def _is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "not_found_error" in text and "model" in text


def _is_missing_protobuf_error(exc: Exception) -> bool:
    """Detect missing protobuf dependency failures surfaced by transformers/tokenizers."""
    text = str(exc).lower()
    return (
        "requires the protobuf library" in text
        or "no module named 'google.protobuf'" in text
        or ("no module named 'google'" in text and "protobuf" in text)
    )


def _is_missing_tokenizer_dependency_error(exc: Exception) -> bool:
    """Detect missing tokenizer conversion/runtime deps (sentencepiece/tiktoken)."""
    text = str(exc).lower()
    return (
        "sentencepiece" in text
        or "tiktoken" in text
        or "couldn't instantiate the backend tokenizer" in text
    )


def _is_missing_mamba_dependency_error(exc: Exception) -> bool:
    """Detect missing Mamba runtime deps for certain Nemotron checkpoints."""
    text = str(exc).lower()
    return (
        "mamba-ssm is required" in text
        or "mamba model but cannot be imported" in text
        or "no module named 'mamba_ssm'" in text
    )


def _module_importable(module_name: str) -> bool:
    """Return True when a module can be imported in the current interpreter."""
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _tokenizer_dependency_status() -> tuple[bool, bool]:
    """Return (sentencepiece_installed, tiktoken_installed)."""
    return _module_importable("sentencepiece"), _module_importable("tiktoken")


def _mamba_dependency_status() -> tuple[bool, bool]:
    """Return (mamba_ssm_installed, causal_conv1d_installed)."""
    return _module_importable("mamba_ssm"), _module_importable("causal_conv1d")


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
class AnthropicMessagesClient:
    """Anthropic Messages API client."""

    model: str = DEFAULT_ANTHROPIC_MODEL
    base_url: str = "https://api.anthropic.com"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 1024
    anthropic_version: str = "2023-06-01"
    api_key_env: tuple[str, ...] = ("ANTHROPIC_API_KEY",)

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        api_key = require_env_any(*self.api_key_env)
        model = _normalize_anthropic_model(getenv_any("ANTHROPIC_MODEL", default=self.model) or self.model)
        base_url = getenv_any("ANTHROPIC_BASE_URL", default=self.base_url) or self.base_url
        anthropic_version = getenv_any("ANTHROPIC_VERSION", default=self.anthropic_version) or self.anthropic_version
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        url = _anthropic_messages_url(base_url)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
        }
        try:
            response = post_json(
                url=url,
                payload=payload,
                headers=headers,
                timeout_sec=self.timeout_sec,
            )
        except RuntimeError as exc:
            if not _is_model_not_found_error(exc) or model == DEFAULT_ANTHROPIC_MODEL:
                raise
            # Fallback to a supported default model when a custom/stale model is rejected.
            payload["model"] = DEFAULT_ANTHROPIC_MODEL
            response = post_json(
                url=url,
                payload=payload,
                headers=headers,
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


def _transformers_bundle(model: str) -> tuple[Any, Any]:
    """Load and cache tokenizer/model for local in-process transformers inference."""
    if model in _TRANSFORMERS_BUNDLE_CACHE:
        return _TRANSFORMERS_BUNDLE_CACHE[model]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise RuntimeError(
            "Transformers backend requires `transformers` and a compatible runtime (typically `torch`). "
            "Install with: pip install transformers torch accelerate"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:
        if not _is_missing_tokenizer_dependency_error(exc):
            raise
        # Retry with slow tokenizer to avoid fast-conversion dependency requirements.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as slow_exc:
            has_sentencepiece, has_tiktoken = _tokenizer_dependency_status()
            if not (has_sentencepiece and has_tiktoken):
                raise RuntimeError(
                    "Transformers backend tokenizer initialization failed due to missing dependencies. "
                    "Install with: pip install sentencepiece tiktoken (then restart the runtime). "
                    f"Interpreter={sys.executable!r}, sentencepiece_installed={has_sentencepiece}, "
                    f"tiktoken_installed={has_tiktoken}."
                ) from slow_exc
            if model.strip().lower() == "nvidia/llama-3.1-nemotron-70b-instruct":
                raise RuntimeError(
                    "Model 'nvidia/llama-3.1-nemotron-70b-instruct' is not compatible with the in-process "
                    "Transformers tokenizer path in this runtime. Use backend='openai_compat' for that model "
                    "or switch to model 'nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1' for backend='transformers'. "
                    f"Original error: {slow_exc}"
                ) from slow_exc
            raise RuntimeError(
                "Transformers backend tokenizer initialization failed even with `use_fast=False`, "
                "despite sentencepiece/tiktoken being importable. "
                f"Original error: {slow_exc}"
            ) from slow_exc

    try:
        lm_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as model_exc:
        if _is_missing_mamba_dependency_error(model_exc):
            has_mamba, has_causal = _mamba_dependency_status()
            raise RuntimeError(
                "This checkpoint requires Mamba runtime dependencies unavailable in this environment. "
                "For local CPU-only setup, use backend='openai_compat' with a served model endpoint instead of "
                "in-process transformers. "
                "If you specifically need in-process loading, install mamba-ssm/causal-conv1d in a compatible "
                f"CUDA toolchain runtime. Interpreter={sys.executable!r}, mamba_ssm_installed={has_mamba}, "
                f"causal_conv1d_installed={has_causal}. Original error: {model_exc}"
            ) from model_exc
        raise
    _TRANSFORMERS_BUNDLE_CACHE[model] = (tokenizer, lm_model)
    return tokenizer, lm_model


def _complete_with_transformers(
    *,
    model: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int | None,
) -> str:
    """Run one in-process generation step using Hugging Face transformers."""
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise RuntimeError(
            "Transformers backend requires `torch`. "
            "Install with: pip install torch"
        ) from exc

    tokenizer, lm_model = _transformers_bundle(model)
    max_new_tokens = max_tokens or int(
        getenv_any("LOCAL_TRANSFORMERS_MAX_TOKENS", "TRANSFORMERS_MAX_TOKENS", default="1024") or "1024"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # Newer tokenizer implementations can return a dict-like batch.
            rendered = tokenizer.apply_chat_template(  # type: ignore[call-arg]
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            # Older tokenizer implementations may not accept `tokenize`/`return_dict`.
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        if isinstance(rendered, Mapping):
            if "input_ids" not in rendered:
                raise ValueError("Chat template output missing input_ids.")
            input_ids = rendered["input_ids"]
            attention_mask = rendered.get("attention_mask")
        else:
            input_ids = rendered
            attention_mask = None
    else:
        rendered_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    model_device = getattr(lm_model, "device", None)
    if model_device is None or str(model_device) == "cpu":
        model_device = next(lm_model.parameters()).device

    input_ids = input_ids.to(model_device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = lm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    generated_ids = outputs[0][input_ids.shape[-1] :]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not content:
        raise ValueError("Transformers backend produced empty output.")
    return content


@dataclass(frozen=True)
class LocalLLMInferenceEngine:
    """
    Local inference engine abstraction for the LLMAgent client protocol.

    `backend`:
    - `ollama`: uses Ollama local server.
    - `openai_compat`: uses local OpenAI-compatible server (vLLM/NIM/etc.).
    - `transformers`: runs local in-process inference with Hugging Face transformers.
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
        if backend == "transformers":
            try:
                return _complete_with_transformers(
                    model=self.model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                if _is_missing_protobuf_error(exc):
                    raise RuntimeError(
                        "Transformers backend is missing `protobuf`, required by this model tokenizer. "
                        "Install with: pip install protobuf (then restart the runtime)."
                    ) from exc
                if _is_missing_tokenizer_dependency_error(exc):
                    has_sentencepiece, has_tiktoken = _tokenizer_dependency_status()
                    if not (has_sentencepiece and has_tiktoken):
                        raise RuntimeError(
                            "Transformers backend tokenizer dependencies are missing. "
                            "Install with: pip install sentencepiece tiktoken (then restart the runtime). "
                            f"Interpreter={sys.executable!r}, sentencepiece_installed={has_sentencepiece}, "
                            f"tiktoken_installed={has_tiktoken}."
                        ) from exc
                    raise RuntimeError(
                        "Transformers backend tokenizer failed even though sentencepiece/tiktoken are installed. "
                        "This is likely a model/tokenizer compatibility issue. "
                        f"Original error: {exc}"
                    ) from exc
                if _is_missing_mamba_dependency_error(exc):
                    has_mamba, has_causal = _mamba_dependency_status()
                    raise RuntimeError(
                        "Transformers backend cannot load this model because Mamba runtime deps are unavailable. "
                        "Use backend='openai_compat' with a served model endpoint, or a transformers-compatible "
                        "checkpoint. "
                        f"Interpreter={sys.executable!r}, mamba_ssm_installed={has_mamba}, "
                        f"causal_conv1d_installed={has_causal}. Original error: {exc}"
                    ) from exc
                raise
        raise ValueError(f"Unsupported local backend: {self.backend!r}")


def default_nemotron_engine(
    *,
    model: str | None = None,
    backend: str | None = None,
    base_url: str | None = None,
    timeout_sec: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LocalLLMInferenceEngine:
    """
    Build a local engine configured for NVIDIA Nemotron.

    Environment variables:
    - `NEMOTRON_LOCAL_BACKEND` (`ollama`, `openai_compat`, or `transformers`)
    - `NEMOTRON_MODEL`
    - `NEMOTRON_BASE_URL`
    """
    resolved_backend = backend or getenv_any("NEMOTRON_LOCAL_BACKEND", default="transformers") or "transformers"
    if resolved_backend == "ollama":
        default_model = "nemotron-mini"
    elif resolved_backend == "transformers":
        default_model = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
    else:
        default_model = "nvidia/llama-3.1-nemotron-70b-instruct"
    resolved_model = model or getenv_any("NEMOTRON_MODEL", default=default_model) or default_model
    return LocalLLMInferenceEngine(
        model=resolved_model,
        backend=resolved_backend,
        base_url=base_url or getenv_any("NEMOTRON_BASE_URL"),
        timeout_sec=timeout_sec or float(getenv_any("NEMOTRON_TIMEOUT_SEC", default="120") or "120"),
        temperature=temperature if temperature is not None else float(getenv_any("NEMOTRON_TEMPERATURE", default="0") or "0"),
        max_tokens=max_tokens if max_tokens is not None else int(getenv_any("NEMOTRON_MAX_TOKENS", default="1024") or "1024"),
    )
