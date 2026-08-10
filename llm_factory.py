# llm_factory.py - Provider-agnostic LangChain chat model factory
"""
Enrich.AI - LLM factory

Returns a LangChain chat client for any configured provider so the rest of the
codebase never imports a provider SDK directly.

Supported provider types (see CONFIG["providers"] in config.py):
  - "google" -> ChatGoogleGenerativeAI (Gemini). Gemini-only params such as
    thinking_budget / include_thoughts / top_k apply here only.
  - "openai" -> ChatOpenAI pointed at the provider's base_url. Used for any
    OpenAI-compatible API (Groq, OpenAI, ...).

Provider SDKs are imported lazily so a missing optional dependency (e.g.
langchain-openai) never breaks the default Gemini path.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from config import CONFIG

logger = logging.getLogger(__name__)

# Params that only ChatGoogleGenerativeAI understands - never forwarded to
# the OpenAI-compatible path.
_GOOGLE_ONLY_PARAMS = {
    "top_k",
    "thinking_budget",
    "include_thoughts",
    "convert_system_message_to_human",
    "max_output_tokens",
    "google_api_key",
}


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Return the registry entry for a provider, or raise if unknown."""
    providers = CONFIG.get("providers", {})
    if provider not in providers:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {', '.join(sorted(providers))}"
        )
    return providers[provider]


def list_providers() -> List[str]:
    """Names of all configured providers."""
    return list(CONFIG.get("providers", {}).keys())


def resolve_api_key(provider: str) -> Optional[str]:
    """Look up an API key for a provider from CONFIG then its env vars."""
    section_key = CONFIG.get(provider, {}).get("api_key") if isinstance(CONFIG.get(provider), dict) else None
    if section_key:
        return section_key

    for env_var in get_provider_config(provider).get("api_key_env", []):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **params: Any,
):
    """
    Build a LangChain chat client.

    Args:
        provider: key in CONFIG["providers"]. Defaults to CONFIG["default_provider"].
        model: model id. Defaults to the provider's default_model.
        api_key: explicit key. Falls back to the provider's env vars.
        **params: overrides merged over the config-derived defaults.

    Returns:
        A LangChain BaseChatModel ready to bind tools.
    """
    provider = provider or CONFIG.get("default_provider", "gemini")
    provider_cfg = get_provider_config(provider)

    model = model or provider_cfg.get("default_model")
    if not model:
        raise ValueError(f"No model specified and no default_model configured for '{provider}'")

    if not api_key:
        api_key = resolve_api_key(provider)

    provider_type = provider_cfg.get("type", "openai")

    if provider_type == "google":
        return _build_google(model=model, api_key=api_key, **params)
    if provider_type == "openai":
        return _build_openai_compatible(
            provider_cfg=provider_cfg, model=model, api_key=api_key, **params
        )

    raise ValueError(f"Unsupported provider type '{provider_type}' for provider '{provider}'")


def get_utility_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    **params: Any,
):
    """
    Chat client for one-shot text generation: visualization interpretations,
    response summaries and paper-relevance scoring.

    Same provider/model as the agent, but without the agent's thinking-trace
    settings - those exist to populate the reasoning trace, and here they would
    only consume the output budget and turn the reply into content blocks.
    """
    provider = provider or CONFIG.get("default_provider", "gemini")
    provider_cfg = get_provider_config(provider)

    kwargs: Dict[str, Any] = {}
    if provider_cfg.get("type") == "google":
        kwargs["thinking_budget"] = 0
        kwargs["include_thoughts"] = False
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
    elif max_tokens:
        kwargs["max_tokens"] = max_tokens

    kwargs.update(params)
    return get_llm(provider=provider, model=model, api_key=api_key, **kwargs)


def extract_text(message: Any) -> str:
    """Pull plain text out of a LangChain reply.

    Content is a string for OpenAI-compatible providers but may be a list of
    blocks (including thinking blocks) for Gemini.
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "thinking":
                    continue
                if "text" in block:
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()

    return str(content).strip()


def complete_text(llm: Any, prompt: str) -> str:
    """Run a one-shot prompt and return plain text.

    Raises on provider errors so callers keep their own fallback behaviour.
    """
    if llm is None:
        raise ValueError("No LLM client available")
    return extract_text(llm.invoke(prompt))


def _build_google(model: str, api_key: Optional[str], **params: Any):
    """ChatGoogleGenerativeAI with the exact parameter set the agent has always used."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    gemini_config = CONFIG.get("gemini", {})
    kwargs: Dict[str, Any] = {
        "model": model,
        "google_api_key": api_key,
        "temperature": gemini_config.get("temperature", 0.4),
        "top_p": gemini_config.get("top_p", 0.95),
        "top_k": gemini_config.get("top_k", 40),
        "max_output_tokens": gemini_config.get("max_output_tokens", 4096),
        "max_retries": gemini_config.get("max_retries", 2),
        "convert_system_message_to_human": gemini_config.get("convert_system_message_to_human", True),
        "thinking_budget": gemini_config.get("thinking_budget", 512),
        "include_thoughts": gemini_config.get("include_thoughts", True),
    }
    kwargs.update(params)

    logger.info(f"Creating Gemini chat model: {model}")
    return ChatGoogleGenerativeAI(**kwargs)


def _build_openai_compatible(provider_cfg: Dict[str, Any], model: str, api_key: Optional[str], **params: Any):
    """ChatOpenAI against a configurable base_url (Groq, OpenAI, ...)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for OpenAI-compatible providers. "
            "Install it with: pip install langchain-openai"
        ) from e

    defaults = CONFIG.get("llm", {})
    kwargs: Dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": defaults.get("temperature", 0.4),
        "top_p": defaults.get("top_p", 0.95),
        "max_tokens": defaults.get("max_tokens", 4096),
        "max_retries": defaults.get("max_retries", 2),
    }

    base_url = provider_cfg.get("base_url")
    if base_url:
        kwargs["base_url"] = base_url

    # Drop Gemini-only params so they never reach ChatOpenAI
    for key, value in params.items():
        if key in _GOOGLE_ONLY_PARAMS:
            logger.debug(f"Ignoring Gemini-only param '{key}' for OpenAI-compatible provider")
            continue
        kwargs[key] = value

    logger.info(f"Creating OpenAI-compatible chat model: {model} @ {base_url or 'default endpoint'}")
    return ChatOpenAI(**kwargs)
