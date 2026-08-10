# config.py - Configuration for Enrich.AI Biology Assistant

import os

CONFIG = {
    # Provider used when the caller doesn't specify one.
    "default_provider": "gemini",

    # Provider registry consumed by llm_factory.get_llm().
    #   type "google"  -> ChatGoogleGenerativeAI (Gemini-specific params apply)
    #   type "openai"  -> ChatOpenAI against base_url (OpenAI-compatible API)
    "providers": {
        "gemini": {
            "type": "google",
            "label": "Google Gemini",
            "api_key_env": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            # Kept in sync with CONFIG["gemini"]["model_name"] by load_env_overrides()
            "default_model": "gemini-2.5-flash",
            "models": ["gemini-2.5-flash", "gemini-3.5-flash"],
        },
        "groq": {
            "type": "openai",
            "label": "Groq",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": ["GROQ_API_KEY"],
            # llama-3.3-70b-versatile is deliberately excluded: it emits tool calls as
            # plain text instead of native tool_calls, so the agent silently runs no
            # tools and fabricates results. gpt-oss-120b calls tools correctly.
            "default_model": "openai/gpt-oss-120b",
            "models": ["openai/gpt-oss-120b"],
        },
        "openai": {
            "type": "openai",
            "label": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": ["OPENAI_API_KEY"],
            "default_model": "gpt-5.4-mini",
            "models": ["gpt-5.4-mini"],
        },
    },

    # Models offered as FREE (no user key) in the sidebar. Only combinations
    # verified to drive the ReAct agent's tool calling reliably belong here.
    "free_models": [
        {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "label": "Gemini 2.5 Flash",
        },
    ],

    # Generation defaults for OpenAI-compatible providers (groq, openai).
    # Gemini keeps its own section below so its behaviour is untouched.
    "llm": {
        "temperature": 0.4,
        "top_p": 0.95,
        "max_tokens": 4096,
        "max_retries": 2,
    },

    "gemini": {
        # Single model for ALL tasks:
        # - ReAct reasoning agent (reasoning_engine.py) - see self.llm around line 700
        # - Visualization interpretations (visualizer.py)
        # - Plot descriptions (visualizer.py)
        # - Community naming (visualizer.py)
        # - Response summaries (app.py)
        # - API key validation (app.py)
        "model_name": "gemini-2.5-flash",
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
        "max_retries": 2,

        # Gemini-only knobs for the ReAct agent LLM (reasoning_engine.py)
        "thinking_budget": 512,
        "include_thoughts": True,
        "convert_system_message_to_human": True,

        # Higher token limit for response summaries (app.py)
        "summary_max_tokens": 8192,
    },

    # Logging configuration
    "logging": {
        "level": "INFO",
    },
}


# Environment variable overrides
def load_env_overrides():
    """Load configuration overrides from environment variables"""
    env_mapping = {
        "ENRICHAI_GEMINI_MODEL": ("gemini", "model_name"),
        "ENRICHAI_LOG_LEVEL": ("logging", "level"),
        "ENRICHAI_PROVIDER": (None, "default_provider"),
    }

    for env_var, (section, key) in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # Type conversion
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif "." in value and value.replace(".", "").isdigit():
                value = float(value)

            if section is None:
                CONFIG[key] = value
            else:
                CONFIG[section][key] = value

    # Keep the gemini provider entry aligned with the gemini section
    CONFIG["providers"]["gemini"]["default_model"] = CONFIG["gemini"]["model_name"]


# Load overrides on import
load_env_overrides()