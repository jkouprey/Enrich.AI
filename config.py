# config.py - Configuration for Enrich.AI Biology Assistant

import os

CONFIG = {
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

            CONFIG[section][key] = value


# Load overrides on import
load_env_overrides()