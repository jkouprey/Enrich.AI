# tool_registry.py - Enrichr Library Registry
"""
Provides Enrichr library listing and categorization.
Used by app.py for the library browser UI.

The reasoning engine (LangGraph ReAct agent) handles all tool selection
and orchestration autonomously via its own StructuredTools.
"""

from typing import Dict, List
import logging

from tools import get_all_enrichr_libraries

logger = logging.getLogger(__name__)


def get_enrichr_libraries() -> Dict[str, List[str]]:
    """Get Enrichr libraries organized by category"""
    try:
        all_libs = get_all_enrichr_libraries()

        # Organize by category
        categories = {}
        for lib_name, lib_info in all_libs.items():
            category = lib_info.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(lib_name)

        return categories
    except Exception as e:
        logger.error(f"Error getting Enrichr libraries: {e}")
        return {}


def get_available_enrichr_libraries() -> Dict[str, List[str]]:
    """Get all available Enrichr libraries organized by category"""
    return get_enrichr_libraries()