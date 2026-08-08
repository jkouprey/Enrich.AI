# app.py - Enrich.AI: Autonomous Biology Research Assistant
# Full integration with visualizer.py, Gemini interpretations, and beautiful UI
import socket
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _orig(*a, **k) if r[0] == socket.AF_INET]

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import hashlib
from pathlib import Path
import os
import uuid
from datetime import datetime
import traceback
import logging
from typing import Dict, List, Optional, Any
import json
import html
import re
from collections import defaultdict
import google.generativeai as genai
from PIL import Image

from reasoning_engine import create_reasoning_engine
from config import CONFIG
from tool_registry import get_available_enrichr_libraries
from visualizer import Visualizer, get_gemini_interpretation, add_pdf_download_button

# Configure logging
logging.basicConfig(level=getattr(logging, CONFIG["logging"]["level"], "INFO"))
logger = logging.getLogger(__name__)

# Load favicon
_favicon_path = Path(__file__).parent / "assets" / "dimitris_tool_icon.jpeg"
_favicon = Image.open(_favicon_path) if _favicon_path.exists() else "🧬"

st.set_page_config(
    page_title="Enrich.AI - Biology Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=_favicon
)


# ===============================================================================
# ASSET HELPERS
# ===============================================================================

def _get_logo_base64():
    """Get logo image as base64 string for embedding in HTML"""
    logo_path = Path(__file__).parent / "assets" / "dimitris_tool_icon.jpeg"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def _get_user_avatar():
    """Get user avatar path for chat messages"""
    avatar_path = Path(__file__).parent / "assets" / "user.png"
    if avatar_path.exists():
        return str(avatar_path)
    return "🧑‍🔬"  # Fallback emoji


def _get_assistant_avatar(avatar_type: str = "instant"):
    """
    Get assistant avatar path based on context.

    Args:
        avatar_type: One of:
            - "instant" : Quick answer without tools (robot2.png - welcoming)
            - "tools"   : Used function calls (robot1.png - lightbulb idea)
            - "sad"     : Response to insult (robot_sad.png)
    """
    assets_dir = Path(__file__).parent / "assets"

    avatar_map = {
        "instant": assets_dir / "robot2.png",  # Welcoming, arms open
        "tools": assets_dir / "robot1.png",  # Lightbulb, finger up (eureka!)
        "sad": assets_dir / "robot_sad.png",  # Sad face for insults
    }

    avatar_path = avatar_map.get(avatar_type, avatar_map["instant"])
    if avatar_path.exists():
        return str(avatar_path)
    return "🤖"  # Fallback emoji


def _detect_insult(user_message: str) -> bool:
    """Detect if user message contains insults or rude language"""
    insult_patterns = [
        "stupid", "dumb", "idiot", "useless", "trash", "garbage", "suck",
        "worst", "terrible", "awful", "hate you", "shut up", "fuck", "shit",
        "damn", "crap", "worthless", "pathetic", "incompetent", "moron",
        "imbecile", "retard", "loser", "jerk", "ass", "bullshit"
    ]
    message_lower = user_message.lower()
    return any(pattern in message_lower for pattern in insult_patterns)


def _determine_assistant_avatar(envelope: dict, previous_user_message: str = "") -> str:
    """
    Determine which assistant avatar to use based on context.

    Returns avatar_type: "instant", "tools", or "sad"
    """
    # Check if user was rude
    if previous_user_message and _detect_insult(previous_user_message):
        return "sad"

    # Check if tools were used
    tools_used = envelope.get("tools_used", [])
    if tools_used and len(tools_used) > 0:
        return "tools"

    # Default to instant answer avatar
    return "instant"


# ===============================================================================
# STYLING
# ===============================================================================

def apply_fancy_styling():
    """Apply theme based on dark_mode setting"""
    is_dark = st.session_state.get("dark_mode", True)
    color_scheme = "dark" if is_dark else "light"

    if is_dark:
        # Dark theme colors
        bg_primary = "#0a0e17"
        bg_secondary = "#111827"
        bg_card = "#1a1f2e"
        bg_card_hover = "#242937"
        border_color = "#2d3748"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        sidebar_bg = "linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%)"
        sidebar_btn_bg = "linear-gradient(135deg, #1e293b 0%, #334155 100%)"
        sidebar_btn_text = "#f1f5f9"  # Light text for dark theme buttons
        input_bg = "#1e293b"
    else:
        # Light theme colors
        bg_primary = "#f8fafc"
        bg_secondary = "#f1f5f9"
        bg_card = "#ffffff"
        bg_card_hover = "#f1f5f9"
        border_color = "#e2e8f0"
        text_primary = "#1e293b"
        text_secondary = "#475569"
        text_muted = "#64748b"
        sidebar_bg = "linear-gradient(180deg, #f1f5f9 0%, #e0e7ff 100%)"
        sidebar_btn_bg = "linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%)"
        sidebar_btn_text = "#1e293b"  # Dark text for light theme buttons
        input_bg = "#ffffff"

    # Dynamic theme CSS (f-string)
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

        :root {{
            color-scheme: {color_scheme};
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-card: {bg_card};
            --bg-card-hover: {bg_card_hover};
            --border-color: {border_color};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --text-muted: {text_muted};
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --gradient-1: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --gradient-2: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
                background-color: {bg_primary} !important;
            }}
        }}

        .stApp {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }}

        .main .block-container {{
            padding-top: 1rem;
            max-width: 1400px;
        }}

        [data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: 1px solid var(--border-color);
        }}

        [data-testid="stSidebar"] .stMarkdown {{
            color: var(--text-primary);
        }}

        [data-testid="stSidebar"] .stButton > button {{
            background: {sidebar_btn_bg} !important;
            border: 1px solid var(--border-color) !important;
            color: {sidebar_btn_text} !important;
            border-radius: 10px !important;
            padding: 0.5rem 0.75rem !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            transition: all 0.3s ease !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border-color: var(--accent-blue) !important;
            color: #ffffff !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        }}

        /* Main area buttons */
        .stButton > button {{
            color: {text_primary} !important;
        }}

        .stButton > button:hover {{
            color: #ffffff !important;
        }}

        .stTextInput input, .stTextArea textarea {{
            background: {input_bg} !important;
            color: {text_primary} !important;
            border-color: var(--border-color) !important;
        }}

        /* Selectbox and other inputs */
        .stSelectbox > div > div {{
            background: {input_bg} !important;
            color: {text_primary} !important;
        }}

        /* Expander text - all elements */
        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary div {{
            color: {text_primary} !important;
        }}

        /* General markdown text */
        .stMarkdown, .stMarkdown p, .stMarkdown span {{
            color: {text_primary} !important;
        }}

        /* Sidebar text */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {{
            color: {text_primary} !important;
        }}

        /* Caption text */
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {text_secondary} !important;
        }}

        /* Metric labels */
        [data-testid="stMetricLabel"] {{
            color: {text_secondary} !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {text_primary} !important;
        }}

        /* Example cards - force text color */
        .example-card,
        .example-card div {{
            color: {text_primary} !important;
        }}

        /* Chat messages - force text color */
        [data-testid="stChatMessage"],
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] .stMarkdown {{
            color: {text_primary} !important;
        }}

        /* ========================================= */
        /* SIDEBAR EXPANDERS - THEME COLORS         */
        /* ========================================= */

        /* Expander container */
        [data-testid="stSidebar"] div[data-testid="stExpander"] {{
            background-color: {bg_card} !important;
        }}

        /* ===== Tables + download buttons: follow theme (fixes Windows light-theme leak in dark mode) ===== */
        [data-testid="stDataFrame"] {{
            background-color: {bg_card} !important;
        }}

        /* Table header - slightly lighter than cells */
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] thead th {{
            background-color: {bg_card_hover} !important;
            color: {text_primary} !important;
        }}

      /* Download-as-CSV button - unhovered follows theme */
        [data-testid="stDownloadButton"] > button {{
            background-color: {bg_card} !important;
            color: {text_primary} !important;
            border: 1px solid {border_color} !important;
        }}
        /* Download button - hover (kept so hovering still works) */
        [data-testid="stDownloadButton"] > button:hover {{
            background-color: {bg_card_hover} !important;
            border-color: #3b82f6 !important;
            color: {text_primary} !important;
        }}

        /* Dataframe hover toolbar */
        [data-testid="stElementToolbar"] {{
            background-color: {bg_card} !important;
        }}
        [data-testid="stElementToolbar"] button {{
            color: {text_primary} !important;
        }}
        [data-testid="stElementToolbar"] button:hover {{
            background-color: {bg_card_hover} !important;
        }}

        /* Expander header (closed state) - follows theme */
        [data-testid="stSidebar"] div[data-testid="stExpander"] > details > summary {{
            background-color: {bg_card} !important;
            color: {text_primary} !important;
        }}

        /* Expander content wrapper when OPEN */
        [data-testid="stSidebar"] div[data-testid="stExpander"] > details[open] > div[data-testid="stExpanderDetails"] {{
            background-color: {bg_card} !important;
            color: {text_primary} !important;
        }}

        /* Inner vertical block in expander */
        [data-testid="stSidebar"] div[data-testid="stExpander"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {bg_card} !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {{
            background-color: {bg_card} !important;
        }}

        /* All text elements inside sidebar expanders */
        [data-testid="stSidebar"] div[data-testid="stExpander"] p,
        [data-testid="stSidebar"] div[data-testid="stExpander"] span,
        [data-testid="stSidebar"] div[data-testid="stExpander"] div,
        [data-testid="stSidebar"] div[data-testid="stExpander"] label {{
            color: {text_primary} !important;
        }}

        /* Caption text in sidebar */
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: {text_secondary} !important;
        }}

        /* ========================================= */
        /* ALL EXPANDERS - GLOBAL THEME COLORS      */
        /* (Works for sidebar + main content)       */
        /* ========================================= */

        /* ALL expander containers */
        div[data-testid="stExpander"] {{
            background-color: {bg_card} !important;
        }}

        /* ALL expander headers */
        div[data-testid="stExpander"] > details > summary {{
            background-color: {bg_card} !important;
            color: {text_primary} !important;
        }}

        /* ALL expander open content */
        div[data-testid="stExpander"] > details[open] > div[data-testid="stExpanderDetails"] {{
            background-color: {bg_card} !important;
            color: {text_primary} !important;
        }}

        /* ALL inner blocks */
        div[data-testid="stExpander"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {bg_card} !important;
        }}

        div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {{
            background-color: {bg_card} !important;
        }}

        /* ALL text inside expanders */
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span,
        div[data-testid="stExpander"] div,
        div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] code,
        div[data-testid="stExpander"] pre {{
            color: {text_primary} !important;
        }}

        /* Captions */
        div[data-testid="stExpander"] .stCaption,
        div[data-testid="stExpander"] [data-testid="stCaptionContainer"] {{
            color: {text_secondary} !important;
        }}

        /* ========================================= */
        /* CHAT INPUT - FORCE WHITE TEXT            */
        /* ========================================= */

        /* Every possible selector for the textarea */
        textarea[data-testid="stChatInputTextArea"],
        [data-testid="stChatInputTextArea"],
        textarea[aria-label*="genes"],
        textarea[placeholder*="genes"],
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInputContainer"] textarea {{
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
        }}

        textarea[data-testid="stChatInputTextArea"]::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {{
            color: var(--text-muted) !important;
        }}

        /* ========================================= */
        /* CODE BLOCKS - ALWAYS LIGHT TEXT ON DARK  */
        /* (code blocks keep dark bg in both themes)*/
        /* ========================================= */

        /* Target ALL code blocks globally */
        [data-testid="stCodeBlock"],
        [data-testid="stCodeBlock"] > div,
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code,
        pre[class*="language-"],
        code[class*="language-"] {{
            color: #e2e8f0 !important;
            background-color: #1e293b !important;
            -webkit-text-fill-color: #e2e8f0 !important;
        }}

        /* Inside expanders */
        div[data-testid="stExpander"] pre,
        div[data-testid="stExpander"] code,
        div[data-testid="stExpander"] [data-testid="stCodeBlock"],
        div[data-testid="stExpander"] [data-testid="stCodeBlock"] pre,
        div[data-testid="stExpander"] [data-testid="stCodeBlock"] code {{
            color: #e2e8f0 !important;
            background-color: #1e293b !important;
            -webkit-text-fill-color: #e2e8f0 !important;
        }}

        /* Syntax highlighting tokens */
        [data-testid="stCodeBlock"] .token,
        pre .token,
        code .token {{
            color: #e2e8f0 !important;
        }}

        /* ========================================= */
        /* LOADING SPINNER - VISIBLE IN LIGHT MODE  */
        /* ========================================= */

        /* Target spinner text */
        [data-testid="stSpinner"],
        [data-testid="stSpinner"] > div,
        [data-testid="stSpinner"] p,
        [data-testid="stSpinner"] span {{
            color: {text_primary} !important;
        }}

        /* The actual spinning animation - target SVG and CSS animations */
        [data-testid="stSpinner"] svg {{
            color: {text_primary} !important;
        }}

        [data-testid="stSpinner"] svg circle,
        [data-testid="stSpinner"] svg path {{
            stroke: {text_primary} !important;
            fill: {text_primary} !important;
        }}

        /* CSS spinner animation */
        .stSpinner > div,
        [data-testid="stSpinner"] > div > div {{
            border-color: {text_primary} transparent transparent transparent !important;
            border-top-color: {text_primary} !important;
        }}

        /* Target any div with spinning animation */
        [data-testid="stSpinner"] div[class*="spinner"],
        [data-testid="stSpinner"] div[class*="Spinner"] {{
            border-color: {text_primary} transparent transparent transparent !important;
        }}

        /* Emoji in spinner text */
        [data-testid="stSpinner"] [data-testid="stMarkdownContainer"] {{
            color: {text_primary} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Static CSS (regular string - no f-string needed)
    st.markdown("""
        <style>
        /* Pulse animation for status dot */
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(0.9); }
        }

        /* Dashboard header - kept for backwards compatibility */
        .dashboard-header {
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 1rem;
        }

        .dashboard-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .dashboard-subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
        }

        /* Example cards */
        .example-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            cursor: pointer;
            transition: all 0.3s ease;
            height: 100%;
            min-height: 130px;
            color: var(--text-primary);
        }

        .example-card div {
            color: var(--text-primary) !important;
        }

        .example-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(6, 182, 212, 0.15);
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 14px !important;
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div {
            color: var(--text-primary) !important;
        }

        /* Bigger chat avatars - targeting by alt attribute which ACTUALLY EXISTS */
        img[alt="user avatar"],
        img[alt="assistant avatar"] {
            width: 80px !important;
            height: 80px !important;
            min-width: 80px !important;
            min-height: 80px !important;
            max-width: 80px !important;
            max-height: 80px !important;
            border-radius: 14px !important;
            object-fit: cover !important;
            flex-shrink: 0 !important;
        }

        /* Summary box - MUCH BIGGER */
        .summary-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.18) 0%, rgba(139, 92, 246, 0.18) 100%);
            border: 2px solid rgba(59, 130, 246, 0.5);
            border-radius: 20px;
            padding: 2.5rem 3rem;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
        }

        .summary-title {
            color: var(--accent-cyan);
            font-weight: 700;
            font-size: 1.4rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .summary-text {
            color: var(--text-primary);
            font-size: 1.15rem;
            line-height: 1.9;
        }

        /* Yellow pin button */
        .pin-btn-yellow button {
            background-color: #f59e0b !important;
            color: white !important;
            border: none !important;
        }

        .pin-btn-yellow button:hover {
            background-color: #d97706 !important;
        }

        /* Viz grid card */
        .viz-mini-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem;
            transition: all 0.3s ease;
            cursor: pointer;
            height: 100%;
        }

        .viz-mini-card:hover {
            border-color: var(--accent-cyan);
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.15);
        }

        .viz-mini-title {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .viz-mini-info {
            color: var(--text-muted);
            font-size: 0.7rem;
        }

        /* Full viz panel */
        .full-viz-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }

        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span,
        .streamlit-expanderHeader svg {
            color: var(--text-primary) !important;
            fill: var(--text-primary) !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary p {
            color: var(--text-primary) !important;
        }

        .streamlit-expanderHeader:hover {
            border-color: var(--accent-cyan) !important;
        }

        .streamlit-expanderContent {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
        }

        /* Inputs */
        .stTextInput > div > div > input {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--accent-cyan) !important;
        }

        /* Chat input */
        [data-testid="stBottomBlockContainer"] {
            background-color: var(--bg-primary) !important;
        }
        [data-testid="stBottom"] > div {
            background-color: var(--bg-primary) !important;
        }
        [data-testid="stChatInput"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 14px !important;
        }

        [data-testid="stChatInput"] textarea {
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] div[data-baseweb="base-input"],
        [data-testid="stChatInput"] div[data-baseweb="textarea"] {
            background: var(--bg-card) !important;
}

        /* Buttons - General */
        .stButton > button {
            background: var(--gradient-1) !important;
            border: none !important;
            border-radius: 10px !important;
            color: white !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
        }

        .stButton > button:disabled {
            background: #334155 !important;
            color: #64748b !important;
        }

        /* ============= COLUMN HEADERS ============= */
        .column-header {
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 1rem;
        }

        .column-header-terms {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
        }

        .column-header-clusters {
            background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
            color: white;
        }

        .column-header-other {
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
            color: white;
        }

        /* ============= PLOT CARD STYLING ============= */
        .plot-card {
            background: #1a1f2e;
            border: 2px solid #3b82f6;
            border-radius: 12px;
            padding: 0.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }

        .plot-card:hover {
            border-color: #10b981;
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
            transform: translateY(-2px);
        }

        .plot-card-btn {
            display: block;
            width: 100%;
            background: var(--bg-card);
            border: 2px solid var(--accent-blue);
            border-radius: 10px;
            color: var(--text-primary);
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.75rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 0.5rem;
        }

        .plot-card-btn:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border-color: #10b981;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
            transform: translateY(-3px);
            color: white;
        }

        /* Force ALL buttons in main content to have fancy style */
        .main .stButton > button {
            background: var(--bg-card) !important;
            border: 2px solid var(--accent-blue) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            padding: 0.75rem 1rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .main .stButton > button:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border-color: #10b981 !important;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
            transform: translateY(-3px) !important;
            color: white !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-card);
            border-radius: 10px;
            padding: 0.3rem;
            gap: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--gradient-1) !important;
            color: white !important;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem;
        }

        [data-testid="stMetric"] label {
            color: var(--text-secondary) !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-family: 'Outfit', sans-serif !important;
        }

        /* API status */
        .api-valid { color: #10b981; font-size: 0.8rem; margin-top: 0.25rem; }
        .api-invalid { color: #ef4444; font-size: 0.8rem; margin-top: 0.25rem; }

        /* Follow-up */
        .followup-container {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
            border: 1px solid rgba(6, 182, 212, 0.2);
            border-radius: 10px;
            padding: 0.75rem;
            margin-top: 0.75rem;
        }

        /* Interpretation box */
        .interpretation-box {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            padding: 1.25rem;
            margin: 1rem 0;
        }

        .interpretation-title {
            color: var(--accent-purple);
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Overview/tool cards */
        .overview-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .overview-card:hover {
            border-color: var(--accent-cyan);
        }

        .overview-card h3 {
            color: var(--accent-cyan);
            font-family: 'Outfit', sans-serif;
            margin-bottom: 0.75rem;
        }

        .tool-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            height: 100%;
            transition: all 0.3s ease;
        }

        .tool-card:hover {
            border-color: var(--accent-purple);
            transform: translateY(-2px);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ============= WIDE DIALOG STYLING ============= */
        div[data-testid="stDialog"] div[role="dialog"]:has(.wide-dialog) {
            width: 98vw !important;
            max-width: 2000px !important;
            height: 92vh !important;
        }

        div[data-testid="stDialog"] div[role="dialog"]:has(.wide-dialog) > div {
            max-height: 88vh !important;
            overflow-y: auto !important;
        }

        /* Also apply to all dialogs as fallback */
        div[data-testid="stModal"] > div[role="dialog"] {
            width: 95vw !important;
            max-width: 1900px !important;
        }

        /* ============= FORCE SQUARE THUMBNAILS ============= */
        /* REMOVED - was constraining thumbnails to 180px/200px */

        /* ============= ENRICHR-STYLE PLOT GRID ============= */

        /* Themed column headers */
        .column-header {
            padding: 0.75rem 1rem;
            border-radius: 10px 10px 0 0;
            margin-bottom: 0.5rem;
            text-align: center;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .column-header-terms {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
        }

        .column-header-clusters {
            background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
            color: white;
        }

        .column-header-other {
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
            color: white;
        }

        /* Plot thumbnail card */
        .plot-thumb-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .plot-thumb-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.2);
        }

        .plot-thumb-card:hover::after {
            content: '🔍 Click to expand';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(6, 182, 212, 0.9));
            color: white;
            text-align: center;
            padding: 0.5rem;
            font-size: 0.75rem;
        }

        .plot-thumb-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }

        .plot-thumb-desc {
            color: var(--text-muted);
            font-size: 0.7rem;
            line-height: 1.4;
        }

        /* Modal/Dialog styling */
        .plot-modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 2rem;
            overflow-y: auto;
        }

        .plot-modal-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            width: 100%;
            max-width: 1200px;
            padding: 1.5rem;
            position: relative;
        }

        .plot-modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .plot-modal-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--accent-cyan);
        }

        .plot-modal-close {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .plot-modal-close:hover {
            background: var(--accent-red);
            border-color: var(--accent-red);
        }

        /* Parameters panel in modal */
        .params-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }

        .params-panel-title {
            color: var(--accent-purple);
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        /* Legend/Interpretation box in modal */
        .plot-legend-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 1.25rem;
            margin-top: 1rem;
        }

        .plot-legend-title {
            color: var(--accent-green);
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .plot-legend-text {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.7;
        }

        /* Expand details section */
        .details-expand-btn {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            color: var(--text-primary);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            width: 100%;
            margin: 1rem 0;
        }

        .details-expand-btn:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border-color: var(--accent-blue);
        }

        /* ============= STREAMLIT DIALOG STYLING ============= */
        /* Make the st.dialog popup LARGER initial size */
        [data-testid="stModal"] > div:first-child,
        div[data-testid="stModal"] > div[data-testid="stModalContent"],
        section[data-testid="stModalContent"] {
            max-width: 98vw !important;
            width: 1600px !important;
            min-width: 800px !important;
            max-height: 95vh !important;
            min-height: 600px !important;
        }

        /* Modal dialog inner container */
        [data-testid="stModalContentContainer"],
        div[data-testid="stModalContentContainer"] {
            overflow-y: auto !important;
            max-height: 90vh !important;
            padding: 1.5rem !important;
        }

        /* Dialog header styling */
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] h3 {
            color: var(--accent-cyan) !important;
        }

        /* ============= SQUARE PLOT THUMBNAILS ============= */
        /* REMOVED max-width constraint - let thumbnails fill column */

        /* Remove the hover ::after text since we have a title button */
        .plot-thumb-card:hover::after {
            display: none !important;
            content: none !important;
        }

        /* ============= CHAT AVATAR SIZING - Using alt attribute ============= */
        img[alt="user avatar"],
        img[alt="assistant avatar"] {
            width: 80px !important;
            height: 80px !important;
            min-width: 80px !important;
            min-height: 80px !important;
            max-width: 80px !important;
            max-height: 80px !important;
            border-radius: 14px !important;
            object-fit: cover !important;
            flex-shrink: 0 !important;
        }

        </style>
    """, unsafe_allow_html=True)


# ===============================================================================
# SESSION STATE
# ===============================================================================

def init_session_state():
    """Initialize session state"""
    defaults = {
        "api_key": os.environ.get("GOOGLE_API_KEY", ""),
        "api_key_valid": None,
        "reasoning_engine": None,
        "gemini_model": None,
        "visualizer": None,
        "current_page": "chat",
        "chat_sessions": [],
        "current_session": {
            "id": str(uuid.uuid4()),
            "title": "New Chat",
            "messages": [],
            "created_at": datetime.now().isoformat()
        },
        "selected_libraries": [],
        "enrichment_results": None,
        "expanded_viz": None,  # Which library visualization is expanded
        "expanded_plot": None,  # Which plot type is expanded in modal
        "pending_query": None,
        "pinned_messages": [],  # List of pinned message indices
        "dark_mode": True,  # Theme: True = dark, False = light
        "processing": False,  # Flag to prevent double-submission during analysis
        "scroll_to_bottom": False,  # Auto-scroll after new message
        "scroll_to_message": None,  # Scroll to specific message index
        "all_enrichr_libs": [],  # Cached flat list of Enrichr library names
        "lib_expander_states": {},  # Library browser expander states
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def validate_api_key(api_key: str) -> bool:
    """Check if API key is valid by pinging the model"""
    if not api_key or len(api_key) < 10:
        return False
    try:
        genai.configure(api_key=api_key)
        model_name = CONFIG.get("gemini", {}).get("model_name", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hi", generation_config={"max_output_tokens": 5})
        return True
    except Exception as e:
        logger.warning(f"API key validation failed: {e}")
        return False


def get_gemini_model():
    """Get or create the Gemini model instance.

    Uses the same model as the ReAct agent (CONFIG model_name).
    Used for: visualization interpretations, plot descriptions,
    response summaries, community naming.
    """
    if st.session_state.gemini_model is None and st.session_state.api_key:
        try:
            genai.configure(api_key=st.session_state.api_key)
            model_name = CONFIG.get("gemini", {}).get("model_name", "gemini-2.5-flash")
            st.session_state.gemini_model = genai.GenerativeModel(model_name)
        except Exception as e:
            logger.warning(f"Failed to create Gemini model: {e}")
    return st.session_state.gemini_model


def get_visualizer():
    """Get or create visualizer"""
    if st.session_state.visualizer is None:
        gemini = get_gemini_model()
        st.session_state.visualizer = Visualizer(gemini_model=gemini)
    return st.session_state.visualizer


# ===============================================================================
# SIDEBAR
# ===============================================================================

# Callback functions for sidebar (must be at module level)
def unpin_from_sidebar(idx):
    """Callback to unpin message from sidebar"""
    if idx in st.session_state.pinned_messages:
        st.session_state.pinned_messages.remove(idx)


def scroll_to_message(idx):
    """Callback to scroll to a specific message"""
    st.session_state.scroll_to_message = idx


def inject_scroll_js(element_id):
    """Inject JavaScript to scroll to an element"""
    import streamlit.components.v1 as components
    import time
    nonce = int(time.time() * 1000)
    components.html(f'''
        <script>
            (function() {{
                const _ = {nonce};
                let tries = 0;
                function scroll() {{
                    const el = window.parent.document.getElementById("{element_id}");
                    if (el) {{
                        el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                    }} else if (tries++ < 20) {{
                        setTimeout(scroll, 50);
                    }}
                }}
                scroll();
            }})();
        </script>
    ''', height=0)


def toggle_theme():
    """Callback to toggle light/dark theme"""
    st.session_state.dark_mode = not st.session_state.get("dark_mode", True)


def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        # Logo - actual image
        logo_b64 = _get_logo_base64()
        st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <img src="data:image/jpeg;base64,{logo_b64}" 
                     style="width: 120px; height: 120px; border-radius: 16px; object-fit: cover; 
                            box-shadow: 0 4px 15px rgba(59,130,246,0.3); border: 2px solid rgba(59,130,246,0.2);" />
                <h2 style="margin: 0.5rem 0 0 0; font-family: 'Outfit', sans-serif;
                    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                    font-weight: 700; font-size: 1.4rem;">
                    Enrich.AI
                </h2>
                <p style="color: #64748b; font-size: 0.7rem; margin-top: 0;">Biology Research Assistant</p>
            </div>
        """, unsafe_allow_html=True)

        # New Chat
        if st.button("➕ New Chat", use_container_width=True, key="new_chat"):
            start_new_chat()

        # Pinned Messages (if any) - just show with unpin option
        pinned = st.session_state.get("pinned_messages", [])
        messages = st.session_state.current_session.get("messages", [])
        valid_pinned = [idx for idx in pinned if idx < len(messages)]

        if valid_pinned:
            with st.expander("📌 Pinned Messages", expanded=True):
                for idx in valid_pinned:
                    msg = messages[idx]
                    if msg.get("role") == "assistant":
                        envelope = msg.get("envelope", {})
                        content = msg.get("content", "")

                        # Create preview
                        if envelope.get("tools_used"):
                            tools = envelope.get("tools_used", [])
                            preview = f"{', '.join(tools[:2])}"[:25]
                        else:
                            preview = content[:25] + "..."

                        msg_num = (idx // 2) + 1

                        col1, col2 = st.columns([0.85, 0.15])
                        with col1:
                            # Clickable button to jump to message
                            st.button(
                                f"#{msg_num}: {preview}",
                                key=f"goto_pin_{idx}",
                                on_click=scroll_to_message,
                                args=(idx,),
                                help="Jump to message",
                                use_container_width=True
                            )
                        with col2:
                            st.button(
                                "✕",
                                key=f"unpin_sb_{idx}",
                                on_click=unpin_from_sidebar,
                                args=(idx,),
                                help="Unpin"
                            )

        # Chat Sessions - expanded by default
        with st.expander("💬 Chat Sessions", expanded=True):
            if st.session_state.chat_sessions:
                for session in reversed(st.session_state.chat_sessions[-8:]):
                    is_current = session["id"] == st.session_state.current_session.get("id")
                    if st.button(f"{'🔵 ' if is_current else '⚪ '}{session['title'][:20]}...",
                                 key=f"sess_{session['id']}", use_container_width=True):
                        load_chat_session(session["id"])
            else:
                st.caption("No previous chats")

        # Configuration - centered header
        st.markdown("<h5 style='text-align: center;'>⚙️ Configuration</h5>", unsafe_allow_html=True)

        api_key = st.text_input("Gemini API Key", value=st.session_state.api_key,
                                type="password", key="api_key_input", placeholder="Enter API key...")

        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.api_key_valid = None
            st.session_state.reasoning_engine = None
            st.session_state.gemini_model = None
            st.session_state.visualizer = None

        # Center the check button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Check", key="check_api", use_container_width=True):
                if api_key:
                    with st.spinner("..."):
                        st.session_state.api_key_valid = validate_api_key(api_key)
                    st.rerun()

        if st.session_state.api_key_valid is True:
            st.markdown('<p class="api-valid" style="text-align: center;">✓ API key is valid</p>',
                        unsafe_allow_html=True)
        elif st.session_state.api_key_valid is False:
            st.markdown('<p class="api-invalid" style="text-align: center;">✗ Invalid API key</p>',
                        unsafe_allow_html=True)

        # Enrichr Libraries
        with st.expander("📚 Enrichr Libraries", expanded=False):
            st.caption("Quick Select:")
            col1, col2 = st.columns(2)
            with col1:
                go_sel = "GO_Biological_Process_2023" in st.session_state.selected_libraries
                if st.button(f"{'✓ ' if go_sel else ''}GO BP", key="quick_go", use_container_width=True):
                    toggle_library("GO_Biological_Process_2023")
            with col2:
                kegg_sel = "KEGG_2021_Human" in st.session_state.selected_libraries
                if st.button(f"{'✓ ' if kegg_sel else ''}KEGG", key="quick_kegg", use_container_width=True):
                    toggle_library("KEGG_2021_Human")

            if st.session_state.selected_libraries:
                st.caption(f"Selected: {len(st.session_state.selected_libraries)}")

            if st.button("📋 Browse All 222 Libraries", use_container_width=True, key="browse_libs"):
                st.session_state.current_page = "libraries"
                st.rerun()

        # Navigation
        with st.expander("📖 Navigation", expanded=False):
            if st.button("🏠 Chat", use_container_width=True, key="nav_chat"):
                st.session_state.current_page = "chat"
                st.rerun()
            if st.button("📊 Architecture Overview", use_container_width=True, key="nav_overview"):
                st.session_state.current_page = "overview"
                st.rerun()

        # GitHub link at bottom
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem 0; margin-top: 1rem;">
                <a href="https://github.com/jkouprey/Enrich.AI" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-block; text-decoration: none; transition: transform 0.2s ease;"
                   onmouseover="this.style.transform='translateY(-3px)'" 
                   onmouseout="this.style.transform='translateY(0)'">
                    <svg height="32" width="32" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
            </div>
        """, unsafe_allow_html=True)


def toggle_library(library: str):
    if library in st.session_state.selected_libraries:
        st.session_state.selected_libraries.remove(library)
    else:
        st.session_state.selected_libraries.append(library)
    st.rerun()


# ===============================================================================
# CHAT MANAGEMENT
# ===============================================================================

def start_new_chat():
    if st.session_state.current_session.get("messages"):
        save_current_session()
    st.session_state.current_session = {
        "id": str(uuid.uuid4()), "title": "New Chat", "messages": [],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_page = "chat"
    st.rerun()


def save_current_session():
    session = st.session_state.current_session
    if session["title"] == "New Chat" and session["messages"]:
        session["title"] = session["messages"][0].get("content", "")[:25] + "..."

    existing_idx = next((i for i, s in enumerate(st.session_state.chat_sessions)
                         if s["id"] == session["id"]), None)
    if existing_idx is not None:
        st.session_state.chat_sessions[existing_idx] = session.copy()
    else:
        st.session_state.chat_sessions.append(session.copy())


def load_chat_session(session_id: str):
    save_current_session()
    for session in st.session_state.chat_sessions:
        if session["id"] == session_id:
            st.session_state.current_session = session.copy()
            st.session_state.current_page = "chat"
            st.rerun()
            break


# ===============================================================================
# DASHBOARD
# ===============================================================================

def render_dashboard():
    # Theme toggle in top right
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        is_dark = st.session_state.get("dark_mode", True)
        theme_icon = "🌙" if is_dark else "☀️"
        st.button(
            theme_icon,
            key="theme_toggle",
            on_click=toggle_theme,
            help="Toggle light/dark theme"
        )

    # Fancy Dashboard Header Container - adapts to theme
    logo_b64 = _get_logo_base64()
    is_dark = st.session_state.get("dark_mode", True)

    if is_dark:
        # Dark mode styles
        container_bg = "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"
        container_border = "1px solid rgba(59,130,246,0.2)"
        container_shadow = "0 10px 40px rgba(0,0,0,0.3)"
        circle1_bg = "linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(6,182,212,0.08) 100%)"
        circle2_bg = "linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,182,212,0.08) 100%)"
        logo_shadow = "0 8px 25px rgba(59,130,246,0.3)"
        logo_border = "3px solid rgba(59,130,246,0.3)"
        subtitle_color = "#e2e8f0"
    else:
        # Light mode styles
        container_bg = "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)"
        container_border = "1px solid rgba(59,130,246,0.1)"
        container_shadow = "0 10px 40px rgba(0,0,0,0.08)"
        circle1_bg = "linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(6,182,212,0.05) 100%)"
        circle2_bg = "linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(6,182,212,0.05) 100%)"
        logo_shadow = "0 8px 25px rgba(59,130,246,0.2)"
        logo_border = "3px solid rgba(59,130,246,0.1)"
        subtitle_color = "#000000"

    header_html = f"""
<div style="position: relative; background: {container_bg}; border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; box-shadow: {container_shadow}; border: {container_border}; overflow: hidden;">
    <div style="position: absolute; top: -30px; right: -30px; width: 150px; height: 150px; background: {circle1_bg}; border-radius: 50%;"></div>
    <div style="position: absolute; bottom: -20px; left: -20px; width: 100px; height: 100px; background: {circle2_bg}; border-radius: 50%;"></div>
    <div style="position: relative; display: flex; align-items: center; gap: 1.5rem; z-index: 1;">
        <div style="width: 90px; height: 90px; border-radius: 16px; overflow: hidden; box-shadow: {logo_shadow}; border: {logo_border}; flex-shrink: 0;">
            <img src="data:image/jpeg;base64,{logo_b64}" style="width: 100%; height: 100%; object-fit: cover;" />
        </div>
        <div>
            <h1 style="margin: 0; font-family: 'Outfit', sans-serif; font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.5px;">Enrich.AI</h1>
            <p style="margin: 0.5rem 0 0 0; color: {subtitle_color}; font-size: 1.1rem; font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">
                <span style="display: inline-block; width: 8px; height: 8px; background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%); border-radius: 50%; animation: pulse 2s infinite;"></span>
                Your AI-powered biology research companion
            </p>
        </div>
    </div>
    <div style="position: absolute; bottom: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 50%, #10b981 100%); border-radius: 0 0 20px 20px;"></div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)

    st.markdown("#### 💡 Try an Example")

    examples = [
        {"icon": "🔬", "title": "Gene Deep Dive", "desc": "Explore TP53 function",
         "query": "Tell me about the TP53 gene"},
        {"icon": "📚", "title": "Literature Search", "desc": "Find CAR-T papers",
         "query": "Find recent papers on CAR-T cell therapy in solid tumors"},
        {"icon": "📊", "title": "Pathway Enrichment", "desc": "Analyze immune genes",
         "query": "Run enrichment analysis on IL6, TNF, IL1B, CXCL8, CCL2, IFNG"},
        {"icon": "🗄️", "title": "Database Query", "desc": "Search Enrichr libraries",
         "query": "Search the Enrichr database for all KEGG pathways and GO BP terms that contain PTPRC and CD8A"},
        {"icon": "🧬", "title": "Complex Analysis", "desc": "Combine functions",
         "query": "Create a list of 50 genes with immune cell genes, perform enrichment analysis with GO BP and identify which of these terms are more related in the context of lung cancer."},
    ]

    cols = st.columns(5)  # 5 columns for 5 cards
    for col, ex in zip(cols, examples):
        with col:
            st.markdown(f"""
                <div class="example-card" style="padding: 0.75rem; min-height: 90px;">
                    <div style="font-size: 1.3rem; margin-bottom: 0.3rem;">{ex['icon']}</div>
                    <div style="font-weight: 600; margin-bottom: 0.15rem; font-size: 0.85rem; color: inherit;">{ex['title']}</div>
                    <div style="font-size: 0.7rem; line-height: 1.3; color: inherit; opacity: 0.8;">{ex['desc']}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Use", key=f"use_{ex['title']}", use_container_width=True):
                st.session_state.pending_query = ex['query']
                st.rerun()

    st.markdown("<hr style='border-color: var(--border-color); margin: 1rem 0;'>", unsafe_allow_html=True)


# ===============================================================================
# ENRICHMENT VISUALIZATION - ENRICHR-STYLE WITH REAL POPUP MODALS
# ===============================================================================

# Fixed thumbnail size for all plots (square)


# Plot interpretation moved to Visualizer.get_plot_interpretation() in visualizer.py


# Define plot configurations
# Column 1: Term Analysis - Bar, Bubble, UpSet
# Column 2: Clustering - Dendrogram, Similarity
# Column 3: Networks - Cnetplot
PLOT_CONFIGS = {
    "term_plots": [
        {"id": "bar", "icon": "📊", "title": "Bar Plot", "desc": "Significance ranking"},
        {"id": "bubble", "icon": "🫧", "title": "Bubble Chart", "desc": "Size = gene count"},
        {"id": "upset", "icon": "📊", "title": "UpSet Plot", "desc": "Set intersections"},
    ],
    "cluster_plots": [
        {"id": "dendrogram", "icon": "🌳", "title": "Dendrogram", "desc": "Hierarchical clustering"},
        {"id": "similarity", "icon": "🧬", "title": "Similarity", "desc": "Term grouping"},
    ],
    "other_plots": [
        {"id": "cnetplot", "icon": "🗺️", "title": "Cnetplot", "desc": "Term-gene network"},
    ]
}


# ==================== DIALOG FUNCTIONS ====================

@st.dialog("📊 Bar Plot", width="large")
def show_bar_dialog():
    """Modal dialog for Bar Plot with term selection"""
    df = st.session_state.get("_enrichment_df")
    if df is None:
        st.error("No data available")
        return
    _render_bar_bubble_dialog("bar", df)


@st.dialog("🫧 Bubble Chart", width="large")
def show_bubble_dialog():
    """Modal dialog for Bubble Chart with term selection"""
    df = st.session_state.get("_enrichment_df")
    if df is None:
        st.error("No data available")
        return
    _render_bar_bubble_dialog("bubble", df)


@st.dialog("🌳 Dendrogram", width="large")
def show_dendrogram_dialog():
    df = st.session_state.get("_enrichment_df")
    if df is None:
        st.error("No data available")
        return
    _render_simple_dialog("dendrogram", df)


@st.dialog("🧬 Similarity Clusters", width="large")
def show_similarity_dialog():
    df = st.session_state.get("_enrichment_df")
    if df is None:
        st.error("No data available")
        return
    _render_simple_dialog("similarity", df)


@st.dialog("🗺️ Concept Network", width="large")
def show_cnetplot_dialog():
    df = st.session_state.get("_enrichment_df")
    if df is None:
        st.error("No data available")
        return
    _render_simple_dialog("cnetplot", df)


@st.dialog("📊 UpSet Plot", width="large")
def show_upset_dialog():
    df = st.session_state.get("_enrichment_df")
    if df is None:
        st.error("No data available")
        return
    _render_simple_dialog("upset", df)


DIALOG_FUNCTIONS = {
    "bar": show_bar_dialog,
    "bubble": show_bubble_dialog,
    "dendrogram": show_dendrogram_dialog,
    "similarity": show_similarity_dialog,
    "cnetplot": show_cnetplot_dialog,
    "upset": show_upset_dialog,
}


def _render_bar_bubble_dialog(plot_id: str, df: pd.DataFrame):
    """Render Bar/Bubble dialog with term selection and deselect all"""
    # Add marker for wide dialog CSS
    st.markdown("<span class='wide-dialog'></span>", unsafe_allow_html=True)

    visualizer = get_visualizer()

    # Get all available terms sorted by significance
    all_terms = df.nsmallest(100, 'adjusted_p_value')['term'].tolist()

    # Get unique libraries for color-by-library option
    libraries = df['library'].unique().tolist() if 'library' in df.columns else []

    # Check available metrics
    has_combined_score = 'combined_score' in df.columns
    has_z_score = 'z_score' in df.columns

    # Parameters in SLIDING EXPANDER
    with st.expander("Parameters", expanded=True):
        # Row 1: Number of terms, X-axis, Color by
        col1, col2, col3 = st.columns(3)
        with col1:
            n_terms = st.slider("Number of terms", 5, min(50, len(all_terms)), 10, key=f"dlg_{plot_id}_n")
        with col2:
            # X-axis options differ for bar vs bubble
            if plot_id == "bar":
                x_options = ["Gene Count", "-log10(p-value)"]
                if has_combined_score:
                    x_options.append("Combined Score")
                x_axis = st.selectbox("X-axis", x_options, key=f"dlg_{plot_id}_xaxis")
            else:  # bubble
                x_options = ["-log10(p-value)"]
                if has_combined_score:
                    x_options.append("Combined Score")
                x_options.append("Gene Ratio")
                x_axis = st.selectbox("X-axis", x_options, key=f"dlg_{plot_id}_xaxis")
        with col3:
            color_options = ["-log10(p-value)"]
            if has_combined_score:
                color_options.append("Combined Score")
            color_options.append("Library")
            color_by = st.selectbox("Color by", color_options, key=f"dlg_{plot_id}_colorby")

        # Row 2: Palette, Width, Height, Font Size
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            palette = st.selectbox("Color palette", ["Reds", "Blues", "Viridis", "Plasma", "Inferno", "Cividis"],
                                   key=f"dlg_{plot_id}_pal")
        with col2:
            width = st.slider("Width (px)", 200, 1500, 800, step=50, key=f"dlg_{plot_id}_w")
        with col3:
            height = st.slider("Height (px)", 200, 1500, 550, step=50, key=f"dlg_{plot_id}_h")
        with col4:
            font_size = st.slider("Font size", 8, 16, 11, key=f"dlg_{plot_id}_fs")

        # Track n_terms changes to auto-update selection
        prev_n_key = f"prev_n_terms_{plot_id}"
        if prev_n_key not in st.session_state:
            st.session_state[prev_n_key] = n_terms

        # If n_terms slider changed, update the term selection
        if st.session_state[prev_n_key] != n_terms:
            st.session_state[f"selected_terms_{plot_id}"] = all_terms[:n_terms]
            st.session_state[prev_n_key] = n_terms

        # Term selection with deselect all
        st.markdown("##### Select Terms to Include")
        col_sel, col_desel = st.columns([4, 1])
        with col_desel:
            if st.button("Deselect All", key=f"desel_{plot_id}"):
                st.session_state[f"selected_terms_{plot_id}"] = []
                st.rerun()

        # Initialize selected terms if not exists (use n_terms from slider)
        if f"selected_terms_{plot_id}" not in st.session_state:
            st.session_state[f"selected_terms_{plot_id}"] = all_terms[:n_terms]

        # Use key directly without default to avoid double-click issue
        selected_terms = st.multiselect(
            "Terms",
            options=all_terms,
            default=st.session_state.get(f"selected_terms_{plot_id}", all_terms[:n_terms]),
            key=f"terms_select_{plot_id}",
            label_visibility="collapsed"
        )
        # Update session state after selection
        st.session_state[f"selected_terms_{plot_id}"] = selected_terms

    if not selected_terms:
        st.warning("Please select at least one term")
        return

    # Filter dataframe
    plot_df = df[df['term'].isin(selected_terms)].copy()
    plot_df['-log10p'] = -np.log10(plot_df['adjusted_p_value'].replace(0, 1e-300))

    # Calculate gene count
    if 'genes' in plot_df.columns:
        plot_df['gene_count'] = plot_df['genes'].apply(lambda x: len(x) if isinstance(x, list) else 1)
    else:
        plot_df['gene_count'] = 1

    # Calculate gene ratio from overlap (e.g., "4/150" -> 4/150)
    if 'overlap' in plot_df.columns:
        def parse_ratio(x):
            try:
                parts = str(x).split('/')
                if len(parts) == 2:
                    return int(parts[0]) / int(parts[1])
            except:
                pass
            return 0.0

        plot_df['gene_ratio'] = plot_df['overlap'].apply(parse_ratio)
    else:
        plot_df['gene_ratio'] = plot_df['gene_count'] / 100  # fallback

    # Get combined_score if available
    if 'combined_score' not in plot_df.columns:
        plot_df['combined_score'] = plot_df['-log10p']  # fallback

    # Library color mapping for "Color by Library"
    library_colors = {}
    if 'library' in plot_df.columns:
        unique_libs = plot_df['library'].unique()
        lib_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                       '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
        for i, lib in enumerate(unique_libs):
            library_colors[lib] = lib_palette[i % len(lib_palette)]

    # Determine X and Color values based on selections
    x_col_map = {
        "Gene Count": "gene_count",
        "-log10(p-value)": "-log10p",
        "Combined Score": "combined_score",
        "Gene Ratio": "gene_ratio"
    }
    color_col_map = {
        "-log10(p-value)": "-log10p",
        "Combined Score": "combined_score"
    }

    x_col = x_col_map.get(x_axis, "-log10p")
    color_col = color_col_map.get(color_by, "-log10p") if color_by != "Library" else None

    # Render plot
    if plot_id == "bar":
        # Handle duplicate term names by appending library
        term_counts = plot_df['term'].value_counts()
        duplicate_terms = term_counts[term_counts > 1].index.tolist()

        def make_unique_term(row):
            term = row['term']
            if term in duplicate_terms and 'library' in row.index:
                # Shorten library name for display
                lib_short = row['library'].replace('_', ' ')
                if len(lib_short) > 20:
                    lib_short = lib_short[:17] + "..."
                return f"{term[:40]}... - {lib_short}" if len(term) > 40 else f"{term} - {lib_short}"
            return term[:50] + "..." if len(str(term)) > 50 else term

        plot_df['term_short'] = plot_df.apply(make_unique_term, axis=1)

        # Sort by color_by value descending (ascending=True for horizontal bar bottom-to-top)
        plot_df = plot_df.sort_values(x_col, ascending=True)

        # Calculate left margin based on max term length (approximate 7px per character)
        max_term_len = plot_df['term_short'].str.len().max()
        left_margin = max(200, min(400, int(max_term_len * 6)))

        if color_by == "Library" and 'library' in plot_df.columns:
            # Create traces per library for legend
            fig = go.Figure()
            for lib in plot_df['library'].unique():
                lib_df = plot_df[plot_df['library'] == lib]
                fig.add_trace(go.Bar(
                    y=lib_df['term_short'],
                    x=lib_df[x_col],
                    orientation='h',
                    name=lib,
                    marker=dict(color=library_colors.get(lib, '#377eb8')),
                    hovertemplate=f"<b>%{{y}}</b><br>{x_axis}: %{{x:.2f}}<br>Library: {lib}<extra></extra>"
                ))
            fig.update_layout(
                barmode='group',
                legend=dict(
                    title=dict(text="Library", font=dict(color='black', size=font_size)),
                    font=dict(color='black', size=font_size),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            )
        else:
            # Color by significance or combined score
            fig = go.Figure(go.Bar(
                y=plot_df['term_short'],
                x=plot_df[x_col],
                orientation='h',
                marker=dict(
                    color=plot_df[color_col],
                    colorscale=palette,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=color_by, font=dict(color='black', size=font_size)),
                        tickfont=dict(color='black', size=font_size)
                    )
                ),
                hovertemplate=f"<b>%{{y}}</b><br>{x_axis}: %{{x:.2f}}<br>{color_by}: %{{marker.color:.2f}}<extra></extra>"
            ))

        fig.update_layout(
            width=width, height=height,
            xaxis_title=x_axis,
            yaxis_title="",
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='black', size=font_size),
            xaxis=dict(
                title_font=dict(color='black', size=font_size),
                tickfont=dict(color='black', size=font_size),
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title_font=dict(color='black', size=font_size),
                tickfont=dict(color='black', size=font_size)
            ),
            legend=dict(font=dict(color='black', size=font_size)),
            margin=dict(l=left_margin, r=80, t=40, b=60)
        )
        st.plotly_chart(fig, use_container_width=False, key=f"pc_bar_{width}_{height}_{font_size}_{palette}_{color_by}_{x_axis}_{len(selected_terms)}")
        add_pdf_download_button(fig, "bar_plot.pdf", "dlg_pdf_bar", width=width, height=height)

    else:  # bubble
        # Handle duplicate term names by appending library
        term_counts = plot_df['term'].value_counts()
        duplicate_terms = term_counts[term_counts > 1].index.tolist()

        def make_unique_term(row):
            term = row['term']
            if term in duplicate_terms and 'library' in row.index:
                lib_short = row['library'].replace('_', ' ')
                if len(lib_short) > 20:
                    lib_short = lib_short[:17] + "..."
                return f"{term[:35]}... - {lib_short}" if len(term) > 35 else f"{term} - {lib_short}"
            return term[:45] + "..." if len(str(term)) > 45 else term

        plot_df['term_short'] = plot_df.apply(make_unique_term, axis=1)

        # Sort by color_by value for display order (most significant at top)
        plot_df = plot_df.sort_values(x_col, ascending=True)

        # Calculate left margin based on max term length
        max_term_len = plot_df['term_short'].str.len().max()
        left_margin = max(200, min(400, int(max_term_len * 6)))

        # Bubble size scaling - consistent sizing based on gene count range
        min_bubble = 10
        max_bubble = 50
        gene_min = plot_df['gene_count'].min()
        gene_max = plot_df['gene_count'].max()

        if gene_max > gene_min:
            plot_df['bubble_size'] = min_bubble + (plot_df['gene_count'] - gene_min) / (gene_max - gene_min) * (
                        max_bubble - min_bubble)
        else:
            plot_df['bubble_size'] = (min_bubble + max_bubble) / 2

        if color_by == "Library" and 'library' in plot_df.columns:
            # Create traces per library for legend
            fig = go.Figure()
            for lib in plot_df['library'].unique():
                lib_df = plot_df[plot_df['library'] == lib]
                fig.add_trace(go.Scatter(
                    x=lib_df[x_col],
                    y=lib_df['term_short'],
                    mode='markers',
                    name=lib,
                    marker=dict(
                        size=lib_df['bubble_size'],
                        color=library_colors.get(lib, '#377eb8'),
                        opacity=0.7,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate=f"<b>%{{y}}</b><br>{x_axis}: %{{x:.2f}}<br>Genes: %{{customdata}}<br>Library: {lib}<extra></extra>",
                    customdata=lib_df['gene_count']
                ))
            fig.update_layout(
                legend=dict(
                    title=dict(text="Library", font=dict(color='black', size=font_size)),
                    font=dict(color='black', size=font_size),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            )
        else:
            # Color by significance or combined score
            fig = go.Figure(go.Scatter(
                x=plot_df[x_col],
                y=plot_df['term_short'],
                mode='markers',
                marker=dict(
                    size=plot_df['bubble_size'],
                    color=plot_df[color_col],
                    colorscale=palette,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=color_by, font=dict(color='black', size=font_size)),
                        tickfont=dict(color='black', size=font_size)
                    ),
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                hovertemplate=f"<b>%{{y}}</b><br>{x_axis}: %{{x:.2f}}<br>Genes: %{{customdata}}<extra></extra>",
                customdata=plot_df['gene_count']
            ))

        # Add size legend annotation
        size_legend_text = f"Bubble size: Gene count ({gene_min}-{gene_max})"

        fig.update_layout(
            width=width, height=height,
            xaxis_title=x_axis,
            yaxis_title="",
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='black', size=font_size),
            xaxis=dict(
                title_font=dict(color='black', size=font_size),
                tickfont=dict(color='black', size=font_size),
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title_font=dict(color='black', size=font_size),
                tickfont=dict(color='black', size=font_size)
            ),
            legend=dict(font=dict(color='black', size=font_size)),
            margin=dict(l=left_margin, r=80, t=40, b=70),
            annotations=[dict(
                x=0.5, y=-0.08, xref="paper", yref="paper",
                text=size_legend_text, showarrow=False,
                font=dict(size=font_size - 1, color='black')
            )]
        )
        st.plotly_chart(fig, use_container_width=False, key=f"pc_bub_{width}_{height}_{font_size}_{palette}_{color_by}_{x_axis}_{len(selected_terms)}")
        add_pdf_download_button(fig, "bubble_chart.pdf", "dlg_pdf_bubble", width=width, height=height)

    st.markdown("---")

    # AI Interpretation as a BUTTON
    if st.button("🤖 Generate AI Interpretation", key=f"ai_interp_{plot_id}"):
        with st.spinner("Generating interpretation..."):
            interpretation = get_visualizer().get_plot_interpretation(plot_id, plot_df)
        st.info(interpretation)


def _render_simple_dialog(plot_id: str, df: pd.DataFrame):
    """Render dialog for other plot types with term/gene selection"""
    # Add marker for wide dialog CSS
    st.markdown("<span class='wide-dialog'></span>", unsafe_allow_html=True)

    visualizer = get_visualizer()

    # Get all available terms sorted by significance
    all_terms = df.nsmallest(100, 'adjusted_p_value')['term'].tolist()

    # Get all genes if available
    all_genes = set()
    if 'genes' in df.columns:
        for g in df['genes']:
            if isinstance(g, list):
                all_genes.update(g)
    all_genes = sorted(list(all_genes))

    # Parameters in SLIDING EXPANDER
    params = {}
    selected_terms = None
    selected_genes = None

    with st.expander("Parameters", expanded=True):
        if plot_id == "dendrogram":
            # Row 1: Number of terms, Method, Width, Height, Font Size
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                params["n_terms"] = st.slider("Number of terms", 5, min(50, len(all_terms)), 10, key="dlg_den_n")
            with col2:
                params["method"] = st.selectbox("Clustering method", ["ward", "complete", "average"], key="dlg_den_m")
            with col3:
                params["width"] = st.slider("Width (px)", 200, 1500, 800, step=50, key="dlg_den_w")
            with col4:
                params["height"] = st.slider("Height (px)", 200, 1500, 600, step=50, key="dlg_den_h")
            with col5:
                params["font_size"] = st.slider("Font size", 8, 16, 10, key="dlg_den_fs")

            # Track n_terms changes to auto-update selection
            n_terms = params["n_terms"]
            prev_n_key = "prev_n_terms_dendrogram"
            if prev_n_key not in st.session_state:
                st.session_state[prev_n_key] = n_terms

            # If n_terms slider changed, update the term selection
            if st.session_state[prev_n_key] != n_terms:
                st.session_state["selected_terms_dendrogram"] = all_terms[:n_terms]
                st.session_state[prev_n_key] = n_terms

            st.markdown("##### Select Terms to Include")
            col_sel, col_desel = st.columns([4, 1])
            with col_desel:
                if st.button("Deselect All", key="desel_dendrogram"):
                    st.session_state["selected_terms_dendrogram"] = []
                    st.rerun()

            if "selected_terms_dendrogram" not in st.session_state:
                st.session_state["selected_terms_dendrogram"] = all_terms[:n_terms]

            selected_terms = st.multiselect(
                "Terms", options=all_terms,
                default=st.session_state.get("selected_terms_dendrogram", all_terms[:n_terms]),
                key="terms_select_dendrogram", label_visibility="collapsed"
            )
            st.session_state["selected_terms_dendrogram"] = selected_terms

        elif plot_id == "similarity":
            # Row 1: Number of terms, Clusters, Width, Height, Font Size
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                params["n_terms"] = st.slider("Number of terms", 5, min(50, len(all_terms)), 10, key="dlg_sim_n")
            with col2:
                params["n_clusters"] = st.slider("Clusters", 3, 15, 8, key="dlg_sim_c")
            with col3:
                params["width"] = st.slider("Width (px)", 200, 1500, 1100, step=50, key="dlg_sim_w")
            with col4:
                params["height"] = st.slider("Height (px)", 200, 1500, 800, step=50, key="dlg_sim_h")
            with col5:
                params["font_size"] = st.slider("Font size", 8, 16, 12, key="dlg_sim_fs")

            # Track n_terms changes to auto-update selection
            n_terms = params["n_terms"]
            prev_n_key = "prev_n_terms_similarity"
            if prev_n_key not in st.session_state:
                st.session_state[prev_n_key] = n_terms

            # If n_terms slider changed, update the term selection
            if st.session_state[prev_n_key] != n_terms:
                st.session_state["selected_terms_similarity"] = all_terms[:n_terms]
                st.session_state[prev_n_key] = n_terms

            st.markdown("##### Select Terms to Include")
            col_sel, col_desel = st.columns([4, 1])
            with col_desel:
                if st.button("Deselect All", key="desel_similarity"):
                    st.session_state["selected_terms_similarity"] = []
                    st.rerun()

            if "selected_terms_similarity" not in st.session_state:
                st.session_state["selected_terms_similarity"] = all_terms[:n_terms]

            selected_terms = st.multiselect(
                "Terms", options=all_terms,
                default=st.session_state.get("selected_terms_similarity", all_terms[:n_terms]),
                key="terms_select_similarity", label_visibility="collapsed"
            )
            st.session_state["selected_terms_similarity"] = selected_terms

        elif plot_id == "upset":
            # Row 1: Number of terms, Width, Height, Font Size
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                params["n_terms"] = st.slider("Number of terms", 5, min(50, len(all_terms)), 10, key="dlg_ups_n")
            with col2:
                params["width"] = st.slider("Width (px)", 200, 1500, 900, step=50, key="dlg_ups_w")
            with col3:
                params["height"] = st.slider("Height (px)", 200, 1500, 700, step=50, key="dlg_ups_h")
            with col4:
                params["font_size"] = st.slider("Font size", 8, 16, 10, key="dlg_ups_fs")

            # Row 2: Colors
            col1, col2 = st.columns(2)
            with col1:
                params["dot_color"] = st.color_picker("Dot color", "#16213e", key="dlg_ups_dot")
            with col2:
                params["bar_color"] = st.color_picker("Bar color", "#1a1a2e", key="dlg_ups_bar")

            # Track n_terms changes to auto-update selection
            n_terms = params["n_terms"]
            prev_n_key = "prev_n_terms_upset"
            if prev_n_key not in st.session_state:
                st.session_state[prev_n_key] = n_terms

            # If n_terms slider changed, update the term selection
            if st.session_state[prev_n_key] != n_terms:
                st.session_state["selected_terms_upset"] = all_terms[:n_terms]
                st.session_state[prev_n_key] = n_terms

            st.markdown("##### Select Terms to Include")
            col_sel, col_desel = st.columns([4, 1])
            with col_desel:
                if st.button("Deselect All", key="desel_upset"):
                    st.session_state["selected_terms_upset"] = []
                    st.rerun()

            if "selected_terms_upset" not in st.session_state:
                st.session_state["selected_terms_upset"] = all_terms[:n_terms]

            selected_terms = st.multiselect(
                "Terms", options=all_terms,
                default=st.session_state.get("selected_terms_upset", all_terms[:n_terms]),
                key="terms_select_upset", label_visibility="collapsed"
            )
            st.session_state["selected_terms_upset"] = selected_terms

        elif plot_id == "cnetplot":
            # Row 1: Number of terms, Width, Height, Clusters, Font Size
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                params["n_terms"] = st.slider("Number of terms", 5, min(50, len(all_terms)), 10, key="dlg_cnet_n")
            with col2:
                params["width"] = st.slider("Width (px)", 200, 1500, 1000, step=50, key="dlg_cnet_w")
            with col3:
                params["height"] = st.slider("Height (px)", 200, 1500, 800, step=50, key="dlg_cnet_h")
            with col4:
                params["n_clusters"] = st.slider("Clusters", 2, 10, 5, key="dlg_cnet_c")
            with col5:
                params["font_size"] = st.slider("Font size", 8, 16, 10, key="dlg_cnet_fs")

            # Track n_terms changes to auto-update selection
            n_terms = params["n_terms"]
            prev_n_key = "prev_n_terms_cnetplot"
            if prev_n_key not in st.session_state:
                st.session_state[prev_n_key] = n_terms

            # If n_terms slider changed, update the term selection
            if st.session_state[prev_n_key] != n_terms:
                st.session_state["selected_terms_cnetplot"] = all_terms[:n_terms]
                st.session_state[prev_n_key] = n_terms

            st.markdown("##### Select Terms to Include")
            col_sel, col_desel = st.columns([4, 1])
            with col_desel:
                if st.button("Deselect All Terms", key="desel_cnet_terms"):
                    st.session_state["selected_terms_cnetplot"] = []
                    st.rerun()

            if "selected_terms_cnetplot" not in st.session_state:
                st.session_state["selected_terms_cnetplot"] = all_terms[:n_terms]

            selected_terms = st.multiselect(
                "Terms", options=all_terms,
                default=st.session_state.get("selected_terms_cnetplot", all_terms[:n_terms]),
                key="terms_select_cnetplot", label_visibility="collapsed"
            )
            st.session_state["selected_terms_cnetplot"] = selected_terms

            # Gene selection
            default_n_genes = min(50, len(all_genes))
            st.markdown("##### Select Genes to Include")
            col_sel, col_desel = st.columns([4, 1])
            with col_desel:
                if st.button("Deselect All Genes", key="desel_cnet_genes"):
                    st.session_state["selected_genes_cnetplot"] = []
                    st.rerun()

            if "selected_genes_cnetplot" not in st.session_state:
                st.session_state["selected_genes_cnetplot"] = all_genes[:default_n_genes]

            selected_genes = st.multiselect(
                "Genes", options=all_genes,
                default=st.session_state.get("selected_genes_cnetplot", all_genes[:default_n_genes]),
                key="genes_select_cnetplot", label_visibility="collapsed"
            )
            st.session_state["selected_genes_cnetplot"] = selected_genes

    # For plots with term selection, check if terms are selected
    if plot_id in ["dendrogram", "similarity", "upset", "cnetplot"]:
        if not selected_terms or len(selected_terms) < 2:
            st.warning("Please select at least 2 terms")
            return
        params["selected_terms"] = selected_terms
        if plot_id == "cnetplot" and selected_genes:
            params["selected_genes"] = selected_genes

    # Render the actual plot
    _render_full_plot(plot_id, df, params, visualizer)

    st.markdown("---")

    # AI Interpretation as a BUTTON
    if st.button("🤖 Generate AI Interpretation", key=f"ai_interp_{plot_id}"):
        with st.spinner("Generating interpretation..."):
            interpretation = get_visualizer().get_plot_interpretation(plot_id, df)
        st.info(interpretation)


def _render_full_plot(plot_id: str, df: pd.DataFrame, params: Dict, visualizer):
    """Render the full plot with selected terms and params"""

    # Compute a hash of current params to use as plotly_chart key (forces re-render on param change)
    _params_hash = hashlib.md5(str(sorted((k, str(v)[:50]) for k, v in params.items() if k != "selected_terms")).encode()).hexdigest()[:8]
    _n_terms = len(params.get("selected_terms", []))
    _chart_key = f"pc_{plot_id}_{_params_hash}_{_n_terms}"

    # Filter by selected terms if provided
    selected_terms = params.get("selected_terms")
    if selected_terms:
        plot_df = df[df['term'].isin(selected_terms)].copy()
    else:
        plot_df = df.copy()

    if plot_id == "dendrogram":
        if visualizer and len(plot_df) >= 3:
            method = params.get("method", "ward")
            width = params.get("width", 800)
            height = params.get("height", 600)
            font_size = params.get("font_size", 10)
            result = visualizer.render_dendrogram_with_clusters(plot_df, n_terms=len(plot_df), method=method,
                                                                font_size=font_size)
            if result and result[0]:
                fig = result[0]
                fig.update_layout(
                    width=width, height=height,
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(size=font_size, color='black'),
                    xaxis=dict(tickfont=dict(color='black', size=font_size), title_font=dict(color='black')),
                    yaxis=dict(tickfont=dict(color='black', size=font_size), title_font=dict(color='black')),
                    margin=dict(l=10, r=10, t=30, b=30)
                )
                st.plotly_chart(fig, use_container_width=False, key=_chart_key)
                add_pdf_download_button(fig, "dendrogram.pdf", "dlg_pdf_dend", width=width, height=height)

    elif plot_id == "upset":
        if visualizer and len(plot_df) >= 2:
            height = params.get("height", 700)
            width = params.get("width", 900)
            font_size = params.get("font_size", 10)
            bar_color = params.get("bar_color", "#1a1a2e")
            dot_color = params.get("dot_color", "#16213e")

            fig = visualizer.render_upset_plot(plot_df, n_terms=len(plot_df), bar_color=bar_color, dot_color=dot_color,
                                               font_size=font_size)
            if fig:
                # Add black line at y=0 for intersection size axis
                fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)
                fig.update_layout(
                    height=height, width=width,
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='black', size=font_size),
                    margin=dict(l=10, r=10, t=30, b=30)
                )
                # Update all subplots' axes
                fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
                fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
                st.plotly_chart(fig, use_container_width=False, key=_chart_key)
                add_pdf_download_button(fig, "upset_plot.pdf", "dlg_pdf_upset", width=width, height=height)

    elif plot_id == "similarity":
        if visualizer and len(plot_df) >= 3:
            clusters = params.get("n_clusters", 8)
            height = params.get("height", 800)
            width = params.get("width", 1100)
            font_size = params.get("font_size", 12)
            result = visualizer.render_similarity_clusters_with_data(plot_df, n_terms=len(plot_df), n_clusters=clusters,
                                                                     height=height, width=width,
                                                                     word_font_size=font_size)
            if result and result[0]:
                fig = result[0]
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='black', size=font_size),
                    margin=dict(l=10, r=10, t=30, b=30)
                )
                fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
                fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
                fig.update_coloraxes(colorbar_tickfont=dict(color='black'), colorbar_title_font=dict(color='black'))
                st.plotly_chart(fig, use_container_width=False, key=_chart_key)
                add_pdf_download_button(fig, "similarity.pdf", "dlg_pdf_sim", width=width, height=height)

    elif plot_id == "cnetplot":
        if visualizer and len(plot_df) >= 2:
            clusters = params.get("n_clusters", 5)
            height = params.get("height", 800)
            width = params.get("width", 1000)
            font_size = params.get("font_size", 10)
            selected_genes = params.get("selected_genes")
            max_genes = len(selected_genes) if selected_genes else 50

            result = visualizer.render_cnetplot_with_clusters(plot_df, n_terms=len(plot_df), n_clusters=clusters,
                                                              max_genes=max_genes, font_size=font_size)
            if result and result[0]:
                fig = result[0]
                fig.update_layout(
                    height=height, width=width,
                    paper_bgcolor='white', plot_bgcolor='white',
                    font=dict(color='black', size=font_size),
                    margin=dict(l=10, r=10, t=30, b=30)
                )
                fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
                fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
                st.plotly_chart(fig, use_container_width=False, key=_chart_key)
                add_pdf_download_button(fig, "cnetplot.pdf", "dlg_pdf_cnet", width=width, height=height)

def _create_full_figure(plot_id: str, df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create the FULL-SIZE figure for thumbnail conversion.
    Uses white/light theme for clean thumbnail appearance.
    """
    # Create lightweight Visualizer without gemini_model - plot rendering doesn't need AI.
    # Avoids unhashable session_state access when called from @st.cache_data context.
    visualizer = Visualizer()

    try:
        # ===== BAR PLOT =====
        if plot_id == "bar":
            plot_df = df.nsmallest(8, 'adjusted_p_value').copy()
            plot_df['-log10p'] = -np.log10(plot_df['adjusted_p_value'].replace(0, 1e-300))
            plot_df['term_short'] = plot_df['term'].apply(lambda x: x[:25] + "..." if len(str(x)) > 25 else x)

            fig = go.Figure(go.Bar(
                y=plot_df['term_short'],
                x=plot_df['-log10p'],
                orientation='h',
                marker=dict(color=plot_df['-log10p'], colorscale='Blues'),
            ))
            fig.update_layout(
                width=900,
                height=550,
                paper_bgcolor='white',
                plot_bgcolor='white',
                yaxis=dict(categoryorder='total ascending'),
                xaxis_title="-log10(p)",
                margin=dict(l=180, r=30, t=30, b=50),
                font=dict(size=14),
            )
            return fig

        # ===== BUBBLE CHART =====
        elif plot_id == "bubble":
            plot_df = df.nsmallest(8, 'adjusted_p_value').copy()
            plot_df['-log10p'] = -np.log10(plot_df['adjusted_p_value'].replace(0, 1e-300))
            plot_df['gene_count'] = plot_df['genes'].apply(
                lambda x: len(x) if isinstance(x, list) else 1
            ) if 'genes' in plot_df.columns else 5
            plot_df['term_short'] = plot_df['term'].apply(lambda x: x[:20] + "..." if len(str(x)) > 20 else x)

            fig = go.Figure(go.Scatter(
                x=plot_df['-log10p'],
                y=plot_df['term_short'],
                mode='markers',
                marker=dict(
                    size=plot_df['gene_count'],
                    sizemode='area',
                    sizeref=2.0 * plot_df['gene_count'].max() / (40 ** 2),
                    sizemin=6,
                    color=plot_df['-log10p'],
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.8,
                ),
            ))
            fig.update_layout(
                width=900,
                height=550,
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis_title="-log10(p)",
                margin=dict(l=180, r=30, t=30, b=50),
                font=dict(size=14),
            )
            return fig

        # ===== DENDROGRAM =====
        elif plot_id == "dendrogram":
            if visualizer:
                result = visualizer.render_dendrogram_with_clusters(df, n_terms=20, method="ward")
                if result and result[0]:
                    fig = result[0]
                    fig.update_layout(
                        width=900,
                        height=550,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        margin=dict(l=30, r=30, t=30, b=30),
                        font=dict(size=12)
                    )
                    return fig
            return None

        # ===== SIMILARITY CLUSTERS =====
        elif plot_id == "similarity":
            if visualizer:
                result = visualizer.render_similarity_clusters_with_data(df, n_terms=40, n_clusters=6, height=550)
                if result and result[0]:
                    fig = result[0]
                    fig.update_layout(
                        width=900,
                        height=550,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        margin=dict(l=30, r=30, t=30, b=30),
                        font=dict(size=12)
                    )
                    return fig
            return None

        # ===== CNETPLOT =====
        elif plot_id == "cnetplot":
            if visualizer:
                result = visualizer.render_cnetplot_with_clusters(df, n_terms=15, n_clusters=5, max_genes=40)
                if result and result[0]:
                    fig = result[0]
                    fig.update_layout(
                        width=900,
                        height=550,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        margin=dict(l=30, r=30, t=30, b=30),
                        font=dict(size=12)
                    )
                    return fig
            return None

        # ===== UPSET PLOT =====
        elif plot_id == "upset":
            if visualizer:
                fig = visualizer.render_upset_plot(df, n_terms=10)
                if fig:
                    fig.update_layout(
                        width=900,
                        height=550,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        margin=dict(l=30, r=30, t=30, b=30),
                        font=dict(size=12)
                    )
                    return fig
            return None

        # ===== NETWORK/ENRICHMENT MAP =====
    except Exception as e:
        logger.warning(f"Full figure creation failed for {plot_id}: {e}")

    return None

def _style_df(df):
    """Apply theme-aware colors to a dataframe (dark/light)."""
    is_dark = st.session_state.get("dark_mode", True)
    if is_dark:
        _bg, _fg, _hdr_bg, _border = "#1a1f2e", "#f1f5f9", "#0f1420", "#3a4358"
    else:
        _bg, _fg, _hdr_bg, _border = "#ffffff", "#1e293b", "#e2e8f0", "#cbd5e1"
    _line = "#64748b" if is_dark else "#cbd5e1"
    return df.style.set_properties(
        **{"background-color": _bg, "color": _fg, "border": f"1px solid {_line}"}
    ).set_table_styles([
        {"selector": "th", "props": [
            ("background-color", _hdr_bg), ("color", _fg), ("border", f"1px solid {_line}")
        ]},
        {"selector": "td", "props": [("border", f"1px solid {_line}")]},
    ])

def render_enrichment_results(enrichment_df: pd.DataFrame, envelope: Dict, msg_idx: int = None):
    """Render enrichment results with Enrichr-style layout"""
    if enrichment_df is None or enrichment_df.empty:
        return

    if isinstance(enrichment_df, list):
        enrichment_df = pd.DataFrame(enrichment_df)

    st.session_state["_enrichment_df"] = enrichment_df

    st.markdown("### 📊 Enrichment Analysis Results")

    # === METRICS ROW ===
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Terms", len(enrichment_df))
    with col2:
        n_sig = len(enrichment_df[enrichment_df["adjusted_p_value"] < 0.05])
        st.metric("Significant (p<0.05)", n_sig)
    with col3:
        st.metric("Libraries", enrichment_df["library"].nunique())
    with col4:
        if "genes" in enrichment_df.columns:
            all_genes = set()
            for g in enrichment_df["genes"]:
                if isinstance(g, list):
                    all_genes.update(g)
            st.metric("Unique Genes", len(all_genes))

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # === INJECT FANCY BUTTON CSS RIGHT HERE ===
    st.markdown("""
    <style>
    /* ============= FANCY COLUMN HEADERS ============= */
    .viz-column-header {
        padding: 0.85rem 1rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .viz-column-header-blue {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: 2px solid #60a5fa;
    }
    .viz-column-header-purple {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        border: 2px solid #a78bfa;
    }
    .viz-column-header-cyan {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        border: 2px solid #22d3ee;
    }

    /* ============= CENTER BUTTON CONTAINER ============= */
    /* Constrain button container to 300px and center it */
    div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] > div:has(> div.stButton) {
        display: flex !important;
        justify-content: center !important;
    }

    /* Force markdown containers in columns to be full width */
    div[data-testid="column"] .stMarkdown {
        width: 100% !important;
    }

    div[data-testid="column"] .stMarkdown > div {
        width: 100% !important;
    }

    /* ============= FORCE FANCY BUTTONS IN COLUMNS ============= */
    /* Style the buttons themselves */
    section[data-testid="stVerticalBlock"] div[data-testid="column"] button,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="column"] button,
    div[data-testid="column"] .stButton > button,
    div[data-testid="column"] button[kind="secondary"],
    div[data-testid="column"] button[kind="primary"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.9rem 1.2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        cursor: pointer !important;
    }

    section[data-testid="stVerticalBlock"] div[data-testid="column"] button:hover,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="column"] button:hover,
    .stButton button:hover,
    div[data-testid="column"] .stButton > button:hover,
    div[data-testid="column"] button[kind="secondary"]:hover,
    div[data-testid="column"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        border-color: #10b981 !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
        transform: translateY(-3px) !important;
    }

    /* Fancy gradient text for buttons */
    div[data-testid="column"] button p,
    div[data-testid="column"] .stButton button p,
    .stButton button p {
        background: linear-gradient(135deg, #93c5fd 0%, #c4b5fd 50%, #67e8f9 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* On hover, make text green gradient */
    div[data-testid="column"] button:hover p,
    div[data-testid="column"] .stButton button:hover p,
    .stButton button:hover p {
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Unique suffix for keys in this message
    _ks = f"_{msg_idx}" if msg_idx is not None else ""

    # === PLOT GRID - Square cards, single clickable button ===
    st.markdown("#### 🎨 Visualizations")

    col_terms, col_clusters, col_other = st.columns(3)

    with col_terms:
        st.markdown('<div class="viz-column-header viz-column-header-blue">📊 Term Analysis</div>',
                    unsafe_allow_html=True)
        for plot in PLOT_CONFIGS["term_plots"]:
            _render_plot_card(plot, enrichment_df, "term", _ks)

    with col_clusters:
        st.markdown('<div class="viz-column-header viz-column-header-purple">🧬 Clustering</div>',
                    unsafe_allow_html=True)
        for plot in PLOT_CONFIGS["cluster_plots"]:
            _render_plot_card(plot, enrichment_df, "cluster", _ks)

    with col_other:
        st.markdown('<div class="viz-column-header viz-column-header-cyan">🔬 Networks</div>', unsafe_allow_html=True)
        for plot in PLOT_CONFIGS["other_plots"]:
            _render_plot_card(plot, enrichment_df, "other", _ks)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # === SINGLE EXPAND FOR TABLE + FULL INTERPRETATIONS ===
    with st.expander("📋 View Full Details (Table & Interpretations)", expanded=False):
        _render_full_details(enrichment_df, _ks)


@st.cache_data(ttl=600, show_spinner=False)
def _get_thumbnail_image(plot_id: str, df_hash: str, df_json: str, _version: str = "v39") -> Optional[str]:
    """
    Generate and CACHE the thumbnail image as base64.
    """
    import base64

    df = pd.read_json(df_json)
    fig = _create_full_figure(plot_id, df)

    if fig is None:
        return None

    try:
        # Export - scale=2 for crisp display
        img_bytes = fig.to_image(format="png", scale=2)
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        logger.warning(f"Thumbnail image generation failed for {plot_id}: {e}")
        return None


def _render_plot_card(plot_config: Dict, df: pd.DataFrame, theme: str, key_suffix: str = ""):
    """Render a plot card with button + cached image thumbnail"""
    plot_id = plot_config["id"]
    icon = plot_config["icon"]
    title = plot_config["title"]

    # Button - use_container_width with CSS controlling the container width
    if st.button(
            f"{icon} {title}",
            key=f"btn_{plot_id}_{theme}{key_suffix}",
            use_container_width=True,
            help=plot_config['desc']
    ):
        dialog_func = DIALOG_FUNCTIONS.get(plot_id)
        if dialog_func:
            dialog_func()

    # Create a hash of the dataframe for caching (handle list columns)
    df_subset = df.head(50).copy()

    # Convert list columns to strings for hashing
    for col in df_subset.columns:
        if df_subset[col].apply(lambda x: isinstance(x, list)).any():
            df_subset[col] = df_subset[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Create hash from the string representation
    df_hash = hashlib.md5(df_subset.to_json().encode()).hexdigest()

    # Get CACHED thumbnail image
    img_b64 = _get_thumbnail_image(plot_id, df_hash, df.head(50).to_json())

    if img_b64:
        # Smaller container, image fills 80%
        st.markdown(f"""
        <div style="width: 90%; min-height: 150px; background: #FFFFFF; border-radius: 12px; padding: 8px; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
            <img src="data:image/png;base64,{img_b64}" style="width: 80%; height: auto; border-radius: 8px;" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"{icon} Loading...")

    # Spacing
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


def _render_full_details(df: pd.DataFrame, key_suffix: str = ""):
    """Render full table and ALL interpretations in the expander"""

    # === TABLE ===
    st.markdown("##### Results Table")

    # Build display columns - include z_score and combined_score if available
    display_df = df[['term', 'library', 'adjusted_p_value', 'overlap']].copy()
    display_df['p-value'] = display_df['adjusted_p_value'].apply(lambda x: f"{x:.2e}")
    display_df['-log10(p)'] = -np.log10(display_df['adjusted_p_value'].replace(0, 1e-300))

    # Add z_score if available
    if 'z_score' in df.columns:
        display_df['z_score'] = df['z_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    # Add combined_score if available
    if 'combined_score' in df.columns:
        display_df['combined_score'] = df['combined_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    if 'genes' in df.columns:
        display_df['genes'] = df['genes'].apply(
            lambda x: ", ".join(x[:5]) + f"... (+{len(x) - 5})" if isinstance(x, list) and len(x) > 5
            else ", ".join(x) if isinstance(x, list) else ""
        )

    # Build column order
    cols = ['term', 'library', 'p-value', '-log10(p)']
    if 'z_score' in display_df.columns:
        cols.append('z_score')
    if 'combined_score' in display_df.columns:
        cols.append('combined_score')
    cols.append('overlap')
    if 'genes' in display_df.columns:
        cols.append('genes')

    st.dataframe(_style_df(display_df[cols]), use_container_width=True, hide_index=True, height=400)
    csv = display_df.to_csv(index=False)
    st.download_button("Download as CSV", csv, "enrichment_results.csv", "text/csv",
                       key=f"dl_enrich_{hash(str(display_df.shape)) % 100000}{key_suffix}")

    st.markdown("---")

    # === DETAILED INTERPRETATIONS (ONLY IN EXPANDER) ===
    st.markdown("##### 🤖 Detailed AI Interpretations")

    visualizer = get_visualizer()
    if visualizer and visualizer.gemini_model:
        # Overall interpretation
        with st.expander("📊 Overall Biological Interpretation", expanded=True):
            top_terms = df.nsmallest(15, 'adjusted_p_value')['term'].tolist()
            prompt = """Provide a detailed biological interpretation of these enrichment results (3-4 paragraphs):
1. Main biological themes and their significance
2. Connections between pathways/terms
3. Potential disease relevance
4. Suggested follow-up analyses"""

            data = {"top_terms": top_terms, "libraries": df['library'].unique().tolist()}
            interp = get_gemini_interpretation(visualizer.gemini_model, prompt, json.dumps(data))
            st.markdown(interp)


# ===============================================================================
# MESSAGE DISPLAY
# ===============================================================================

def toggle_pin_message(idx):
    """Callback to toggle pin status of a message"""
    if "pinned_messages" not in st.session_state:
        st.session_state.pinned_messages = []

    if idx in st.session_state.pinned_messages:
        st.session_state.pinned_messages.remove(idx)
    else:
        st.session_state.pinned_messages.append(idx)


def display_message_card(message: Dict, message_idx: int = None, previous_user_message: str = ""):
    """Display message with results - AI Summary first, then data, then detailed text in expander"""
    role = message.get("role", "")
    content = message.get("content", "")
    envelope = message.get("envelope", {})

    # Get avatar based on role and context
    if role == "user":
        avatar = _get_user_avatar()
    else:
        # Determine assistant avatar type based on context
        avatar_type = _determine_assistant_avatar(envelope, previous_user_message)
        avatar = _get_assistant_avatar(avatar_type)

    with st.chat_message(role, avatar=avatar):
        # Add anchor ID for scroll-to functionality
        if message_idx is not None:
            st.markdown(f'<div id="msg-{message_idx}"></div>', unsafe_allow_html=True)

        if role == "user":
            st.markdown(content)
            return

        # === PIN BUTTON FOR ASSISTANT MESSAGES ===
        if message_idx is not None:
            is_pinned = message_idx in st.session_state.get("pinned_messages", [])

            # Pin button in corner - use on_click callback
            col1, col2 = st.columns([0.92, 0.08])
            with col2:
                pin_label = "📌" if is_pinned else "📍"
                st.button(
                    pin_label,
                    key=f"pin_btn_{message_idx}",
                    help="Unpin" if is_pinned else "Pin",
                    on_click=toggle_pin_message,
                    args=(message_idx,)
                )

        # === 1. AI SUMMARY BOX (Primary visible content) ===
        if envelope and envelope.get("tools_used"):
            # Split at "Detailed Analysis" header
            summary = content
            detailed = ""

            for marker in ["## Detailed Analysis", "### Detailed Analysis", "**Detailed Analysis"]:
                pos = content.find(marker)
                if pos > 0:
                    summary = content[:pos].strip()
                    detailed = content[pos:].strip()
                    break

            # === 1. SUMMARY BOX ===
            summary_safe = html.escape(summary).replace('\n\n', '</p><p>').replace('\n', '<br>')
            st.markdown(f"""
                <div class="summary-box">
                    <div class="summary-title">🤖 AI Analysis Summary</div>
                    <div class="summary-text"><p>{summary_safe}</p></div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            # === 2. DATA VISUALIZATIONS & TABLES ===

            # Gene Info card
            if envelope.get("gene_info"):
                st.markdown("### 🧬 Gene Information")
                render_gene_card(envelope["gene_info"])

            # Literature results
            papers = envelope.get("full_literature_results") or envelope.get("literature")
            if papers:
                st.markdown("### 📚 Literature Search Results")
                lit_query = envelope.get("literature_query", "")
                papers = _add_relevance_scores(papers, lit_query)
                card_key = f"msg_{message_idx}" if message_idx is not None else f"lit_{id(papers)}"
                render_literature_card(papers, lit_query, card_key=card_key)

            # Enrichment results
            enrichment_df = envelope.get("enrichment_df")
            if enrichment_df is not None:
                if isinstance(enrichment_df, list):
                    enrichment_df = pd.DataFrame(enrichment_df)
                if not enrichment_df.empty:
                    render_enrichment_results(enrichment_df, envelope, msg_idx=message_idx)

            # Database results
            db_results = envelope.get("full_db_results") or envelope.get("db")
            if db_results:
                st.markdown("### 🗄️ Database Query Results")
                render_database_card(db_results, msg_idx=message_idx)

            # === 3. DETAILED ANALYSIS (in expander) ===
            if detailed:
                with st.expander("🔎 Detailed Analysis", expanded=False):
                    st.markdown(detailed)

            # Follow-ups
            render_followup_suggestions(envelope, msg_idx=message_idx)

        else:
            # No tools used - just show the response directly
            st.markdown(content)

        # === EXECUTION DETAILS - ENHANCED REACT DISPLAY ===
        with st.expander("⚙️ Execution Details", expanded=False):
            tools_used = envelope.get("tools_used", []) if envelope else []
            exec_time = envelope.get("execution_time", 0) if envelope else 0
            trace = envelope.get("trace", []) if envelope else []
            reasoning_steps = envelope.get("reasoning_steps", []) if envelope else []
            confidence = envelope.get("confidence_score", 0) if envelope else 0



            def summarize_observation(obs_text: str, tool_name: str) -> tuple:
                """Generate a clear assessment of the tool result. Returns (summary, status)"""
                obs_lower = obs_text.lower()

                # Error cases
                if "error" in obs_lower or "failed" in obs_lower or "exception" in obs_lower:
                    return "Execution failed - the tool encountered an error and may need a different approach", "error"
                elif "no results" in obs_lower or "0 results" in obs_lower or "not found" in obs_lower:
                    return "No results found - the query returned empty, may need different search terms", "warning"

                # Tool-specific summaries
                elif tool_name == "run_enrichment_analysis":
                    import re
                    terms_match = re.search(r'(\d+)\s*significant terms', obs_text)
                    genes_match = re.search(r'(\d+)\s*genes?\s*analyzed', obs_text, re.IGNORECASE)
                    if terms_match:
                        n_terms = int(terms_match.group(1))
                        genes_info = f" from {genes_match.group(1)} genes" if genes_match else ""
                        if n_terms > 50:
                            return f"Success: Found {n_terms} significant terms{genes_info} - comprehensive results ready for analysis", "success"
                        elif n_terms > 0:
                            return f"Success: Found {n_terms} significant terms{genes_info} - results ready for analysis", "success"
                        else:
                            return "No significant enrichment found at the current p-value threshold", "warning"
                    return "Enrichment analysis completed - check results tab for details", "success"

                elif tool_name == "search_literature":
                    import re
                    papers_match = re.search(r'(\d+)\s*papers?', obs_text)
                    if papers_match:
                        n_papers = int(papers_match.group(1))
                        if n_papers > 10:
                            return f"Success: Found {n_papers} relevant papers - ready to summarize findings", "success"
                        elif n_papers > 0:
                            return f"Success: Found {n_papers} papers - limited but usable for analysis", "success"
                        return "No papers found matching the search criteria", "warning"
                    return "Literature search completed - check results for papers", "success"

                elif tool_name == "get_gene_info":
                    if "symbol" in obs_lower or "name" in obs_lower or "summary" in obs_lower:
                        return "Success: Gene information retrieved - ready to present findings", "success"
                    return "Gene lookup completed - some information may be limited", "warning"

                elif tool_name == "db_retrieve":
                    import re
                    terms_match = re.search(r'(\d+)\s*terms?', obs_text)
                    if terms_match:
                        n_terms = int(terms_match.group(1))
                        return f"Success: Retrieved {n_terms} database entries", "success"
                    if "results" in obs_lower:
                        return "Success: Database query returned results", "success"
                    return "Database query completed - check results for details", "info"

                # Default - be more descriptive
                if len(obs_text) > 200:
                    return "Tool executed successfully - results obtained and ready for synthesis", "success"
                elif len(obs_text) > 50:
                    return "Tool completed - results received", "success"
                return "Tool completed with brief response - may need additional queries for more detail", "info"

            def format_param_value(key: str, value: str) -> str:
                """Format parameter value - truncate gene lists, show counts"""
                if key.lower() == "genes":
                    # Count genes and show abbreviated
                    genes = [g.strip() for g in value.split(",") if g.strip()]
                    n_genes = len(genes)
                    if n_genes > 3:
                        return f"{n_genes} genes ({genes[0]}, {genes[1]}, ...)"
                    return f"{n_genes} genes ({value})"
                elif len(value) > 80:
                    return value[:77] + "..."
                return value

            def get_default_params(tool_name: str) -> dict:
                """Get default parameters for each tool"""
                defaults = {
                    "run_enrichment_analysis": {
                        "libraries": "GO_Biological_Process_2023, KEGG_2021_Human, Reactome_2022, WikiPathways_2021",
                        "p_value_threshold": "0.05",
                        "top_n": "All significant"
                    },
                    "search_literature": {
                        "max_results": "20",
                        "min_year": "None",
                        "max_year": "None",
                        "sort_by": "relevance"
                    },
                    "get_gene_info": {
                        "organism": "9606 (Human)"
                    },
                    "db_retrieve": {
                        "task": "auto",
                        "organism": "9606 (Human)",
                        "limit": "20",
                        "include_genes": "False"
                    }
                }
                return defaults.get(tool_name, {})

            # === SUMMARY METRICS ===
            st.markdown("#### 📊 Execution Summary")
            col1, col2, col3= st.columns(3)
            with col1:
                st.metric("⏱️ Time", f"{exec_time:.2f}s" if exec_time else "N/A")
            with col2:
                st.metric("🔧 Tools", len(tools_used))
            with col3:
                hops = sum(1 for s in reasoning_steps if s.get("type") == "thought_action")
                st.metric("🔄 Hops", hops)

            # Tools pipeline
            if tools_used:
                tool_badges = " → ".join([f"`{t}`" for t in tools_used])
                st.markdown(f"**Pipeline:** {tool_badges}")

            st.markdown("---")

            # === DETAILED REACT TRACE ===
            if reasoning_steps and len(reasoning_steps) > 0:
                st.markdown("#### 🧠 ReAct Reasoning Trace")
                st.caption("Each step shows the model's Thought → Action → Observation cycle")

                tool_step_num = 0
                for i, step in enumerate(reasoning_steps):
                    if not isinstance(step, dict):
                        continue

                    step_type = step.get("type", "")

                    # --- QUERY STEP ---
                    if step_type == "query":
                        content = step.get("content", "")
                        st.info(f"**Query:** {content[:500]}")
                        continue

                    # --- MODEL REASONING (between tool calls) ---
                    if step_type == "reasoning":
                        content = step.get("content", "")
                        st.markdown("**🔍 Model Analysis (post-observation):**")
                        st.info(content[:400] + ("..." if len(content) > 400 else ""))
                        continue

                    # --- FINAL ANSWER ---
                    if step_type == "final_answer":
                        st.success(
                            "**Final Answer Generated** - Model synthesized all observations into the response above")
                        continue

                    # --- THOUGHT-ACTION-OBSERVATION (tool calls) ---
                    if step_type == "thought_action":
                        tool_step_num += 1
                        tool_name = step.get("tool_name", "unknown")

                        thought_raw = step.get("thought", "")
                        args_raw = step.get("args", "")
                        observation_raw = step.get("observation", "")

                        # Step header
                        st.markdown(f"##### Step {tool_step_num}: `{tool_name}`")

                        # THOUGHT
                        st.markdown("**Thought:**")
                        if thought_raw:
                            st.text(thought_raw)
                        else:
                            st.caption("No explicit reasoning recorded")

                        # ACTION + PARAMETERS in columns
                        action_col, params_col = st.columns([1, 2])

                        with action_col:
                            st.markdown("**Action:**")
                            st.code(tool_name, language=None)

                        with params_col:
                            st.markdown("**Parameters:**")

                            # Parse provided args
                            provided_params = {}
                            if args_raw and "=" in args_raw:
                                for part in args_raw.split(", "):
                                    if "=" in part:
                                        k, v = part.split("=", 1)
                                        provided_params[k.strip()] = v.strip()

                            # Get defaults and merge
                            defaults = get_default_params(tool_name)
                            all_params = {**defaults, **provided_params}

                            for k, v in all_params.items():
                                formatted_v = format_param_value(k, str(v))
                                if k in provided_params:
                                    st.markdown(f"  - **{k}**: `{formatted_v}`")
                                else:
                                    st.markdown(f"  - {k}: {formatted_v} _(default)_")

                        # OBSERVATION
                        if observation_raw:
                            summary, status = summarize_observation(observation_raw, tool_name)

                            st.markdown("**Observation:**")
                            st.info(summary)

                            with st.expander("See full tool response", expanded=False):
                                st.text(observation_raw)

                        st.markdown("---")

            # === FALLBACK: Simple trace ===
            elif trace and len(trace) > 0:
                st.markdown("#### 📋 Execution Trace")
                for i, t in enumerate(trace, 1):
                    st.text(f"{i}. {t}")

            elif tools_used and len(tools_used) > 0:
                # Last resort - show tool names only
                st.markdown("#### 📋 Tools Called")
                for i, tool in enumerate(tools_used, 1):
                    st.code(f"Step {i}: {tool}()", language="python")

            else:
                # Instant answer - no tools
                st.markdown("**Mode:** Instant Answer (no tools used)")
                st.caption("Response generated from model knowledge")


def generate_response_summary(content: str, envelope: Dict) -> str:
    """Generate a comprehensive AI summary from the full response and envelope data"""
    gemini = get_gemini_model()
    if not gemini:
        # Fallback: extract more content
        lines = content.split('\n')
        summary_lines = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                summary_lines.append(line.strip())
                if len(' '.join(summary_lines)) > 2500:
                    break
        return ' '.join(summary_lines)[:3000] + "..." if len(' '.join(summary_lines)) > 3000 else ' '.join(
            summary_lines)

    try:
        # Build rich context from envelope
        tools_used = envelope.get("tools_used", [])

        context_parts = []
        context_parts.append(f"Tools used: {', '.join(tools_used)}")

        # Add detailed gene data
        if envelope.get("gene_info"):
            gene = envelope["gene_info"] if isinstance(envelope["gene_info"], list) else [envelope["gene_info"]]
            gene_context = f"Gene: {gene.get('symbol', 'Unknown')}"
            if gene.get('name'):
                gene_context += f" ({gene.get('name')})"
            if gene.get('alias'):
                aliases = gene.get('alias') if isinstance(gene.get('alias'), list) else [gene.get('alias')]
                gene_context += f" | Aliases: {', '.join(aliases[:5])}"
            context_parts.append(gene_context)

            # Include GO terms in context
            go_terms = gene.get('go_terms', {})
            if go_terms.get('BP'):
                bp_terms = [t.get('term', '') for t in go_terms['BP'][:5]]
                context_parts.append(f"GO Biological Process: {', '.join(bp_terms)}")

        if envelope.get("enrichment_df") is not None:
            df = envelope["enrichment_df"]
            if isinstance(df, pd.DataFrame) and not df.empty:
                n_terms = len(df)
                top_terms = df.nsmallest(5, 'adjusted_p_value')[
                    'term'].tolist() if 'adjusted_p_value' in df.columns else df['term'].head(5).tolist()
                context_parts.append(f"Enrichment: {n_terms} terms including {', '.join(top_terms)}")

        papers = envelope.get("full_literature_results") or envelope.get("literature", [])
        if papers:
            top_papers = [p.get('title', '')[:50] for p in papers[:3]]
            context_parts.append(f"Literature: {len(papers)} papers including '{top_papers[0]}'...")

        db_results = envelope.get("full_db_results") or envelope.get("db")
        if db_results:
            n_results = db_results.get("statistics", {}).get("total_results", 0)
            context_parts.append(f"Database: {n_results} pathway/term results")

        prompt = f"""Write a comprehensive summary (2-3 paragraphs, 8-12 sentences total) of this biology research analysis.

CONTEXT:
{chr(10).join(context_parts)}

FULL ANALYSIS:
{content[:20000]}

Your summary should:

PARAGRAPH 1: State what was analyzed and the main tools/approaches used. What was the biological question?

PARAGRAPH 2: Summarize the KEY biological findings:
- Gene function and role (if gene_info was used)
- Significant pathways or GO terms (with specific examples)
- Literature evidence (mention key papers if relevant)
- Any database hits

PARAGRAPH 3: Biological interpretation and implications:
- What do these findings tell us about the biology?
- Disease relevance or clinical implications
- Connections between different data sources

Write as flowing prose. NO bullet points, NO headers. Be specific - mention actual pathway names, GO terms, p-values where relevant. This is the MAIN content users will see."""

        summary_max_tokens = CONFIG.get("gemini", {}).get("summary_max_tokens", 8192)
        response = gemini.generate_content(prompt, generation_config={"max_output_tokens": summary_max_tokens})
        return response.text if hasattr(response, 'text') else content[:800]
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        # Fallback - extract more content
        lines = content.split('\n')
        summary_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')][:15]
        return ' '.join(summary_lines)[:5000]


def render_gene_card(gene_info):
    """Render comprehensive gene information card"""
    if isinstance(gene_info, list) and gene_info:
        for gi in gene_info:
            if isinstance(gi, dict):
                _render_single_gene_card(gi)
        return
    if isinstance(gene_info, dict):
        _render_single_gene_card(gene_info)


def _render_single_gene_card(gene_info):
    """Render a single gene information card"""
    if not isinstance(gene_info, dict):
        return

    with st.expander("🧬 Gene Information", expanded=True):
        symbol = gene_info.get('symbol') or gene_info.get('gene', 'Unknown')
        name = gene_info.get('name', '')
        summary = gene_info.get('summary', '')

        # Header with symbol and name
        st.markdown(f"### {symbol}")
        if name:
            st.markdown(f"**{name}**")

        # Aliases/Other names - handle both 'alias' and 'aliases' keys
        aliases = gene_info.get('alias') or gene_info.get('aliases', [])
        if aliases:
            if isinstance(aliases, str):
                aliases = [aliases]
            st.markdown(f"**Also known as:** {', '.join(aliases[:10])}")

        # Full summary (not truncated)
        if summary:
            st.markdown("---")
            st.markdown("**Summary:**")
            st.write(summary)

        # Key metrics row
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Type", gene_info.get('type_of_gene', 'protein-coding'))
        with col2:
            st.metric("KEGG Pathways", len(gene_info.get('kegg_pathways', [])))
        with col3:
            st.metric("Reactome Pathways", len(gene_info.get('reactome_pathways', [])))

        # GO Terms section - handle both key formats
        go_data = gene_info.get("go") or gene_info.get("go_terms", {})
        if go_data:
            st.markdown("---")
            st.markdown("**Gene Ontology Terms:**")

            # Map possible key names (handles both formats)
            go_mappings = [
                (["BP", "biological_process"], "Biological Process"),
                (["MF", "molecular_function"], "Molecular Function"),
                (["CC", "cellular_component"], "Cellular Component")
            ]

            go_tabs = st.tabs(["Biological Process", "Molecular Function", "Cellular Component"])

            for tab_idx, (keys, label) in enumerate(go_mappings):
                with go_tabs[tab_idx]:
                    # Find terms using either key format
                    terms = None
                    for key in keys:
                        if key in go_data:
                            terms = go_data[key]
                            break

                    if terms and isinstance(terms, list) and terms:
                        go_table_data = []
                        for t in terms[:20]:
                            if isinstance(t, dict):
                                go_table_data.append({
                                    "GO ID": t.get('id', ''),
                                    "Term": t.get('term', t.get('name', '')),
                                    "Evidence": t.get('evidence', '')
                                })
                            else:
                                go_table_data.append({"Term": str(t)})
                        if go_table_data:
                            st.dataframe(_style_df(pd.DataFrame(go_table_data)), use_container_width=True, hide_index=True)
                    else:
                        st.caption(f"No {label} terms found")

        # Pathways section
        kegg = gene_info.get('kegg_pathways', [])
        reactome = gene_info.get('reactome_pathways', [])

        if kegg or reactome:
            st.markdown("---")
            st.markdown("**Pathways:**")

            pathway_tabs = st.tabs(["KEGG", "Reactome"])

            with pathway_tabs[0]:
                if kegg:
                    kegg_data = []
                    for p in kegg[:20]:
                        if isinstance(p, dict):
                            kegg_data.append({"Pathway ID": p.get('id', ''), "Pathway Name": p.get('name', '')})
                        else:
                            kegg_data.append({"Pathway": str(p)})
                    if kegg_data:
                        st.dataframe(_style_df(pd.DataFrame(kegg_data)), use_container_width=True, hide_index=True)
                else:
                    st.caption("No KEGG pathways found")

            with pathway_tabs[1]:
                if reactome:
                    reactome_data = []
                    for p in reactome[:20]:
                        if isinstance(p, dict):
                            reactome_data.append({"Pathway ID": p.get('id', ''), "Pathway Name": p.get('name', '')})
                        else:
                            reactome_data.append({"Pathway": str(p)})
                    if reactome_data:
                        st.dataframe(_style_df(pd.DataFrame(reactome_data)), use_container_width=True, hide_index=True)
                else:
                    st.caption("No Reactome pathways found")


def _add_relevance_scores(papers: List[Dict], query: str) -> List[Dict]:
    """Normalize relevance field names for display.

    Scoring is done by _score_paper_relevance() in reasoning_engine.py
    using the same model as the reasoning agent. This function maps
    'relevance_rating' → 'specificity' for the display layer.
    """
    if not papers:
        return papers

    for p in papers:
        if not p.get("specificity"):
            # Map from reasoning_engine's field name
            rating = p.get("relevance_rating", "")
            if rating:
                p["specificity"] = rating
            else:
                p["specificity"] = "Medium"
            if not p.get("relevance_details"):
                p["relevance_details"] = ""

    return papers

def render_literature_card(papers: List[Dict], query: str = "", card_key: str = ""):
    """Render literature results with table with relevance, and expandable details"""
    if not papers:
        return

    # Generate unique key for this card
    unique_key = card_key or f"lit_{hash(query) % 100000}_{len(papers)}"

    # === STATS ROW ===
    total = len(papers)
    open_access = sum(1 for p in papers if p.get('is_open_access', False))
    with_abstract = sum(1 for p in papers if p.get('abstract'))
    avg_citations = sum(p.get('citations', 0) for p in papers) / max(total, 1)
    # Count high specificity (check both fields for backwards compatibility)
    high_spec = sum(1 for p in papers if p.get('specificity', p.get('relevance_rating', '')).lower() == 'high')

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📄 Papers", total)
    col2.metric("⭐ High Specificity", high_spec)
    col3.metric("🔓 Open Access", open_access)
    col4.metric("🔎 With Abstract", with_abstract)
    col5.metric("📊 Avg Citations", f"{avg_citations:.1f}")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # === TABLE WITH RELEVANCE COLUMNS ===
    def strip_html(text):
        if not text:
            return ""
        import re
        return re.sub(r'<[^>]+>', '', str(text))

    table_data = []
    for p in papers:
        authors = p.get('authors', [])
        if isinstance(authors, list):
            author_str = ", ".join(authors[:2])
            if len(authors) > 2:
                author_str += " et al."
        else:
            author_str = str(authors)[:50] if authors else "Unknown"

        # Use specificity, fallback to relevance_rating for backwards compatibility
        rating = p.get("specificity", p.get("relevance_rating", ""))
        details = strip_html(p.get("relevance_details", ""))

        # Create link
        pmid = p.get('pmid', '')
        doi = p.get('doi', '')
        url = p.get('url', '')
        if not url:
            if pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            elif doi:
                url = f"https://doi.org/{doi}"

        table_data.append({
            "Title": strip_html(p.get("title", "Untitled"))[:80],
            "Authors": author_str,
            "Year": p.get("year", "N/A"),
            "Citations": p.get("citations", 0),
            "Specificity": rating,
            "Why Relevant": details[:150] if details else "",
            "Link": url
        })

    if table_data:
        table_df = pd.DataFrame(table_data)
        st.dataframe(
            _style_df(table_df),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Title", width="medium"),
                "Authors": st.column_config.TextColumn("Authors", width="small"),
                "Year": st.column_config.NumberColumn("Year", width="small"),
                "Citations": st.column_config.NumberColumn("Citations", width="small"),
                "Specificity": st.column_config.TextColumn("Specificity", width="small"),
                "Why Relevant": st.column_config.TextColumn("Why Relevant", width="large"),
                "Link": st.column_config.LinkColumn("Link", width="small", display_text="View")
            }
        )

    # === PAPER DETAILS EXPANDER ===
    with st.expander("📄 Paper Details & Abstracts", expanded=False):
        for i, p in enumerate(papers, 1):
            title = strip_html(p.get('title', 'Untitled'))
            url = p.get("url", "")
            pmid = p.get('pmid', '')
            doi = p.get('doi', '')

            if not url:
                if pmid:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                elif doi:
                    url = f"https://doi.org/{doi}"

            if url:
                st.markdown(f"**{i}. [{title}]({url})**")
            else:
                st.markdown(f"**{i}. {title}**")

            authors = p.get("authors", "")
            if isinstance(authors, list):
                author_str = authors[0] + " et al." if len(authors) > 1 else (authors[0] if authors else "")
            else:
                author_str = str(authors)[:40] if authors else ""

            year = p.get("year", "")
            journal = p.get("journal", "")[:30] if p.get("journal") else ""
            citations = p.get("citations", 0)
            oa = "🔓" if p.get("is_open_access") else ""

            st.caption(f"*{author_str}* | {journal} ({year}) | {citations} citations {oa}")

            # Use specificity, fallback to relevance_rating for backwards compatibility
            rating = p.get("specificity", p.get("relevance_rating", ""))
            details = p.get("relevance_details", "")
            if rating or details:
                st.info(f"**Specificity: {rating}** — {details}" if rating else f"**Specificity:** {details}")

            abstract = strip_html(p.get("abstract", ""))
            if abstract:
                st.markdown("**Abstract:**")
                st.text(abstract)
            st.markdown("---")

    # === DOWNLOAD BUTTON ===
    if table_data:
        lit_csv = pd.DataFrame(table_data).to_csv(index=False)
        st.download_button(
            label="📥 Download Literature (CSV)",
            data=lit_csv,
            file_name="literature_results.csv",
            mime="text/csv",
            key=f"dl_lit_{unique_key}"
        )


def render_database_card(db_results, msg_idx: int = None):
    """Render database results with full table"""
    if not db_results:
        return

    _ks = f"_{msg_idx}" if msg_idx is not None else ""

    with st.expander("🗄️ Database Results", expanded=True):
        if isinstance(db_results, dict):
            # Metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Task", db_results.get("task_performed", "Unknown"))
            with col2:
                stats = db_results.get("statistics", {})
                st.metric("Total Results", stats.get("total_results", 0))
            with col3:
                st.metric("Time", f"{stats.get('execution_time_seconds', 0):.2f}s")

            # Query details
            with st.expander("Query Details", expanded=False):
                st.json(db_results.get("query_info", {}))

            # Results table
            results_list = db_results.get("results", [])
            if results_list:
                results_df = pd.DataFrame(results_list)

                # Select display columns
                display_cols = [c for c in
                                ["term_name", "term_id", "library", "num_genes", "overlap_count", "matched_query"]
                                if c in results_df.columns]

                # Add truncated description if present
                if "description" in results_df.columns:
                    results_df["description_short"] = results_df["description"].apply(
                        lambda x: str(x)[:100] + "..." if x and len(str(x)) > 100 else str(x) if x else ""
                    )
                    display_cols.append("description_short")

                if display_cols:
                    st.dataframe(_style_df(results_df[display_cols].head(100)), use_container_width=True,
                                 hide_index=True)

                    # Download button
                    db_csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=db_csv,
                        file_name=f"db_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"dl_db_{hash(str(results_list[:3]))}{_ks}"
                    )
            else:
                st.info("No results found")
        else:
            st.json(db_results)


def render_followup_suggestions(envelope: Dict, msg_idx: int = None):
    suggestions = []

    if envelope.get("gene_info"):
        gene = envelope["gene_info"]
        if isinstance(gene, list) and gene:
            gene = gene[0]
        symbol = gene.get("symbol") or gene.get("gene", "")
        if symbol:
            suggestions.append(f"Find papers about {symbol}")

    if envelope.get("enrichment_df") is not None:
        suggestions.append("Explain the top pathway in detail")

    if not suggestions:
        return

    _ks = f"_{msg_idx}" if msg_idx is not None else ""

    st.markdown(
        '<div class="followup-container"><span style="color: #06b6d4; font-size: 0.8rem;">💡 You might ask:</span></div>',
        unsafe_allow_html=True)

    cols = st.columns(len(suggestions[:3]))
    for col, sugg in zip(cols, suggestions[:3]):
        with col:
            if st.button(sugg, key=f"fw_{hash(sugg)}{_ks}", use_container_width=True):
                st.session_state.pending_query = sugg
                st.rerun()


# ===============================================================================
# OVERVIEW & LIBRARY PAGES
# ===============================================================================

def render_overview_page():
    """Comprehensive architecture overview page for Enrich.AI"""

    st.markdown("""
        <div class="dashboard-header">
            <div class="dashboard-title">Architecture Overview</div>
            <div class="dashboard-subtitle">How Enrich.AI works — an autonomous biology research assistant</div>
        </div>
    """, unsafe_allow_html=True)

    # ── 1. FIGURE 1 (centered, 70% width) ──
    import os
    figure_path = os.path.join("assets", "figure1.png")
    if not os.path.exists(figure_path):
        for alt in ["assets/figure1.pdf", "figure1.png"]:
            if os.path.exists(alt):
                figure_path = alt
                break

    if os.path.exists(figure_path):
        col_l, col_fig, col_r = st.columns([0.15, 0.6, 0.15])
        with col_fig:
            st.image(figure_path, use_container_width=True,
                     caption="Figure 1. Enrich.AI system architecture — from user query to synthesized biological insight.")
    else:
        st.info("Place `figure1.png` in the `assets/` folder to display the system architecture diagram.")

    st.divider()

    # ── 2. REACT PATTERN ──
    st.markdown("### ReAct Reasoning Pattern")
    st.markdown("""
    Enrich.AI uses a **ReAct (Reason + Act)** agent pattern powered by **Google Gemini 2.5 Flash** 
    and orchestrated via **LangGraph**. There is no fixed pipeline — the model decides the workflow 
    based on the biology of the question.
    """)

    st.markdown("""
    <div style="display:flex; gap:12px; margin:1rem 0 1.5rem 0;">
        <div style="flex:1; padding:1.1rem 1rem; background:linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%); border-radius:10px; border-left:3px solid #3b82f6;">
            <div style="color:#93c5fd; font-weight:700; font-size:0.95rem; letter-spacing:0.5px; margin-bottom:0.4rem;">THINK</div>
            <div style="color:#cbd5e1; font-size:0.8rem; line-height:1.45;">Analyze the biological question. Determine which tools and parameters are appropriate for the query.</div>
        </div>
        <div style="flex:1; padding:1.1rem 1rem; background:linear-gradient(135deg, #2d1b4e 0%, #1f1635 100%); border-radius:10px; border-left:3px solid #8b5cf6;">
            <div style="color:#c4b5fd; font-weight:700; font-size:0.95rem; letter-spacing:0.5px; margin-bottom:0.4rem;">ACT</div>
            <div style="color:#cbd5e1; font-size:0.8rem; line-height:1.45;">Call biological tools — enrichment analysis, literature search, gene info, database queries.</div>
        </div>
        <div style="flex:1; padding:1.1rem 1rem; background:linear-gradient(135deg, #0c3547 0%, #0f2130 100%); border-radius:10px; border-left:3px solid #06b6d4;">
            <div style="color:#67e8f9; font-weight:700; font-size:0.95rem; letter-spacing:0.5px; margin-bottom:0.4rem;">OBSERVE</div>
            <div style="color:#cbd5e1; font-size:0.8rem; line-height:1.45;">Interpret results. Evaluate relevance and significance. Identify gaps in understanding.</div>
        </div>
        <div style="flex:1; padding:1.1rem 1rem; background:linear-gradient(135deg, #0f3529 0%, #0d2318 100%); border-radius:10px; border-left:3px solid #10b981;">
            <div style="color:#6ee7b7; font-weight:700; font-size:0.95rem; letter-spacing:0.5px; margin-bottom:0.4rem;">ITERATE</div>
            <div style="color:#cbd5e1; font-size:0.8rem; line-height:1.45;">Continue reasoning until the query is comprehensively answered, then synthesize findings.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    > No hardcoded decision trees. If enrichment reveals cancer pathways, the model may look up 
    > gene functions next. If literature reveals a key paper, it may fetch full-text annotations. 
    > **Pure LLM reasoning drives every decision.**
    """)

    st.divider()

    # ── 3. FUNCTIONS ──
    st.markdown("### Autonomous Functions")
    st.markdown("The agent has access to **5 tools** connecting to external biological databases. "
                "It decides which to call, in what order, and with what parameters.")

    with st.expander("**Gene Information** — MyGene.info", expanded=False):
        st.markdown("""
        **What it does:** Retrieves comprehensive gene information including function summary, 
        Gene Ontology annotations (BP, MF, CC), pathway memberships (KEGG, Reactome), 
        disease associations, aliases, and genomic coordinates.

        **Data source:** [MyGene.info](https://mygene.info) — aggregates data from NCBI, UniProt, Ensembl, GO, KEGG, and more.
        """)
        st.markdown("**Parameters:**")
        st.table({
            "Parameter": ["gene", "organism"],
            "Type": ["str", "str"],
            "Default": ["— (required)", "9606"],
            "Description": [
                "Gene symbol (e.g., TP53, BRCA1, EGFR)",
                "NCBI taxonomy ID. Default 9606 for human."
            ]
        })

    with st.expander("**Database Query** — Enrichr (222 libraries)", expanded=False):
        st.markdown("""
        **What it does:** Queries across all **222 Enrichr libraries** to find pathways, GO terms, 
        disease associations, transcription factors, drug signatures, and cell types associated 
        with genes or terms.

        **Data source:** [Enrichr](https://maayanlab.cloud/Enrichr) — covers GO, KEGG, Reactome, WikiPathways, 
        MSigDB, DisGeNET, DrugMatrix, CellMarker, ENCODE, ChEA, and 200+ more.
        """)

        st.markdown("---")
        st.markdown("**`term_search`** — Find terms by keyword")
        st.markdown('*Example: "Search for GO terms related to apoptosis"*')
        st.table({
            "Parameter": ["query", "libraries", "limit"],
            "Default": ["— (required)", "None (defaults)", "20"],
            "Description": [
                "Keywords to search for in term names (e.g. 'apoptosis', 'T cell')",
                "Enrichr libraries to search (None = GO + KEGG + Reactome)",
                "Maximum results to return"
            ]
        })

        st.markdown("**`find_similar`** — Find biologically similar terms (Jaccard similarity)")
        st.markdown('*Example: "Find GO terms similar to T Cell Differentiation (GO:0030217)"*')
        st.table({
            "Parameter": ["query", "libraries", "similarity_threshold", "limit"],
            "Default": ["— (required)", "None (defaults)", "0.05", "20"],
            "Description": [
                "Term ID (e.g. GO:0030217), term name, or comma-separated gene symbols",
                "Enrichr libraries to compare against",
                "Minimum Jaccard score to include (0–1)",
                "Maximum results to return"
            ]
        })

        st.markdown("**`term_details`** — Get info about a specific term")
        st.markdown('*Example: "What is GO:0006915?"*')
        st.table({
            "Parameter": ["query", "libraries", "include_genes"],
            "Default": ["— (required)", "None (defaults)", "False"],
            "Description": [
                "Term ID (e.g. GO:0006915, R-HSA-109582)",
                "Enrichr libraries to search",
                "Whether to include the full gene list"
            ]
        })

        st.markdown("**`term_genes`** — Get genes belonging to a term")
        st.markdown('*Example: "What genes are in the KEGG Apoptosis pathway?"*')
        st.table({
            "Parameter": ["query", "libraries", "limit"],
            "Default": ["— (required)", "None (defaults)", "20"],
            "Description": [
                "Term ID or term name",
                "Enrichr libraries to search",
                "Maximum genes to return"
            ]
        })

        st.markdown("**`gene_terms`** — Find all terms containing a gene")
        st.markdown('*Example: "What pathways is TP53 involved in?"*')
        st.table({
            "Parameter": ["query", "libraries", "limit"],
            "Default": ["— (required)", "None (defaults)", "20"],
            "Description": [
                "Gene symbol(s), comma-separated (e.g. TP53, BRCA1)",
                "Enrichr libraries to search",
                "Maximum results to return"
            ]
        })

        st.markdown("**`library_browse`** — List all terms in a library")
        st.markdown('*Example: "Show me all terms in Reactome_2022"*')
        st.table({
            "Parameter": ["libraries", "limit", "include_genes"],
            "Default": ["— (required)", "20", "False"],
            "Description": [
                "Enrichr library name(s) to browse",
                "Maximum terms to return",
                "Whether to include gene lists for each term"
            ]
        })

    with st.expander("**Enrichment Analysis** — Enrichr", expanded=False):
        st.markdown("""
        **What it does:** Performs statistical enrichment analysis (Fisher\'s exact test) on a gene list 
        against selected Enrichr libraries. Returns enriched terms ranked by adjusted p-value, 
        with gene overlap, odds ratio, and combined score. Minimum 4 genes required.

        **Data source:** [Enrichr](https://maayanlab.cloud/Enrichr) — tests against 222 gene set libraries. 
        Default: GO BP, GO MF, GO CC, KEGG, Reactome, MSigDB Hallmark.
        """)
        st.markdown("**Parameters:**")
        st.table({
            "Parameter": ["genes", "libraries", "p_value_threshold", "top_n"],
            "Type": ["str", "str", "float", "int"],
            "Default": ["— (required)", "None (defaults)", "0.05", "None (all)"],
            "Description": [
                "Comma-separated gene symbols (minimum 4 genes)",
                "Comma-separated Enrichr libraries to test",
                "Adjusted p-value cutoff for significance",
                "Max terms per library. None = all significant."
            ]
        })

    with st.expander("**Literature Search** — Europe PMC", expanded=False):
        st.markdown("""
        **What it does:** Searches Europe PMC for scientific publications. Returns papers with titles, 
        abstracts, authors, journals, and citation counts. Each paper is scored for relevance 
        (High/Medium/Low) by the same Gemini model that drives the reasoning agent.

        **Data source:** [Europe PMC](https://europepmc.org) — indexes >40 million biomedical articles.
        """)
        st.markdown("**Parameters:**")
        st.table({
            "Parameter": ["query", "max_results", "min_year", "max_year", "sort_by"],
            "Type": ["str", "int", "int", "int", "str"],
            "Default": ["— (required)", "20", "None", "None", "relevance"],
            "Description": [
                "Search query for Europe PMC",
                "Maximum papers to return (up to 100)",
                "Minimum publication year filter",
                "Maximum publication year filter",
                "Sort: relevance, citations, or date"
            ]
        })

    with st.expander("**Paper Annotations** — Europe PMC", expanded=False):
        st.markdown("""
        **What it does:** Retrieves text-mined annotations from a specific paper — tagged genes, 
        diseases, drugs, organisms, and GO terms from the full text. The agent calls this 
        autonomously when it identifies high-relevance papers worth deeper analysis.

        **Data source:** [Europe PMC Annotations API](https://europepmc.org/AnnotationsApi)
        """)
        st.markdown("**Parameters:**")
        st.table({
            "Parameter": ["paper_id", "id_type", "annotation_types"],
            "Type": ["str", "str", "str"],
            "Default": ["— (required)", "auto", "None (all)"],
            "Description": [
                "Paper identifier: PMID, PMCID, or DOI",
                "Identifier type: pmid, pmcid, doi, or auto",
                "Comma-separated types (e.g., Gene_Proteins,Diseases)"
            ]
        })

    st.divider()

    # ── 4. VISUALIZATIONS (inline icons + methodology) ──
    st.markdown("### Interactive Visualizations")
    st.markdown("Enrichment results generate **6 interactive visualizations** across three categories. "
                "All are built with **Plotly**, support PDF export, and can be interpreted by the AI.")

    def _make_icon_bar():
        fig = go.Figure(go.Bar(x=[8,6,5,4,3], y=["A","B","C","D","E"], orientation='h',
            marker=dict(color=[8,6,5,4,3], colorscale='Viridis', showscale=False)))
        fig.update_layout(height=180, width=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    def _make_icon_bubble():
        fig = go.Figure(go.Scatter(x=[8,6,5,3.5,2.5], y=[1,2,3,4,5], mode='markers',
            marker=dict(size=[42,32,26,18,14], color=[8,6,5,3.5,2.5], colorscale='Viridis', showscale=False,
                        line=dict(width=1, color='#888'))))
        fig.update_layout(height=180, width=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    def _make_icon_upset():
        fig = make_subplots(rows=2, cols=1, row_heights=[0.55,0.45], vertical_spacing=0.02, shared_xaxes=True)
        fig.add_trace(go.Bar(x=[0,1,2,3,4], y=[15,10,8,5,3], marker=dict(color='#3b82f6'), width=0.5, showlegend=False), row=1, col=1)
        mat = [[1,0,1,1,0],[0,1,1,0,1],[0,0,0,1,1]]
        for ri in range(3):
            for ci in range(5):
                fig.add_trace(go.Scatter(x=[ci],y=[ri],mode='markers',
                    marker=dict(size=10 if mat[ri][ci] else 5, color='#1e293b' if mat[ri][ci] else '#ddd'),
                    showlegend=False), row=2, col=1)
        for ci in range(5):
            a = [r for r in range(3) if mat[r][ci]]
            if len(a) > 1:
                fig.add_trace(go.Scatter(x=[ci,ci],y=[min(a),max(a)],mode='lines',
                    line=dict(color='#1e293b',width=2),showlegend=False), row=2, col=1)
        fig.update_layout(height=180, width=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return fig

    def _make_icon_dendrogram():
        fig = go.Figure()
        # Simplified tree shape
        branches = [
            ([5,5,15,15],[0,0.3,0.3,0]),    # left pair
            ([25,25,35,35],[0,0.4,0.4,0]),   # right pair
            ([10,10,30,30],[0.3,1.0,1.0,0.4]), # top merge
            ([45,45,20,20],[0,0.6,0.6,1.0]),   # far right merges in
        ]
        for xs, ys in branches:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='#3b82f6', width=2.5), showlegend=False))
        fig.update_layout(height=180, width=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    def _make_icon_heatmap():
        import numpy as np
        np.random.seed(7)
        z = np.array([[1,.7,.2,.15],[.7,1,.25,.1],[.2,.25,1,.6],[.15,.1,.6,1]])
        fig = go.Figure(go.Heatmap(z=z, colorscale='Reds', showscale=False))
        fig.update_layout(height=180, width=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    def _make_icon_cnetplot():
        fig = go.Figure()
        edges = [(0.3,0.8,0.15,0.5),(0.3,0.8,0.35,0.45),(0.3,0.8,0.45,0.55),
                 (0.75,0.75,0.65,0.45),(0.75,0.75,0.85,0.5),
                 (0.5,0.2,0.4,0.4),(0.5,0.2,0.6,0.35)]
        for x1,y1,x2,y2 in edges:
            fig.add_trace(go.Scatter(x=[x1,x2],y=[y1,y2],mode='lines',line=dict(color='#bbb',width=1),showlegend=False))
        fig.add_trace(go.Scatter(x=[0.3,0.75,0.5],y=[0.8,0.75,0.2],mode='markers',
            marker=dict(size=16,symbol='triangle-up',color=['#e74c3c','#3498db','#2ecc71'],line=dict(width=1,color='#333')),showlegend=False))
        fig.add_trace(go.Scatter(x=[0.15,0.35,0.45,0.65,0.85,0.4,0.6],y=[0.5,0.45,0.55,0.45,0.5,0.4,0.35],mode='markers',
            marker=dict(size=7,color=['#e74c3c','#e74c3c','#e74c3c','#3498db','#3498db','#2ecc71','#2ecc71']),showlegend=False))
        fig.update_layout(height=180, width=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, range=[-0.05,1.05]), yaxis=dict(visible=False, range=[0.05,0.95]))
        return fig

    plot_config = {'displayModeBar': False, 'staticPlot': True}

    viz_items = [
        ("Term Analysis", [
            ("Bar Plot", _make_icon_bar,
             "Enriched terms are ranked by **-log₁₀(adjusted p-value)** from the Enrichr Fisher's exact test. "
             "The transformation converts small p-values into large bars, making the most statistically "
             "significant terms visually prominent. Color encodes the same significance gradient. "
             "Built directly from Enrichr's `adjusted_p_value` column."),
            ("Bubble Plot", _make_icon_bubble,
             "Two-dimensional encoding: the **x-axis** maps -log₁₀(adjusted p-value) for significance, "
             "while **bubble diameter** scales with the number of overlapping genes between the input list "
             "and the gene set. This separates terms that are significant but involve few genes from "
             "those with broad gene involvement. Same Enrichr data, additional `gene_count` dimension."),
            ("UpSet Plot", _make_icon_upset,
             "Computes **all pairwise and higher-order gene set intersections** between the top enriched terms. "
             "For each term, the set of overlapping genes is extracted. All 2ⁿ possible combinations are evaluated, "
             "and the intersection size (shared gene count) is plotted as a bar. The dot matrix below indicates "
             "which terms participate in each intersection. More scalable than Venn diagrams for >3 sets."),
        ]),
        ("Clustering", [
            ("Dendrogram", _make_icon_dendrogram,
             "A **Jaccard similarity matrix** is computed pairwise between all enriched term gene sets: "
             "J(A,B) = |A∩B| / |A∪B|. This is converted to a distance matrix (1 − J) and clustered using "
             "**Ward's minimum variance linkage** (scipy `linkage`). The resulting hierarchy is rendered "
             "as a tree — terms sharing more genes merge at lower branch heights. Leaf order follows the "
             "optimal ordering from `leaves_list`."),
            ("Similarity Heatmap", _make_icon_heatmap,
             "The same **Jaccard similarity matrix** is displayed as a heatmap, with rows and columns "
             "reordered by **Ward's hierarchical clustering**. A colored sidebar shows cluster assignments "
             "(via `fcluster` with `maxclust` criterion). Each cluster is annotated with **keyword summaries** "
             "extracted from term names using word frequency analysis (stopword-filtered). "
             "Design inspired by the R package "
             "[SimplifyEnrichment](https://bioconductor.org/packages/SimplifyEnrichment/)."),
        ]),
        ("Network", [
            ("Concept Network", _make_icon_cnetplot,
             "Terms are clustered using **Ward's linkage** on the Jaccard distance matrix, then each cluster "
             "is represented as a single triangle node (▲) named by its top keywords. Individual genes (●) are "
             "connected to their parent cluster. A **spring-force layout** (NetworkX `spring_layout`, k=0.5, "
             "30 iterations) positions nodes so that same-cluster elements group spatially. Genes appearing "
             "in multiple clusters are assigned to their primary cluster. Cluster colors use Plotly's Set1 palette."),
        ]),
    ]

    for category, plots in viz_items:
        st.markdown(f"#### {category}")
        for title, icon_fn, description in plots:
            col_icon, col_desc = st.columns([0.22, 0.78])
            with col_icon:
                fig = icon_fn()
                st.plotly_chart(fig, use_container_width=True, config=plot_config)
            with col_desc:
                st.markdown(f"**{title}**")
                st.markdown(description, unsafe_allow_html=True)
            st.markdown("")

    st.markdown("""
    > All visualizations include **PDF export**, **interactive parameter controls** 
    > (term selection, color palettes, clustering method), and optional 
    > **AI interpretation** powered by Gemini.
    """)

    st.divider()

    # ── 5. HOW TO USE ──
    st.markdown("### How to Use Enrich.AI")

    st.markdown("#### Getting a Gemini API Key")
    st.markdown(
        "To use Enrich.AI, you need a free Google Gemini API key. "
        "Get one here: **[Google AI Studio — Get API Key](https://aistudio.google.com/app/apikey)**"
    )

    st.markdown("#### Quick Start")
    st.markdown("""
    1. **Enter your Gemini API key** in the sidebar and verify it shows a green checkmark
    2. **Write anything in the chat** — Enrich.AI understands natural language queries about biology
    3. **For enrichment analysis:**
       - Libraries can be selected from the **Enrichr Libraries** button in the sidebar, or specified manually in chat
       - Paste your gene list directly in chat (e.g., *"Perform enrichment analysis on TP53, BRCA1, MYC, EGFR, AKT1"*)
       - Minimum **4 genes** required for statistical enrichment
    4. **For literature search:** Ask naturally (e.g., *"Find papers about TP53 in lung cancer"*)
    5. **For gene info:** Ask naturally (e.g., *"What does BRCA1 do?"*)
    6. **Follow up** — the system remembers context, so you can refine and ask follow-up questions
    """)

    st.divider()

    # ── IMPORTANT NOTES ──
    st.markdown("### ⚠️ Important Usage Notes")
    st.markdown("""
    <div style="padding: 1.2rem 1.4rem; background: linear-gradient(135deg, #3b1f0b 0%, #1a1008 100%);
                border-radius: 12px; border-left: 4px solid #f59e0b; margin: 0.8rem 0 1.5rem 0;">
        <div style="color: #fbbf24; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem;">
            🔄 Streamlit Single-Threaded Limitation
        </div>
        <div style="color: #e2e8f0; font-size: 0.88rem; line-height: 1.6;">
            Enrich.AI runs on <strong>Streamlit</strong>, which processes one action at a time. 
            While a query is running (spinner visible), <strong>do not click buttons, switch pages, 
            or submit new queries</strong> — this will interrupt the current analysis and may cause 
            unexpected behavior. Wait for the spinner to finish before interacting with the interface.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── RUN LOCALLY ──
    st.markdown("### 🖥️ Run Locally")
    st.markdown(
        "Enrich.AI can be run on your own machine. "
        "You need **Python 3.10+** and a free **Google Gemini API key**."
    )

    st.markdown("#### Option A — Conda (recommended)")
    st.html("""
    <div style="background: #000000; border: 1px solid #1e293b; border-radius: 10px;
                padding: 1.1rem 1.3rem; margin: 0.5rem 0 1.5rem 0; overflow-x: auto;">
<pre style="margin: 0; color: #e2e8f0; font-size: 0.84rem; line-height: 1.75;
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
            background: transparent; white-space: pre; overflow-x: auto;">
<span style="color: #6b7280; font-style: italic;"># 1. Clone the repository</span>
<span style="color: #f472b6;">git</span> <span style="color: #22d3ee;">clone</span> https://github.com/jkouprey/Enrich.AI.git
<span style="color: #f472b6;">cd</span> EnrichAI

<span style="color: #6b7280; font-style: italic;"># 2. Create the conda environment</span>
<span style="color: #f472b6;">conda</span> <span style="color: #22d3ee;">env create</span> -f environment.yml

<span style="color: #6b7280; font-style: italic;"># 3. Activate</span>
<span style="color: #f472b6;">conda</span> <span style="color: #22d3ee;">activate</span> enrichai

<span style="color: #6b7280; font-style: italic;"># 4. Launch</span>
<span style="color: #f472b6;">streamlit</span> <span style="color: #22d3ee;">run</span> app.py
</pre>
    </div>
    """)

    st.markdown("#### Option B — pip + virtual environment")
    st.html("""
    <div style="background: #000000; border: 1px solid #1e293b; border-radius: 10px;
                padding: 1.1rem 1.3rem; margin: 0.5rem 0 1.5rem 0; overflow-x: auto;">
<pre style="margin: 0; color: #e2e8f0; font-size: 0.84rem; line-height: 1.75;
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
            background: transparent; white-space: pre; overflow-x: auto;">
<span style="color: #6b7280; font-style: italic;"># 1. Clone the repository</span>
<span style="color: #f472b6;">git</span> <span style="color: #22d3ee;">clone</span> https://github.com/jkouprey/Enrich.AI.git
<span style="color: #f472b6;">cd</span> EnrichAI

<span style="color: #6b7280; font-style: italic;"># 2. Create a virtual environment</span>
<span style="color: #f472b6;">python</span> -m venv venv
<span style="color: #f472b6;">source</span> venv/bin/activate   <span style="color: #6b7280; font-style: italic;"># Windows: venv&#92;Scripts&#92;activate</span>

<span style="color: #6b7280; font-style: italic;"># 3. Install dependencies</span>
<span style="color: #f472b6;">pip</span> <span style="color: #22d3ee;">install</span> -r requirements.txt

<span style="color: #6b7280; font-style: italic;"># 4. Launch</span>
<span style="color: #f472b6;">streamlit</span> <span style="color: #22d3ee;">run</span> app.py
</pre>
    </div>
    """)

    st.markdown("#### API Key Configuration")
    st.markdown("""
There are two ways to provide your Gemini API key:

**In the UI** — Enter it in the sidebar text field after launching. 
This is the easiest approach and requires no configuration files.

**As an environment variable** — Set it before launching:
""")
    st.html("""
    <div style="background: #000000; border: 1px solid #1e293b; border-radius: 10px;
                padding: 1.1rem 1.3rem; margin: 0.5rem 0 1.5rem 0; overflow-x: auto;">
<pre style="margin: 0; color: #e2e8f0; font-size: 0.84rem; line-height: 1.75;
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
            background: transparent; white-space: pre; overflow-x: auto;">
<span style="color: #6b7280; font-style: italic;"># Linux / macOS</span>
<span style="color: #f472b6;">export</span> <span style="color: #c084fc;">GOOGLE_API_KEY</span>=<span style="color: #34d399;">"your-key-here"</span>
<span style="color: #f472b6;">streamlit</span> <span style="color: #22d3ee;">run</span> app.py

<span style="color: #6b7280; font-style: italic;"># Windows (PowerShell)</span>
<span style="color: #c084fc;">$env:GOOGLE_API_KEY</span>=<span style="color: #34d399;">"your-key-here"</span>
<span style="color: #f472b6;">streamlit</span> <span style="color: #22d3ee;">run</span> app.py
</pre>
    </div>
    """)

    st.markdown(
        "Get a free API key at "
        "**[Google AI Studio](https://aistudio.google.com/app/apikey)**."
    )

    st.markdown("#### Project Structure")
    st.html("""
    <div style="background: #000000; border: 1px solid #1e293b; border-radius: 10px;
                padding: 1.1rem 1.3rem; margin: 0.5rem 0 1.5rem 0; overflow-x: auto;">
<pre style="margin: 0; color: #e2e8f0; font-size: 0.84rem; line-height: 1.75;
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
            background: transparent; white-space: pre; overflow-x: auto;">
EnrichAI/
&#9500;&#9472;&#9472; <span style="color: #22d3ee;">app.py</span>                <span style="color: #6b7280;"># Streamlit interface &amp; visualizations</span>
&#9500;&#9472;&#9472; <span style="color: #22d3ee;">reasoning_engine.py</span>   <span style="color: #6b7280;"># LangGraph ReAct agent</span>
&#9500;&#9472;&#9472; <span style="color: #22d3ee;">tools.py</span>              <span style="color: #6b7280;"># Biological database functions</span>
&#9500;&#9472;&#9472; <span style="color: #22d3ee;">visualizer.py</span>         <span style="color: #6b7280;"># Plot rendering &amp; AI interpretation</span>
&#9500;&#9472;&#9472; <span style="color: #22d3ee;">config.py</span>             <span style="color: #6b7280;"># Model &amp; logging configuration</span>
&#9500;&#9472;&#9472; <span style="color: #22d3ee;">tool_registry.py</span>      <span style="color: #6b7280;"># Enrichr library browser</span>
&#9500;&#9472;&#9472; <span style="color: #c084fc;">requirements.txt</span>      <span style="color: #6b7280;"># pip dependencies</span>
&#9500;&#9472;&#9472; <span style="color: #c084fc;">environment.yml</span>       <span style="color: #6b7280;"># Conda environment</span>
&#9492;&#9472;&#9472; assets/
    &#9492;&#9472;&#9472; <span style="color: #34d399;">figure1.png</span>       <span style="color: #6b7280;"># Architecture diagram</span>
</pre>
    </div>
    """)

    st.markdown("")
    if st.button("← Back to Chat", key="back_chat"):
        st.session_state.current_page = "chat"
        st.rerun()


def render_library_browser():
    """Render fancy library browser with Enrichr-style categories"""

    # Initialize expander states if not present
    if "lib_expander_states" not in st.session_state:
        st.session_state.lib_expander_states = {}

    st.markdown("""
        <div class="dashboard-header">
            <div class="dashboard-title">📚 Enrichr Library Browser</div>
            <div class="dashboard-subtitle">Browse and select from 222+ biological gene set libraries</div>
        </div>
    """, unsafe_allow_html=True)

    # Custom CSS for library buttons
    st.markdown("""
        <style>
        /* Selected library button - vibrant blue */
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4) !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5) !important;
        }

        /* Unselected library button - subtle gray */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: rgba(100, 116, 139, 0.1) !important;
            color: #94a3b8 !important;
            border: 1px solid rgba(100, 116, 139, 0.2) !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            background: rgba(100, 116, 139, 0.2) !important;
            color: #cbd5e1 !important;
            border: 1px solid rgba(100, 116, 139, 0.3) !important;
        }

        /* Category action buttons */
        .stButton > button {
            font-size: 0.85rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Define Enrichr-style categories with icons and descriptions
    ENRICHR_CATEGORIES = {
        "🧬 Transcription": {
            "icon": "🧬",
            "description": "Transcription factor targets, ChIP-seq, regulatory elements",
            "color": "#8b5cf6",
            "keywords": ["ChEA", "ENCODE", "TRANSFAC", "JASPAR", "ARCHS4_TF", "TF_Perturbations",
                         "Transcription_Factor", "TRRUST", "TF-LOF", "Genome_Browser_PWMs",
                         "ENCODE_Histone", "Epigenomics", "TargetScan", "miRTarBase", "PPI"]
        },
        "🛤️ Pathways": {
            "icon": "🛤️",
            "description": "Biological pathways from KEGG, Reactome, WikiPathways & more",
            "color": "#3b82f6",
            "keywords": ["KEGG", "Reactome", "WikiPathways", "BioCarta", "Panther", "BioPlanet",
                         "NCI-Nature", "HumanCyc", "Elsevier_Pathway", "MSigDB_Hallmark",
                         "Signaling_Pathways", "NetPath"]
        },
        "🏷️ Ontologies": {
            "icon": "🏷️",
            "description": "Gene Ontology, phenotypes, cellular components",
            "color": "#10b981",
            "keywords": ["GO_", "Gene_Ontology", "Human_Phenotype", "MGI_Mammalian_Phenotype",
                         "Jensen_COMPARTMENTS", "Jensen_TISSUES", "Chromosome_Location",
                         "MSigDB_Computational", "InterPro", "Pfam"]
        },
        "💊 Diseases & Drugs": {
            "icon": "💊",
            "description": "Disease associations, drug targets, GWAS, pharmacology",
            "color": "#ef4444",
            "keywords": ["GWAS", "DisGeNET", "OMIM", "ClinVar", "DrugMatrix", "DSigDB",
                         "Drug_Perturbations", "Jensen_DISEASES", "Rare_Diseases", "DepMap",
                         "Drug_Signatures", "Orphanet", "CTD", "PharmGKB", "TTD", "SIDER",
                         "L1000", "LINCS", "Connectivity_Map", "Achilles", "CORUM"]
        },
        "🔬 Cell Types & Tissues": {
            "icon": "🔬",
            "description": "Cell type markers, tissue expression, single-cell data",
            "color": "#f59e0b",
            "keywords": ["CellMarker", "PanglaoDB", "Tabula", "HuBMAP", "Azimuth", "Descartes",
                         "GTEx", "Human_Gene_Atlas", "Mouse_Gene_Atlas", "ARCHS4_Tissues",
                         "ARCHS4_Cell", "Allen_Brain", "Cancer_Cell_Line", "NCI-60", "CCLE",
                         "Tissue_Protein", "ProteomicsDB"]
        },
        "🧪 Perturbations & Signatures": {
            "icon": "🧪",
            "description": "Gene perturbations, knockout signatures, disease models",
            "color": "#ec4899",
            "keywords": ["GEO", "Perturbations", "Signatures", "Ligand_Perturbations",
                         "Kinase_Perturbations", "MCF7", "Virus_Perturbations", "MSigDB_Oncogenic",
                         "Old", "Aging", "SysMyo", "SILAC", "Phosphatase", "Kinase_Substrates",
                         "KEA", "RNA-Seq_Disease", "Microbe_Perturbations"]
        },
        "🧠 Legacy & Misc": {
            "icon": "🧠",
            "description": "Older libraries, crowd-sourced data, specialized databases",
            "color": "#6b7280",
            "keywords": ["2013", "2014", "2015", "2016", "Crowd", "GeneSigDB", "NURSA",
                         "VirusMINT", "Viral_CRISPR", "HomoloGene", "dbGaP", "UK_Biobank",
                         "lncRNA", "FANTOM", "Enrichr_Libraries", "Data_Integrations"]
        }
    }

    def categorize_library(lib_name: str) -> str:
        """Categorize a library based on its name"""
        lib_upper = lib_name.upper()

        for category, info in ENRICHR_CATEGORIES.items():
            for keyword in info["keywords"]:
                if keyword.upper() in lib_upper:
                    return category

        # Default categorization by common patterns
        if any(x in lib_upper for x in ["GO_", "ONTOLOGY", "HPO", "MPO"]):
            return "🏷️ Ontologies"
        elif any(x in lib_upper for x in ["PATHWAY", "KEGG", "REACTOME", "WIKI"]):
            return "🛤️ Pathways"
        elif any(x in lib_upper for x in ["TF", "CHIP", "ENCODE", "TRANSC"]):
            return "🧬 Transcription"
        elif any(x in lib_upper for x in ["DRUG", "DISEASE", "GWAS", "OMIM"]):
            return "💊 Diseases & Drugs"
        elif any(x in lib_upper for x in ["CELL", "TISSUE", "ATLAS"]):
            return "🔬 Cell Types & Tissues"
        elif any(x in lib_upper for x in ["PERTURB", "SIG", "GEO"]):
            return "🧪 Perturbations & Signatures"
        else:
            return "🧠 Legacy & Misc"

    # Fetch and categorize libraries
    all_libraries = get_available_enrichr_libraries()

    # Re-categorize using our Enrichr-style categories
    categorized = defaultdict(list)
    flat_libs = []

    for category, libs in all_libraries.items():
        if isinstance(libs, list):
            flat_libs.extend(libs)
        else:
            flat_libs.append(category)

    if not flat_libs:
        flat_libs = list(all_libraries.keys())

    for lib_name in flat_libs:
        cat = categorize_library(lib_name)
        categorized[cat].append(lib_name)

    if "all_enrichr_libs" not in st.session_state:
        st.session_state.all_enrichr_libs = flat_libs

    total_libs = len(flat_libs)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Libraries", total_libs)
    with col2:
        st.metric("✅ Selected", len(st.session_state.selected_libraries))
    with col3:
        st.metric("📁 Categories", len(ENRICHR_CATEGORIES))
    with col4:
        if st.button("🗑️ Clear All", key="clear_libs_btn", use_container_width=True):
            st.session_state.selected_libraries = []
            st.rerun()

    st.markdown("---")

    # Search filter
    search_query = st.text_input("🔍 Search libraries...", placeholder="Type to filter libraries...", key="lib_search")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Quick select buttons
    st.markdown("**⚡ Quick Select:**")
    quick_cols = st.columns(4)

    with quick_cols[0]:
        if st.button("🎯 Core Analysis", key="quick_core", use_container_width=True, help="GO + KEGG + Reactome"):
            core_libs = ["GO_Biological_Process_2023", "GO_Molecular_Function_2023",
                         "GO_Cellular_Component_2023", "KEGG_2021_Human", "Reactome_2022"]
            for lib in core_libs:
                if lib not in st.session_state.selected_libraries and lib in flat_libs:
                    st.session_state.selected_libraries.append(lib)
            st.rerun()

    with quick_cols[1]:
        if st.button("🩺 Disease Focus", key="quick_disease", use_container_width=True, help="GWAS + DisGeNET + OMIM"):
            disease_libs = ["GWAS_Catalog_2023", "DisGeNET", "OMIM_Disease", "OMIM_Expanded",
                            "Jensen_DISEASES", "ClinVar_2019", "Rare_Diseases_GeneRIF_ARCHS4_Predictions"]
            for lib in disease_libs:
                if lib not in st.session_state.selected_libraries and lib in flat_libs:
                    st.session_state.selected_libraries.append(lib)
            st.rerun()

    with quick_cols[2]:
        if st.button("🧬 TF & Regulation", key="quick_tf", use_container_width=True, help="ChEA + ENCODE + TRRUST"):
            tf_libs = ["ChEA_2022", "ENCODE_TF_ChIP-seq_2015", "TRRUST_Transcription_Factors_2019",
                       "TRANSFAC_and_JASPAR_PWMs"]
            for lib in tf_libs:
                if lib not in st.session_state.selected_libraries and lib in flat_libs:
                    st.session_state.selected_libraries.append(lib)
            st.rerun()

    with quick_cols[3]:
        if st.button("🔬 Cell Types", key="quick_cells", use_container_width=True, help="CellMarker + PanglaoDB + GTEx"):
            cell_libs = ["CellMarker_Augmented_2021", "PanglaoDB_Augmented_2021",
                         "GTEx_Tissue_Expression_Up", "GTEx_Tissue_Expression_Down",
                         "Human_Gene_Atlas", "Tabula_Sapiens"]
            for lib in cell_libs:
                if lib not in st.session_state.selected_libraries and lib in flat_libs:
                    st.session_state.selected_libraries.append(lib)
            st.rerun()

    st.markdown("---")

    # Display categories with styled expanders
    for category in ENRICHR_CATEGORIES.keys():
        if category not in categorized:
            continue

        libs_in_cat = sorted(categorized[category])
        cat_info = ENRICHR_CATEGORIES[category]

        # Filter by search
        if search_query:
            libs_in_cat = [lib for lib in libs_in_cat if search_query.lower() in lib.lower()]

        if not libs_in_cat:
            continue

        # Count selected in this category
        selected_in_cat = sum(1 for lib in libs_in_cat if lib in st.session_state.selected_libraries)

        # Create a safe key from category name
        cat_key = re.sub(r'[^a-zA-Z0-9]', '_', category)

        # Check if this category should be expanded (from session state)
        is_expanded = st.session_state.lib_expander_states.get(cat_key, False)

        # Category header
        with st.expander(f"{category} — {len(libs_in_cat)} libraries ({selected_in_cat} selected)",
                         expanded=is_expanded):
            # Track that this expander is now open (user clicked it)
            st.session_state.lib_expander_states[cat_key] = True

            st.caption(f"*{cat_info['description']}*")

            # Select/Deselect all in category
            btn_cols = st.columns([1, 1, 4])
            with btn_cols[0]:
                if st.button(f"✓ Select All", key=f"sel_{cat_key}", use_container_width=True):
                    for lib in libs_in_cat:
                        if lib not in st.session_state.selected_libraries:
                            st.session_state.selected_libraries.append(lib)
                    st.rerun()
            with btn_cols[1]:
                if st.button(f"✗ Clear", key=f"clr_{cat_key}", use_container_width=True):
                    st.session_state.selected_libraries = [
                        lib for lib in st.session_state.selected_libraries
                        if lib not in libs_in_cat
                    ]
                    st.rerun()

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

            # Display as a grid of toggle buttons (3 columns)
            cols = st.columns(3)
            for i, lib_name in enumerate(libs_in_cat):
                with cols[i % 3]:
                    is_selected = lib_name in st.session_state.selected_libraries
                    display_name = lib_name.replace("_", " ")
                    if len(display_name) > 28:
                        display_name = display_name[:26] + "..."

                    # Use button with different style based on selection
                    btn_label = f"{'✓ ' if is_selected else ''}{display_name}"
                    btn_type = "primary" if is_selected else "secondary"

                    if st.button(
                            btn_label,
                            key=f"lib_{lib_name}",
                            use_container_width=True,
                            type=btn_type
                    ):
                        # Toggle selection
                        if lib_name in st.session_state.selected_libraries:
                            st.session_state.selected_libraries.remove(lib_name)
                        else:
                            st.session_state.selected_libraries.append(lib_name)
                        st.rerun()

    st.markdown("---")

    # Selected libraries summary
    if st.session_state.selected_libraries:
        with st.expander(f"✅ Selected Libraries ({len(st.session_state.selected_libraries)})", expanded=True):
            # Group selected by category for nice display
            selected_by_cat = defaultdict(list)
            for lib in st.session_state.selected_libraries:
                cat = categorize_library(lib)
                selected_by_cat[cat].append(lib)

            for cat, libs in sorted(selected_by_cat.items()):
                st.markdown(f"**{cat}** ({len(libs)})")
                for lib in sorted(libs):
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        st.markdown(f"• {lib.replace('_', ' ')}")
                    with col2:
                        if st.button("✗", key=f"rm_{lib}", help=f"Remove {lib}"):
                            st.session_state.selected_libraries.remove(lib)
                            st.rerun()

    # Back button
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Chat", key="back_from_libs", use_container_width=True):
            # Clear expander states when leaving
            st.session_state.lib_expander_states = {}
            st.session_state.current_page = "chat"
            st.rerun()
    with col2:
        if st.button("🔄 Collapse All", key="collapse_all", use_container_width=True):
            st.session_state.lib_expander_states = {}
            st.rerun()


# ===============================================================================
# MAIN
# ===============================================================================

def handle_user_query(query: str):
    if not st.session_state.api_key:
        st.error("Please enter your Gemini API key.")
        return None

    if st.session_state.reasoning_engine is None:
        os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
        try:
            st.session_state.reasoning_engine = create_reasoning_engine()
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return None

    try:
        result = st.session_state.reasoning_engine.run(
            query,
            selected_libraries=st.session_state.selected_libraries or None,
            progress_callback=st.session_state.get("_progress_cb"),
        )
        envelope = result.get("envelope", {})
        return {"content": envelope.get("final_text", "I processed your request."), "envelope": envelope}
    except Exception as e:
        logger.error(f"Query failed: {e}\n{traceback.format_exc()}")
        st.error(f"Error: {str(e)}")
        return None


def render_chat_interface():
    render_dashboard()

    messages = st.session_state.current_session.get("messages", [])
    for idx, msg in enumerate(messages):
        # For assistant messages, get the previous user message for avatar context
        previous_user_msg = ""
        if msg.get("role") == "assistant" and idx > 0:
            prev_msg = messages[idx - 1]
            if prev_msg.get("role") == "user":
                previous_user_msg = prev_msg.get("content", "")

        display_message_card(msg, idx, previous_user_msg)

    # Handle scroll to pinned message if requested
    if st.session_state.get("scroll_to_message") is not None:
        target_idx = st.session_state.scroll_to_message
        st.session_state.scroll_to_message = None  # Clear after use
        inject_scroll_js(f"msg-{target_idx}")

    # Auto-scroll to bottom after new message
    if st.session_state.get("scroll_to_bottom", False):
        st.session_state.scroll_to_bottom = False  # Clear after use
        # Scroll to last message using same method that works for pinned messages
        last_msg_idx = len(messages) - 1
        if last_msg_idx >= 0:
            inject_scroll_js(f"msg-{last_msg_idx}")

    # Check if we're currently processing
    if st.session_state.get("processing", False):
        st.info("🔄 Analysis in progress — please wait for the current query to complete before interacting.")
        return

    def _process_query(query: str):
        """Shared logic for processing a user query (pending or chat input)."""
        st.session_state.processing = True

        st.session_state.current_session["messages"].append({"role": "user", "content": query})
        if st.session_state.current_session["title"] == "New Chat":
            st.session_state.current_session["title"] = query[:25] + "..."

        with st.chat_message("user", avatar=_get_user_avatar()):
            st.markdown(query)

        with st.status("🔄 Analyzing...", expanded=True) as status:
            def _cb(msg):
                status.write(msg)
            st.session_state["_progress_cb"] = _cb
            result = handle_user_query(query)
            st.session_state["_progress_cb"] = None
            status.update(label="✓ Analysis complete", state="complete", expanded=False)

        st.session_state.processing = False

        if result:
            st.session_state.current_session["messages"].append({
                "role": "assistant", "content": result["content"], "envelope": result["envelope"]
            })
            save_current_session()
            st.session_state.scroll_to_bottom = True
            st.rerun()

    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        _process_query(query)

    # Chat input
    user_query = st.chat_input("Ask about genes, pathways, literature...",
                               disabled=st.session_state.get("processing", False))

    if user_query:
        _process_query(user_query)


def main():
    init_session_state()
    apply_fancy_styling()
    render_sidebar()

    if st.session_state.current_page == "overview":
        render_overview_page()
    elif st.session_state.current_page == "libraries":
        render_library_browser()
    else:
        render_chat_interface()


if __name__ == "__main__":
    main()
