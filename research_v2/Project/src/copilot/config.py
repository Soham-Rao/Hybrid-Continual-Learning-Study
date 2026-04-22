"""Configuration helpers for the dashboard copilot."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotSettings:
    """Runtime settings for the local copilot foundation."""

    base_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen2.5:7b-instruct"
    connect_timeout_sec: float = 1.5
    read_timeout_sec: float = 30.0
    health_timeout_sec: float = 1.5
    prompt_budget_chars: int = 12000
    max_context_items: int = 12


def _clean_base_url(url: str) -> str:
    return url.rstrip("/")


def load_copilot_settings() -> CopilotSettings:
    """Load copilot settings from environment variables."""

    return CopilotSettings(
        base_url=_clean_base_url(os.getenv("COPILOT_OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
        default_model=os.getenv("COPILOT_OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        connect_timeout_sec=float(os.getenv("COPILOT_OLLAMA_CONNECT_TIMEOUT_SEC", "1.5")),
        read_timeout_sec=float(os.getenv("COPILOT_OLLAMA_READ_TIMEOUT_SEC", "30.0")),
        health_timeout_sec=float(os.getenv("COPILOT_OLLAMA_HEALTH_TIMEOUT_SEC", "1.5")),
        prompt_budget_chars=int(os.getenv("COPILOT_PROMPT_BUDGET_CHARS", "12000")),
        max_context_items=int(os.getenv("COPILOT_MAX_CONTEXT_ITEMS", "12")),
    )
