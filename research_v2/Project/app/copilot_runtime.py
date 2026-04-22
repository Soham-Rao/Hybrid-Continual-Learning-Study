"""App-facing helpers for copilot runtime status."""

from __future__ import annotations

import streamlit as st

from src.copilot import CopilotSettings, OllamaClient, OllamaStatus, load_copilot_settings


@st.cache_resource(show_spinner=False)
def get_copilot_settings() -> CopilotSettings:
    return load_copilot_settings()


@st.cache_resource(show_spinner=False)
def get_ollama_client() -> OllamaClient:
    return OllamaClient(get_copilot_settings())


@st.cache_data(show_spinner=False, ttl=15)
def get_copilot_status() -> OllamaStatus:
    client = get_ollama_client()
    settings = get_copilot_settings()
    return client.get_status(settings.default_model)
