from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.copilot import CopilotSettings, OllamaClient, load_copilot_settings


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_load_copilot_settings_uses_expected_defaults(monkeypatch) -> None:
    monkeypatch.delenv("COPILOT_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("COPILOT_OLLAMA_BASE_URL", raising=False)
    settings = load_copilot_settings()
    assert settings.default_model == "qwen2.5:7b-instruct"
    assert settings.base_url == "http://127.0.0.1:11434"
    assert settings.prompt_budget_chars >= 4000


def test_ollama_status_reports_ready_when_model_is_installed() -> None:
    def transport(request, timeout):
        assert request.full_url.endswith("/api/tags")
        return _FakeResponse({"models": [{"name": "qwen2.5:7b-instruct"}, {"name": "llama3.1:8b-instruct-q4_K_M"}]})

    client = OllamaClient(CopilotSettings(), transport=transport)
    status = client.get_status()
    assert status.available is True
    assert status.model_available is True
    assert "ready" in status.message.lower()


def test_ollama_status_reports_missing_model_cleanly() -> None:
    def transport(request, timeout):
        return _FakeResponse({"models": [{"name": "llama3.1:8b-instruct-q4_K_M"}]})

    client = OllamaClient(CopilotSettings(default_model="qwen2.5:7b-instruct"), transport=transport)
    status = client.get_status()
    assert status.available is True
    assert status.model_available is False
    assert "not installed" in status.message.lower()


def test_ollama_generate_trims_prompt_and_returns_text() -> None:
    calls = []

    def transport(request, timeout):
        calls.append((request.full_url, request.data))
        if request.full_url.endswith("/api/tags"):
            return _FakeResponse({"models": [{"name": "qwen2.5:7b-instruct"}]})
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "qwen2.5:7b-instruct"
        assert len(payload["prompt"]) == 12
        return _FakeResponse(
            {
                "response": " Grounded explanation. ",
                "prompt_eval_count": 13,
                "eval_count": 42,
                "done_reason": "stop",
            }
        )

    client = OllamaClient(CopilotSettings(prompt_budget_chars=12), transport=transport)
    result = client.generate("abcdefghijklmnopqrstuvwxyz", system="system prompt")
    assert result.text == "Grounded explanation."
    assert result.prompt_eval_count == 13
    assert result.eval_count == 42
    assert len(calls) == 2
