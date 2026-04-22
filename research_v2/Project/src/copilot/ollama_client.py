"""Small Ollama client for the dashboard copilot foundation."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import CopilotSettings


Transport = Callable[[Request, float], Any]


class OllamaError(RuntimeError):
    """Raised when a local Ollama request fails."""


@dataclass(frozen=True)
class OllamaGenerateResult:
    model: str
    text: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    model_available: bool
    selected_model: str
    installed_models: tuple[str, ...]
    message: str


def _default_transport(request: Request, timeout: float) -> Any:
    return urlopen(request, timeout=timeout)


class OllamaClient:
    """Minimal Ollama API wrapper with test-friendly transport injection."""

    def __init__(self, settings: CopilotSettings, transport: Transport | None = None) -> None:
        self.settings = settings
        self._transport = transport or _default_transport

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url=f"{self.settings.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with self._transport(request, timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid JSON: {raw[:200]}") from exc

    def list_models(self) -> tuple[str, ...]:
        payload = self._json_request("GET", "/api/tags", timeout=self.settings.health_timeout_sec)
        models = payload.get("models", [])
        names = []
        for item in models:
            name = item.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(names)

    def get_status(self, model: str | None = None) -> OllamaStatus:
        selected_model = model or self.settings.default_model
        try:
            installed = self.list_models()
        except OllamaError as exc:
            return OllamaStatus(
                available=False,
                model_available=False,
                selected_model=selected_model,
                installed_models=(),
                message=f"Local Ollama unavailable: {exc}",
            )

        model_available = selected_model in installed
        if not model_available:
            return OllamaStatus(
                available=True,
                model_available=False,
                selected_model=selected_model,
                installed_models=installed,
                message=f"Local Ollama is reachable, but model '{selected_model}' is not installed.",
            )
        return OllamaStatus(
            available=True,
            model_available=True,
            selected_model=selected_model,
            installed_models=installed,
            message=f"Copilot ready with local model '{selected_model}'.",
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> OllamaGenerateResult:
        selected_model = model or self.settings.default_model
        status = self.get_status(selected_model)
        if not status.available:
            raise OllamaError(status.message)
        if not status.model_available:
            raise OllamaError(status.message)

        prompt = prompt[: self.settings.prompt_budget_chars]
        payload: dict[str, Any] = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system[: self.settings.prompt_budget_chars]

        response = self._json_request(
            "POST",
            "/api/generate",
            payload,
            timeout=max(self.settings.read_timeout_sec, self.settings.connect_timeout_sec),
        )
        text = response.get("response", "")
        if not isinstance(text, str):
            raise OllamaError("Ollama generate response did not contain text.")
        return OllamaGenerateResult(
            model=selected_model,
            text=text.strip(),
            prompt_eval_count=response.get("prompt_eval_count"),
            eval_count=response.get("eval_count"),
            done_reason=response.get("done_reason"),
        )
