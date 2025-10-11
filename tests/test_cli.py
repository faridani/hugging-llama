"""Tests for the CLI helpers."""
from __future__ import annotations

import asyncio
import importlib

import pytest


def reload_cli_module():
    """Reload ``hugging_llama.cli`` after environment tweaks."""

    import hugging_llama.cli as cli

    return importlib.reload(cli)


def test_default_port_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_LOCAL_PORT", "23000")
    cli = reload_cli_module()

    parser = cli.build_parser()
    args = parser.parse_args(["serve"])

    assert args.port == 23000
    assert args.url == "http://127.0.0.1:23000"

    monkeypatch.delenv("OLLAMA_LOCAL_PORT", raising=False)
    reload_cli_module()


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - no failure path in tests
        return None

    def json(self) -> dict:
        return self._payload


class DummyAsyncClient:
    def __init__(self, response: DummyResponse):
        self._response = response
        self.requested_urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str):
        self.requested_urls.append(url)
        return self._response


def test_command_ps_renders_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import hugging_llama.cli as cli

    payload = {
        "models": [
            {
                "name": "alpha",
                "ref_count": 2,
                "expires_in": 12.4,
                "expires_at": "2024-01-01T00:00:00Z",
            },
            {
                "name": "bravo",
                "ref_count": 0,
                "expires_in": None,
                "expires_at": None,
            },
        ]
    }

    response = DummyResponse(payload)

    def _client_factory(*args, **kwargs):
        return DummyAsyncClient(response)

    monkeypatch.setattr(cli.httpx, "AsyncClient", _client_factory)

    asyncio.run(cli.command_ps("http://localhost:11434"))

    out = capsys.readouterr().out
    assert "NAME" in out
    assert "alpha" in out
    assert "bravo" in out
    assert "∞" in out


def test_command_show_formats_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import hugging_llama.cli as cli

    payload = {
        "models": [
            {
                "name": "repo/model",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 2048,
                "digest": "deadbeef",
                "details": {"format": "safetensors", "family": "test"},
            }
        ]
    }

    response = DummyResponse(payload)

    def _client_factory(*args, **kwargs):
        return DummyAsyncClient(response)

    monkeypatch.setattr(cli.httpx, "AsyncClient", _client_factory)

    asyncio.run(cli.command_show("http://localhost:11434", "repo/model:latest"))

    out = capsys.readouterr().out
    assert "Name: repo/model" in out
    assert "Size: 2.00 KB" in out
    assert "format: safetensors" in out


def test_invalid_port_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_LOCAL_PORT", "not-a-number")
    cli = reload_cli_module()

    parser = cli.build_parser()
    args = parser.parse_args(["serve"])

    assert args.port == 11434
    assert args.url == "http://127.0.0.1:11434"

    monkeypatch.delenv("OLLAMA_LOCAL_PORT", raising=False)
    reload_cli_module()
