"""Tests for the CLI helpers."""
from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

import httpx
import pytest


def reload_cli_module():
    """Reload ``hugging_llama.cli`` after environment tweaks."""

    import hugging_llama.cli as cli

    return importlib.reload(cli)


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubAsyncClient:
    def __init__(self, response: _StubResponse) -> None:
        self._response = response
        self.requested_urls: list[str] = []

    async def __aenter__(self) -> _StubAsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def get(self, url: str) -> _StubResponse:
        self.requested_urls.append(url)
        return self._response


def test_default_port_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_LOCAL_PORT", "23000")
    cli = reload_cli_module()

    parser = cli.build_parser()
    args = parser.parse_args(["serve"])

    assert args.port == 23000
    assert args.url == "http://127.0.0.1:23000"

    monkeypatch.delenv("OLLAMA_LOCAL_PORT", raising=False)
    reload_cli_module()


def test_invalid_port_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_LOCAL_PORT", "not-a-number")
    cli = reload_cli_module()

    parser = cli.build_parser()
    args = parser.parse_args(["serve"])

    assert args.port == 11434
    assert args.url == "http://127.0.0.1:11434"

    monkeypatch.delenv("OLLAMA_LOCAL_PORT", raising=False)
    reload_cli_module()


def test_catalog_parser_arguments() -> None:
    cli = reload_cli_module()

    parser = cli.build_parser()
    args = parser.parse_args(["catalog", "--memory", "24GB"])

    assert args.command == "catalog"
    assert args.memory == "24GB"
    assert not args.all


def test_catalog_filters_by_memory(capsys: pytest.CaptureFixture[str]) -> None:
    cli = reload_cli_module()

    exit_code = cli.command_catalog("24GB", False)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "meta-llama/Meta-Llama-3-8B-Instruct" in captured.out
    assert "tiiuae/falcon-40b-instruct" not in captured.out


def test_catalog_invalid_memory_spec(capsys: pytest.CaptureFixture[str]) -> None:
    cli = reload_cli_module()

    exit_code = cli.command_catalog("twenty", False)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Invalid memory specification" in captured.err


def test_catalog_uses_detected_vram(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cli = reload_cli_module()
    monkeypatch.setattr(cli, "_detect_gpu_memory_bytes", lambda: 12 * 1024**3)

    exit_code = cli.command_catalog(None, False)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Detected approximately 12.0 GB" in captured.out
    assert "mistralai/Mistral-7B-Instruct-v0.2" not in captured.out


def test_serve_command_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = reload_cli_module()

    captured: dict[str, object] = {}

    def fake_run(app: object, host: str, port: int) -> None:
        captured.update({"app": app, "host": host, "port": port})

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr("hugging_llama.server.create_app", lambda **kwargs: "APP")

    exit_code = cli.main([
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "12345",
        "--max-resident-models",
        "3",
        "--model-ttl",
        "10",
    ])

    assert exit_code == 0
    assert captured == {"app": "APP", "host": "127.0.0.1", "port": 12345}


def test_parse_memory_spec_supports_megabytes() -> None:
    cli = reload_cli_module()

    result = cli._parse_memory_spec("512MB")

    assert result == pytest.approx(0.5)


def test_parse_memory_spec_rejects_negative_values() -> None:
    cli = reload_cli_module()

    with pytest.raises(ValueError, match="positive"):
        cli._parse_memory_spec("-1GB")


def test_run_http_command_handles_connect_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = reload_cli_module()
    request = httpx.Request("GET", "http://example")

    def fake_run(_: Any) -> None:
        raise httpx.ConnectError("boom", request=request)

    monkeypatch.setattr(cli.asyncio, "run", fake_run)

    exit_code = cli._run_http_command(object(), "http://example")
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Failed to connect" in captured.err


def test_run_http_command_handles_http_status_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = reload_cli_module()
    request = httpx.Request("POST", "http://example/api")
    response = httpx.Response(404, request=request, text="not found")

    def fake_run(_: Any) -> None:
        raise httpx.HTTPStatusError("error", request=request, response=response)

    monkeypatch.setattr(cli.asyncio, "run", fake_run)

    exit_code = cli._run_http_command(object(), "http://example")
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "status 404" in captured.err
    assert "not found" in captured.err


def test_command_show_outputs_matching_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = reload_cli_module()
    payload = {"models": [{"name": "alpha", "details": {"size": 1}}]}
    client = _StubAsyncClient(_StubResponse(payload))

    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda *args, **kwargs: client)

    asyncio.run(cli.command_show("http://localhost", "alpha"))
    captured = capsys.readouterr()

    assert client.requested_urls == ["http://localhost/api/tags"]
    assert json.loads(captured.out)["name"] == "alpha"


def test_command_show_exits_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = reload_cli_module()
    payload = {"models": [{"name": "beta", "details": {}}]}
    client = _StubAsyncClient(_StubResponse(payload))

    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda *args, **kwargs: client)

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(cli.command_show("http://localhost", "alpha"))

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Model alpha not found" in captured.err


def test_command_catalog_reports_no_matches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = reload_cli_module()
    oversized = cli.CatalogEntry(
        name="huge-model",
        parameters="1T",
        size_gb=256.0,
        precision="FP16",
        description="Too large for small GPUs",
    )
    monkeypatch.setattr(cli, "load_model_catalog", lambda: [oversized])

    exit_code = cli.command_catalog("1GB", False)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No models match the requested memory constraints." in captured.out
