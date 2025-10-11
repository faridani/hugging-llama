"""Tests for the CLI helpers."""
from __future__ import annotations

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


def test_invalid_port_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_LOCAL_PORT", "not-a-number")
    cli = reload_cli_module()

    parser = cli.build_parser()
    args = parser.parse_args(["serve"])

    assert args.port == 11434
    assert args.url == "http://127.0.0.1:11434"

    monkeypatch.delenv("OLLAMA_LOCAL_PORT", raising=False)
    reload_cli_module()
