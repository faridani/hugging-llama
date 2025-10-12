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
