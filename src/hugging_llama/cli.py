"""Command line interface for the Ollama compatible server."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Coroutine, Sequence
from typing import Any

import httpx
import uvicorn

from .server import create_app

LOGGER = logging.getLogger(__name__)


def _resolve_default_port() -> int:
    """Return the default serve port, honoring ``OLLAMA_LOCAL_PORT``."""

    env_value = os.environ.get("OLLAMA_LOCAL_PORT")
    if env_value is None:
        return 11434

    try:
        port = int(env_value)
    except ValueError:
        LOGGER.warning(
            "Invalid value for OLLAMA_LOCAL_PORT=%s; falling back to 11434.",
            env_value,
        )
        return 11434

    if not (0 < port < 65536):
        LOGGER.warning(
            "OLLAMA_LOCAL_PORT %s outside valid range 1-65535; falling back to 11434.",
            port,
        )
        return 11434

    return port


DEFAULT_PORT = _resolve_default_port()
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}"


STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=None, pool=None)


async def stream_pull(url: str, model: str, revision: str | None, trust_remote_code: bool) -> None:
    async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{url}/api/pull",
            json={"model": model, "revision": revision, "trust_remote_code": trust_remote_code},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                status = data.get("status")
                if status:
                    print(f"{status}")


async def command_ps(url: str) -> None:
    async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
        resp = await client.get(f"{url}/api/ps")
        resp.raise_for_status()
        data = resp.json()
        for model in data.get("models", []):
            expires = model.get("expires_at")
            ref = model.get("ref_count")
            print(f"{model['name']}: refs={ref} expires={expires}")


async def command_embed(url: str, model: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
        resp = await client.post(f"{url}/api/embed", json={"model": model, "input": text})
        resp.raise_for_status()
        data = resp.json()
        print(json.dumps(data, indent=2))


async def command_show(url: str, model: str) -> None:
    async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
        resp = await client.get(f"{url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
    models = {item["name"]: item for item in data.get("models", [])}
    info = models.get(model)
    if not info:
        print(f"Model {model} not found in local registry", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(info, indent=2))


def _run_http_command(coro: Coroutine[Any, Any, Any], url: str) -> int:
    try:
        asyncio.run(coro)
        return 0
    except httpx.ConnectError:
        print(
            "Failed to connect to the hugging-llama server at"
            f" {url}. Is it running? You can start it with 'hugging-llama serve'.",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPStatusError as exc:
        response = exc.response
        request = exc.request
        detail = response.text.strip()
        if detail:
            message = f": {detail}"
        else:
            message = ""
        print(
            "Request to"
            f" {request.method} {request.url} failed with status {response.status_code}{message}",
            file=sys.stderr,
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama compatible local runtime")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            "Server URL (for non-serve commands). Defaults to http://127.0.0.1 with "
            "the current port or the value of OLLAMA_LOCAL_PORT."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for the API server (overrides OLLAMA_LOCAL_PORT).",
    )
    serve_parser.add_argument("--max-resident-models", type=int, default=2)
    serve_parser.add_argument("--model-ttl", type=float, default=None)

    pull_parser = sub.add_parser("pull", help="Download a model from Hugging Face")
    pull_parser.add_argument("model")
    pull_parser.add_argument("--revision", default=None)
    pull_parser.add_argument("--trust-remote-code", action="store_true")

    sub.add_parser("ps", help="List loaded models")

    embed_parser = sub.add_parser("embed", help="Request embeddings")
    embed_parser.add_argument("model")
    embed_parser.add_argument("text")

    show_parser = sub.add_parser("show", help="Show effective configuration for a model")
    show_parser.add_argument("model")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        app = create_app(max_resident_models=args.max_resident_models, default_ttl=args.model_ttl)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    url = args.url.rstrip("/")
    if args.command == "pull":
        return _run_http_command(stream_pull(url, args.model, args.revision, args.trust_remote_code), url)
    if args.command == "ps":
        return _run_http_command(command_ps(url), url)
    if args.command == "embed":
        return _run_http_command(command_embed(url, args.model, args.text), url)
    if args.command == "show":
        return _run_http_command(command_show(url, args.model), url)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
