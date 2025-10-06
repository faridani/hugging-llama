"""Command line interface for the Ollama compatible server."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
import uvicorn

from .server import create_app

DEFAULT_URL = "http://127.0.0.1:11434"


async def stream_pull(url: str, model: str, revision: Optional[str], trust_remote_code: bool) -> None:
    async with httpx.AsyncClient(timeout=None) as client:
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
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/api/ps")
        resp.raise_for_status()
        data = resp.json()
        for model in data.get("models", []):
            expires = model.get("expires_at")
            ref = model.get("ref_count")
            print(f"{model['name']}: refs={ref} expires={expires}")


async def command_embed(url: str, model: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{url}/api/embed", json={"model": model, "input": text})
        resp.raise_for_status()
        data = resp.json()
        print(json.dumps(data, indent=2))


async def command_show(url: str, model: str) -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
    models = {item["name"]: item for item in data.get("models", [])}
    info = models.get(model)
    if not info:
        print(f"Model {model} not found in local registry", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama compatible local runtime")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server URL (for non-serve commands)")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=11434)
    serve_parser.add_argument("--max-resident-models", type=int, default=2)
    serve_parser.add_argument("--model-ttl", type=float, default=None)

    pull_parser = sub.add_parser("pull", help="Download a model from Hugging Face")
    pull_parser.add_argument("model")
    pull_parser.add_argument("--revision", default=None)
    pull_parser.add_argument("--trust-remote-code", action="store_true")

    ps_parser = sub.add_parser("ps", help="List loaded models")

    embed_parser = sub.add_parser("embed", help="Request embeddings")
    embed_parser.add_argument("model")
    embed_parser.add_argument("text")

    show_parser = sub.add_parser("show", help="Show effective configuration for a model")
    show_parser.add_argument("model")

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        app = create_app(max_resident_models=args.max_resident_models, default_ttl=args.model_ttl)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    url = args.url.rstrip("/")
    if args.command == "pull":
        asyncio.run(stream_pull(url, args.model, args.revision, args.trust_remote_code))
        return 0
    if args.command == "ps":
        asyncio.run(command_ps(url))
        return 0
    if args.command == "embed":
        asyncio.run(command_embed(url, args.model, args.text))
        return 0
    if args.command == "show":
        asyncio.run(command_show(url, args.model))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
