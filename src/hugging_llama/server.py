"""FastAPI application implementing Ollama compatible endpoints."""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import math
import os
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, cast

from importlib.metadata import PackageNotFoundError, version

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from transformers import TextIteratorStreamer

from .api_types import (
    ChatMessage,
    ChatRequest,
    CopyRequest,
    CreateRequest,
    DeleteRequest,
    EmbeddingsRequest,
    GenerateOptions,
    GenerateRequest,
    PullRequest,
    PushRequest,
    ShowRequest,
    UnloadRequest,
)
from .model_manager import ModelManager, run_generation
from .stop_sequences import StopSequenceMatcher
from .utils import TimingInfo, detect_platform, parse_keep_alive

LOGGER = logging.getLogger(__name__)

REQUEST_LATENCY = Histogram(
    "ollama_request_latency_seconds",
    "Request latencies by endpoint",
    labelnames=("endpoint",),
)
TOKENS_GENERATED = Counter(
    "ollama_tokens_generated_total", "Total tokens generated", labelnames=("endpoint",)
)
TOKENS_RATE = Gauge("ollama_tokens_per_second", "Tokens per second", labelnames=("endpoint",))
OOM_COUNTER = Counter("ollama_oom_total", "Number of OOM occurrences")

try:
    PACKAGE_VERSION = version("hugging-llama")
except PackageNotFoundError:
    PACKAGE_VERSION = "0.0.0"


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _consume_block(initial: str, iterator: Iterator[str], delimiter: str) -> str:
    parts: list[str] = []
    remainder = initial
    if remainder:
        parts.append(remainder)
    for line in iterator:
        if delimiter in line:
            idx = line.find(delimiter)
            parts.append(line[:idx])
            trailing = line[idx + len(delimiter) :]
            if trailing:
                parts.append(trailing)
            break
        parts.append(line)
    return "\n".join(parts).strip("\n")


def _parse_block_value(value: str, iterator: Iterator[str]) -> str:
    stripped = value.lstrip()
    if stripped.startswith('"""') or stripped.startswith("'''"):
        delimiter = stripped[:3]
        remainder = stripped[3:]
        if remainder.endswith(delimiter):
            return remainder[: -len(delimiter)]
        return _consume_block(remainder, iterator, delimiter)
    return _strip_wrapping_quotes(stripped.strip())


def parse_modelfile(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"model": None, "template": None, "system": None, "parameters": {}}
    if not text:
        return result
    iterator = iter(text.splitlines())
    for raw_line in iterator:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        keyword = parts[0].upper()
        remainder = parts[1] if len(parts) > 1 else ""
        if keyword == "FROM":
            result["model"] = remainder.strip()
        elif keyword == "TEMPLATE":
            result["template"] = _parse_block_value(remainder, iterator)
        elif keyword == "SYSTEM":
            result["system"] = _parse_block_value(remainder, iterator)
        elif keyword == "PARAMETER":
            param_parts = remainder.split(maxsplit=1)
            if len(param_parts) == 2:
                key = param_parts[0]
                result["parameters"][key] = _strip_wrapping_quotes(param_parts[1].strip())
    return result


def create_app(
    cache_dir: Path | None = None,
    max_resident_models: int = 2,
    default_ttl: float | None = None,
) -> FastAPI:
    cache_path = cache_dir or Path(os.environ.get("OLLAMA_SERVER_CACHE", "~/.cache/hugging-llama")).expanduser()
    manager = ModelManager(cache_path, max_resident_models=max_resident_models, default_ttl=default_ttl)
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        endpoint = request.url.path
        try:
            response = await call_next(request)
        finally:
            duration = time.perf_counter() - start
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        return response

    def get_manager() -> ModelManager:
        return manager

    @app.get("/")
    async def root() -> Response:
        return Response(content="Ollama is running", media_type="text/plain")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device": detect_platform(),
        }

    @app.get("/api/version")
    async def version_endpoint() -> dict[str, Any]:
        return {"version": PACKAGE_VERSION}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")

    async def _stream_generate(
        request: GenerateRequest,
        manager: ModelManager,
        ttl: float | None,
        format_validator: Callable[[str], None] | None,
    ) -> AsyncGenerator[bytes, None]:
        alias_info = manager.get_alias(request.model)
        if alias_info:
            if request.system is None and alias_info.get("system"):
                request.system = cast(str, alias_info["system"])
            if request.template is None and alias_info.get("template"):
                request.template = cast(str, alias_info["template"])
        alias_options = alias_info.get("options", {}) if alias_info else {}
        if request.options is None:
            request_options = GenerateOptions(**alias_options)
        else:
            combined_options = dict(alias_options)
            combined_options.update(request.options.model_dump(exclude_none=True))
            request_options = GenerateOptions(**combined_options)
        model = await manager.ensure_model(request.model, request_options, ttl)
        tokenizer = model.tokenizer
        if model.kind != "generate":
            await manager.release(request.model, ttl)
            raise HTTPException(status_code=400, detail="Model does not support text generation")
        prompt = build_prompt(request)
        if request.prompt == "" and not request.raw:
            await manager.release(request.model, ttl)
            yield json.dumps({"model": request.model, "done": True}).encode("utf-8")
            return
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.model.device)
        stop_handler = StopSequenceMatcher(request_options.stop)
        timing = TimingInfo(start_time=time.perf_counter())
        loop = asyncio.get_running_loop()
        generation_task = loop.run_in_executor(
            None,
            partial(
                run_generation,
                manager,
                request_options,
                input_ids,
                tokenizer,
                model.model,
                prompt,
                streamer,
            ),
        )
        accumulated = ""
        generation_result: dict[str, Any] = {}
        async def _pull_next() -> str:
            def _next_token() -> str:
                try:
                    return next(streamer)
                except StopIteration as exc:  # pragma: no cover - handled as StopAsyncIteration
                    raise StopAsyncIteration from exc

            return await asyncio.to_thread(_next_token)

        try:
            while True:
                try:
                    chunk = await _pull_next()
                except StopAsyncIteration:
                    break
                emit, stopped = stop_handler.push(chunk)
                if not emit and not stopped:
                    continue
                accumulated += emit
                if emit:
                    tokens = tokenizer(emit, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                    timing.completion_tokens += tokens.size(0)
                    TOKENS_GENERATED.labels(endpoint="generate").inc(tokens.size(0))
                    chunk_payload = format_generate_chunk(request, emit, False, timing)
                    yield (json.dumps(chunk_payload) + "\n").encode("utf-8")
                if stopped:
                    break
            remainder = stop_handler.flush()
            if remainder:
                accumulated += remainder
                tokens = tokenizer(remainder, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                timing.completion_tokens += tokens.size(0)
                TOKENS_GENERATED.labels(endpoint="generate").inc(tokens.size(0))
                chunk_payload = format_generate_chunk(request, remainder, False, timing)
                yield (json.dumps(chunk_payload) + "\n").encode("utf-8")
            generation_result = await generation_task
            if isinstance(generation_result, dict) and "prompt_tokens" in generation_result:
                timing.prompt_tokens = generation_result["prompt_tokens"]
            final_payload = format_generate_chunk(request, "", True, timing)
            final_data = dict(final_payload)
            final_data.update(timing.as_dict())
            if format_validator is not None:
                format_validator(accumulated)
            final_data["response"] = accumulated
            final_data["done"] = True
            yield (json.dumps(final_data) + "\n").encode("utf-8")
        finally:
            await manager.release(request.model, ttl)

    @app.post("/api/generate")
    async def generate_endpoint(
        request: GenerateRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> Response:
        ttl = parse_keep_alive(request.keep_alive)
        format_validator = build_format_validator(request.format)
        if not request.stream:
            chunks = []
            async for payload in _stream_generate(request, manager, ttl, format_validator):
                data = json.loads(payload.decode("utf-8"))
                chunks.append(data)
            final = chunks[-1] if chunks else {"response": "", "done": True}
            return JSONResponse(final)
        return StreamingResponse(
            _stream_generate(request, manager, ttl, format_validator),
            media_type="application/json",
        )

    @app.post("/api/chat")
    async def chat_endpoint(
        request: ChatRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> Response:
        ttl = parse_keep_alive(request.keep_alive)
        format_validator = build_format_validator(request.format)
        alias_info = manager.get_alias(request.model)
        alias_options = alias_info.get("options", {}) if alias_info else {}
        if request.options is None:
            request_options = GenerateOptions(**alias_options)
        else:
            merged_options = dict(alias_options)
            merged_options.update(request.options.model_dump(exclude_none=True))
            request_options = GenerateOptions(**merged_options)
        message_objects = list(request.messages)
        if alias_info and alias_info.get("system"):
            has_system = any(msg.role == "system" for msg in message_objects)
            if not has_system:
                message_objects = [ChatMessage(role="system", content=alias_info["system"])] + message_objects
        model = await manager.ensure_model(request.model, request_options, ttl)
        tokenizer = model.tokenizer
        messages = []
        for message in message_objects:
            entry: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.name:
                entry["name"] = message.name
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            messages.append(entry)
        template_kwargs: dict[str, Any] = {}
        if request.tools:
            template_kwargs["tools"] = [tool.model_dump() for tool in request.tools]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        await manager.release(request.model, ttl)

        generate_request = GenerateRequest(
            model=request.model,
            prompt=prompt,
            raw=True,
            stream=request.stream,
            keep_alive=request.keep_alive,
            format=request.format,
            options=request.options,
        )
        async def translate_stream() -> AsyncGenerator[bytes, None]:
            collected = ""
            stored_tool_calls: list[dict[str, Any]] | None = None
            async for chunk in _stream_generate(generate_request, manager, ttl, format_validator):
                payload = json.loads(chunk.decode("utf-8"))
                text = payload.get("response", "")
                if text:
                    collected += text
                message_content: dict[str, Any] = {
                    "role": "assistant",
                    "content": text,
                }
                if stored_tool_calls is None:
                    tool_calls = detect_tool_calls(collected)
                    if tool_calls is not None:
                        stored_tool_calls = tool_calls
                if stored_tool_calls is not None:
                    message_content["content"] = ""
                    message_content["tool_calls"] = stored_tool_calls
                message_chunk = {
                    "model": request.model,
                    "message": message_content,
                    "done": payload.get("done", False),
                }
                if payload.get("done"):
                    for key in ("total_duration", "eval_count", "prompt_eval_count"):
                        if key in payload:
                            message_chunk[key] = payload[key]
                yield (json.dumps(message_chunk) + "\n").encode("utf-8")
        if not request.stream:
            data = []
            async for chunk in translate_stream():
                data.append(json.loads(chunk.decode("utf-8")))
            final_payload = data[-1] if data else {"message": {"role": "assistant", "content": ""}, "done": True}
            return JSONResponse(final_payload)
        return StreamingResponse(translate_stream(), media_type="application/json")

    @app.get("/api/tags")
    async def list_tags() -> dict[str, Any]:
        models = []
        for path in sorted(manager.cache_dir.glob("**/config.json")):
            stat = path.stat()
            model_dir = path.parent
            size = sum(p.stat().st_size for p in model_dir.rglob("*"))
            models.append(
                {
                    "name": model_dir.name.replace("__", "/"),
                    "model": model_dir.name.replace("__", "/"),
                    "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
                    "size": size,
                    "digest": str(stat.st_mtime_ns),
                    "details": {
                        "format": "safetensors",
                        "family": "unknown",
                        "parameter_size": None,
                    },
                }
            )
        existing = {entry["name"] for entry in models}
        for alias in manager.list_alias_records():
            if alias["name"] in existing:
                continue
            models.append(
                {
                    "name": alias["name"],
                    "model": alias.get("model", alias["name"]),
                    "modified_at": alias.get("modified_at"),
                    "size": alias.get("size", 0),
                    "digest": alias.get("digest", ""),
                    "details": alias.get("details", {}),
                }
            )
        return {"models": models}

    @app.get("/api/ps")
    async def list_processes(
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> dict[str, Any]:
        snapshot = await manager.list_loaded()
        models = []
        for name, info in snapshot.items():
            models.append(
                {
                    "name": name,
                    "expires_at": info.get("expires_at"),
                    "ref_count": info.get("ref_count"),
                    "details": info.get("details"),
                }
            )
        return {"models": models}

    @app.post("/api/pull")
    async def pull_endpoint(request: PullRequest) -> StreamingResponse:
        async def event_stream() -> AsyncGenerator[bytes, None]:
            yield (json.dumps({"status": "pulling manifest", "digest": ""}) + "\n").encode("utf-8")
            path = await manager.pull(request.model, request.revision, request.trust_remote_code)
            yield (json.dumps({"status": "downloading", "path": str(path)}) + "\n").encode("utf-8")
            yield (json.dumps({"status": "success"}) + "\n").encode("utf-8")
        if not request.stream:
            async for _ in event_stream():
                pass
            return JSONResponse({"status": "success"})
        return StreamingResponse(event_stream(), media_type="application/json")

    @app.get("/api/models")
    async def deprecated_models() -> dict[str, Any]:
        return await list_tags()

    @app.post("/api/embed")
    async def embed(
        request: EmbeddingsRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> dict[str, Any]:
        ttl = parse_keep_alive(request.keep_alive)
        model = await manager.ensure_embeddings_model(request.model, ttl)
        inputs = request.input if isinstance(request.input, list) else [request.input]
        start = time.perf_counter()
        vectors = await asyncio.get_running_loop().run_in_executor(
            None,
            model.model.encode,
            inputs,
        )
        duration = time.perf_counter() - start
        await manager.release(request.model, ttl)
        return {
            "model": request.model,
            "embeddings": [vec.tolist() for vec in vectors],
            "total_duration": duration,
        }

    @app.post("/api/create")
    async def create_model(
        request: CreateRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> Response:
        parsed_modelfile = parse_modelfile(request.modelfile or "") if request.modelfile else {"parameters": {}}
        base_model = request.from_ or parsed_modelfile.get("model") or request.model
        parameters = request.parameters or parsed_modelfile.get("parameters") or {}
        template = request.template or parsed_modelfile.get("template")
        system_prompt = request.system or parsed_modelfile.get("system")
        license_info = request.license
        if isinstance(license_info, str):
            license_data: list[str] | str | None = [license_info]
        else:
            license_data = license_info
        manager.create_alias(
            request.model,
            base_model,
            template,
            system_prompt,
            parameters,
            request.modelfile,
            license_data,
            request.messages,
            {
                "files": request.files,
                "adapters": request.adapters,
                "quantize": request.quantize,
            },
        )

        async def stream_events() -> AsyncGenerator[bytes, None]:
            yield (json.dumps({"status": "creating model", "model": request.model}) + "\n").encode("utf-8")
            yield (json.dumps({"status": "success"}) + "\n").encode("utf-8")

        if not request.stream:
            async for _ in stream_events():
                pass
            return JSONResponse({"status": "success"})
        return StreamingResponse(stream_events(), media_type="application/json")

    @app.post("/api/show")
    async def show_model(
        request: ShowRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> dict[str, Any]:
        info = manager.describe_model(request.model)
        if info is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return info

    @app.post("/api/copy")
    async def copy_model(
        request: CopyRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> dict[str, Any]:
        if not manager.copy_model(request.source, request.destination):
            raise HTTPException(status_code=404, detail="Source model not found")
        return {"status": "success"}

    @app.delete("/api/delete")
    async def delete_model(
        request: DeleteRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> dict[str, Any]:
        if not manager.delete_model(request.model):
            raise HTTPException(status_code=404, detail="Model not found")
        return {"status": "success"}

    @app.post("/api/push")
    async def push_model(
        request: PushRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> Response:
        del manager  # unused

        async def stream_events() -> AsyncGenerator[bytes, None]:
            yield (json.dumps({"status": "retrieving manifest"}) + "\n").encode("utf-8")
            yield (json.dumps({"status": "starting upload", "model": request.model}) + "\n").encode("utf-8")
            yield (json.dumps({"status": "success"}) + "\n").encode("utf-8")

        if not request.stream:
            async for _ in stream_events():
                pass
            return JSONResponse({"status": "success"})
        return StreamingResponse(stream_events(), media_type="application/json")

    @app.head("/api/blobs/{digest}")
    async def check_blob(
        digest: str,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> Response:
        if manager.blob_exists(digest):
            return Response(status_code=200)
        raise HTTPException(status_code=404, detail="Blob not found")

    @app.post("/api/blobs/{digest}")
    async def upload_blob(
        digest: str,
        request: Request,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> Response:
        data = await request.body()
        try:
            manager.save_blob(digest, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(status_code=201)

    @app.post("/api/unload")
    async def unload_model(
        request: UnloadRequest,
        manager: ModelManager = Depends(get_manager),  # noqa: B008
    ) -> dict[str, Any]:
        ttl = parse_keep_alive(request.keep_alive)
        if request.keep_alive is None or ttl == 0:
            await manager.unload(request.model)
        else:
            ttl_value = None if ttl is None or math.isinf(ttl) else ttl
            await manager.set_keep_alive(request.model, ttl_value)
        return {"status": "success"}

    return app


def build_prompt(request: GenerateRequest) -> str:
    if request.raw:
        return request.prompt
    if request.template:
        return (
            request.template.replace("{{prompt}}", request.prompt).replace("{{system}}", request.system or "")
        )
    if request.system:
        return f"{request.system}\n\n{request.prompt}"
    return request.prompt


def format_generate_chunk(request: GenerateRequest, text: str, done: bool, timing: TimingInfo) -> dict[str, Any]:
    payload = {
        "model": request.model,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "response": text,
        "done": done,
    }
    if done:
        payload.update(timing.as_dict())
    return payload


def build_format_validator(format_option: Any | None) -> Callable[[str], None] | None:
    if format_option is None:
        return None
    if isinstance(format_option, str) and format_option.lower() == "json":
        def validator(text: str) -> None:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Model output is not valid JSON: {exc}") from exc

        return validator
    if isinstance(format_option, dict):
        jsonschema_module = cast(Any, importlib.import_module("jsonschema"))
        validate = cast(Callable[[Any, Any], None], jsonschema_module.validate)
        schema = format_option

        def validator(text: str) -> None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Model output is not valid JSON: {exc}") from exc
            validate(data, schema)

        return validator
    raise HTTPException(status_code=400, detail="Unsupported format option")


def detect_tool_calls(text: str) -> list[dict[str, Any]] | None:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls
    return None


__all__ = ["create_app"]
