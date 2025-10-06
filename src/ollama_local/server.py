"""FastAPI application implementing Ollama compatible endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from transformers import TextIteratorStreamer

from .api_types import ChatRequest, EmbeddingsRequest, GenerateOptions, GenerateRequest, PullRequest
from .model_manager import ManagedModel, ModelManager, run_generation
from .stop_sequences import StopSequenceMatcher
from .utils import TimingInfo, detect_platform, now_utc, parse_keep_alive

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


def create_app(
    cache_dir: Optional[Path] = None,
    max_resident_models: int = 2,
    default_ttl: Optional[float] = None,
) -> FastAPI:
    cache_path = cache_dir or Path(os.environ.get("OLLAMA_SERVER_CACHE", "~/.cache/ollama-local")).expanduser()
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
    async def metrics_middleware(request: Request, call_next):  # type: ignore[override]
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

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "device": detect_platform(),
        }

    @app.get("/metrics")
    async def metrics() -> StreamingResponse:
        return StreamingResponse(generate_latest(), media_type="text/plain; version=0.0.4")

    async def _stream_generate(
        request: GenerateRequest,
        manager: ModelManager,
        ttl: Optional[float],
        format_validator,
    ) -> AsyncGenerator[bytes, None]:
        if request.options is None:
            request_options = GenerateOptions()
        else:
            request_options = request.options
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
        generation_result: Dict[str, Any] = {}
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
    async def generate_endpoint(request: GenerateRequest, manager: ModelManager = Depends(get_manager)):
        ttl = parse_keep_alive(request.keep_alive)
        format_validator = build_format_validator(request.format)
        if not request.stream:
            chunks = []
            async for payload in _stream_generate(request, manager, ttl, format_validator):
                data = json.loads(payload.decode("utf-8"))
                chunks.append(data)
            final = chunks[-1] if chunks else {"response": "", "done": True}
            return JSONResponse(final)
        return StreamingResponse(_stream_generate(request, manager, ttl, format_validator), media_type="application/json")

    @app.post("/api/chat")
    async def chat_endpoint(request: ChatRequest, manager: ModelManager = Depends(get_manager)):
        ttl = parse_keep_alive(request.keep_alive)
        format_validator = build_format_validator(request.format)

        if request.options is None:
            request_options = GenerateOptions()
        else:
            request_options = request.options
        model = await manager.ensure_model(request.model, request_options, ttl)
        tokenizer = model.tokenizer
        messages = []
        for message in request.messages:
            entry: Dict[str, Any] = {"role": message.role, "content": message.content}
            if message.name:
                entry["name"] = message.name
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            messages.append(entry)
        template_kwargs: Dict[str, Any] = {}
        if request.tools:
            template_kwargs["tools"] = [tool.dict() for tool in request.tools]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **template_kwargs)
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
        async def translate_stream():
            collected = ""
            stored_tool_calls: Optional[List[Dict[str, Any]]] = None
            async for chunk in _stream_generate(generate_request, manager, ttl, format_validator):
                payload = json.loads(chunk.decode("utf-8"))
                text = payload.get("response", "")
                if text:
                    collected += text
                message_content: Dict[str, Any] = {
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
                    message_chunk.update({k: v for k, v in payload.items() if k in {"total_duration", "eval_count", "prompt_eval_count"}})
                yield (json.dumps(message_chunk) + "\n").encode("utf-8")
        if not request.stream:
            data = []
            async for chunk in translate_stream():
                data.append(json.loads(chunk.decode("utf-8")))
            return JSONResponse(data[-1] if data else {"message": {"role": "assistant", "content": ""}, "done": True})
        return StreamingResponse(translate_stream(), media_type="application/json")

    @app.get("/api/tags")
    async def list_tags() -> Dict[str, Any]:
        models = []
        for path in sorted(manager.cache_dir.glob("**/config.json")):
            stat = path.stat()
            model_dir = path.parent
            size = sum(p.stat().st_size for p in model_dir.rglob("*"))
            models.append(
                {
                    "name": model_dir.name.replace("__", "/"),
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
        return {"models": models}

    @app.get("/api/ps")
    async def list_processes(manager: ModelManager = Depends(get_manager)):
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
    async def pull_endpoint(request: PullRequest):
        async def event_stream():
            yield (json.dumps({"status": "pulling manifest", "digest": ""}) + "\n").encode("utf-8")
            path = await manager.pull(request.model, request.revision, request.trust_remote_code)
            yield (json.dumps({"status": "downloading", "path": str(path)}) + "\n").encode("utf-8")
            yield (json.dumps({"status": "success"}) + "\n").encode("utf-8")
        return StreamingResponse(event_stream(), media_type="application/json")

    @app.get("/api/models")
    async def deprecated_models():
        tags = await list_tags()
        return tags

    @app.post("/api/embed")
    async def embed(request: EmbeddingsRequest, manager: ModelManager = Depends(get_manager)):
        ttl = parse_keep_alive(request.keep_alive)
        model = await manager.ensure_embeddings_model(request.model, ttl)
        inputs = request.input if isinstance(request.input, list) else [request.input]
        start = time.perf_counter()
        vectors = await asyncio.get_running_loop().run_in_executor(None, model.model.encode, inputs)
        duration = time.perf_counter() - start
        await manager.release(request.model, ttl)
        return {
            "model": request.model,
            "embeddings": [vec.tolist() for vec in vectors],
            "total_duration": duration,
        }

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


def format_generate_chunk(request: GenerateRequest, text: str, done: bool, timing: TimingInfo) -> Dict[str, Any]:
    payload = {
        "model": request.model,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "response": text,
        "done": done,
    }
    if done:
        payload.update(timing.as_dict())
    return payload


def build_format_validator(format_option: Optional[Any]):
    if format_option is None:
        return None
    if isinstance(format_option, str) and format_option.lower() == "json":
        def validator(text: str) -> None:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Model output is not valid JSON: {exc}")

        return validator
    if isinstance(format_option, dict):
        import jsonschema

        schema = format_option

        def validator(text: str) -> None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Model output is not valid JSON: {exc}")
            jsonschema.validate(data, schema)

        return validator
    raise HTTPException(status_code=400, detail="Unsupported format option")


def detect_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
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
