"""Model manager with LRU + TTL semantics."""
from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import infer_auto_device_map
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from .api_types import GenerateOptions
from .logits import build_logits_processors
from .stop_sequences import StopSequenceMatcher
from .utils import AsyncLRU, choose_dtype, detect_default_device, parse_keep_alive

MODEL_LOCKS: Dict[str, asyncio.Lock] = {}


@dataclass
class ManagedModel:
    model: Any
    tokenizer: Any
    kind: str
    path: Path
    device: str
    dtype: torch.dtype
    details: Dict[str, Any] = field(default_factory=dict)
    keep_alive: Optional[float] = None

    def close(self) -> None:
        if hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception:
                pass


class ModelManager:
    def __init__(
        self,
        cache_dir: Path,
        max_resident_models: int = 2,
        default_ttl: Optional[float] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = AsyncLRU(max_items=max_resident_models)
        self.default_ttl = default_ttl
        self.aliases: Dict[str, Dict[str, Any]] = {}

    def resolve_model(self, name: str) -> Tuple[str, Dict[str, Any]]:
        info = self.aliases.get(name, {"model": name, "options": {}})
        model_name = info.get("model", name)
        options = info.get("options", {})
        return model_name, options

    async def ensure_model(self, name: str, options: Optional[GenerateOptions], ttl: Optional[float]) -> ManagedModel:
        resolved_name, alias_options = self.resolve_model(name)
        merged_options = {}
        merged_options.update(alias_options or {})
        if options:
            merged_options.update(options.model_dump(exclude_none=True))
        entry = await self.models.get(resolved_name)
        if entry is None:
            lock = MODEL_LOCKS.setdefault(resolved_name, asyncio.Lock())
            async with lock:
                entry = await self.models.get(resolved_name)
                if entry is None:
                    model = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: self._load_text_model(resolved_name, merged_options),
                    )
                    entry = await self.models.upsert(resolved_name, model, ttl or self.default_ttl)
        if ttl is not None:
            await self.models.update_ttl(resolved_name, ttl)
        await self.models.increment(resolved_name)
        return entry.value

    def _load_text_model(self, name: str, options: Dict[str, Any]) -> ManagedModel:
        load_kwargs: Dict[str, Any] = {}
        device = detect_default_device()
        dtype = choose_dtype(device)
        if options.get("load_in_8bit"):
            load_kwargs["load_in_8bit"] = True
        elif options.get("load_in_4bit"):
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["torch_dtype"] = dtype
        load_kwargs["device_map"] = "auto"
        if options.get("trust_remote_code"):
            load_kwargs["trust_remote_code"] = True
        repo_dir = self.cache_dir / name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(repo_dir if any(repo_dir.iterdir()) else name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(repo_dir if any(repo_dir.iterdir()) else name, use_fast=True)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        managed = ManagedModel(
            model=model,
            tokenizer=tokenizer,
            kind="generate",
            path=repo_dir,
            device=device,
            dtype=dtype,
            details={
                "family": model.config.model_type,
                "parameter_size": getattr(model.config, "num_parameters", None),
                "format": "safetensors",
            },
        )
        return managed

    async def release(self, name: str, ttl: Optional[float]) -> None:
        resolved_name, _ = self.resolve_model(name)
        await self.models.decrement(resolved_name)
        if ttl == 0:
            await self.models.update_ttl(resolved_name, 0)
            await self.models.evict_expired()

    async def list_loaded(self) -> Dict[str, Dict[str, Any]]:
        snapshot = await self.models.snapshot()
        result: Dict[str, Dict[str, Any]] = {}
        for key, entry in snapshot.items():
            model: ManagedModel = entry.value
            result[key] = {
                "ref_count": entry.ref_count,
                "expires_at": entry.expires_at,
                "details": model.details,
            }
        return result

    async def unload(self, name: str) -> None:
        resolved_name, _ = self.resolve_model(name)
        async with self.models._global_lock:  # noqa: SLF001
            await self.models._evict_key_locked(resolved_name)

    async def pull(self, name: str, revision: Optional[str], trust_remote_code: bool) -> Path:
        repo_dir = self.cache_dir / name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            name,
            revision=revision,
            local_dir=repo_dir,
            local_dir_use_symlinks=False,
            trust_remote_code=trust_remote_code,
        )
        return repo_dir

    async def ensure_embeddings_model(self, name: str, ttl: Optional[float]) -> ManagedModel:
        entry = await self.models.get(name)
        if entry is None:
            lock = MODEL_LOCKS.setdefault(name, asyncio.Lock())
            async with lock:
                entry = await self.models.get(name)
                if entry is None:
                    managed = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: self._load_embeddings_model(name)
                    )
                    entry = await self.models.upsert(name, managed, ttl or self.default_ttl)
        if ttl is not None:
            await self.models.update_ttl(name, ttl)
        await self.models.increment(name)
        return entry.value

    def _load_embeddings_model(self, name: str) -> ManagedModel:
        from sentence_transformers import SentenceTransformer

        repo_dir = self.cache_dir / name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(str(repo_dir if any(repo_dir.iterdir()) else name))
        managed = ManagedModel(
            model=model,
            tokenizer=None,
            kind="embedding",
            path=repo_dir,
            device=detect_default_device(),
            dtype=torch.float32,
            details={"format": "safetensors", "family": "sentence-transformers"},
        )
        return managed

    async def evict_expired(self) -> None:
        await self.models.evict_expired()


async def run_generation(
    manager: ModelManager,
    request_options: GenerateOptions,
    input_ids: torch.LongTensor,
    tokenizer,
    model,
    prompt_text: str,
    streamer: TextIteratorStreamer,
) -> Dict[str, Any]:
    prompt_tokens = len(tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])
    generation_kwargs: Dict[str, Any] = {
        "input_ids": input_ids.to(model.device),
        "streamer": streamer,
    }
    if request_options.max_tokens is not None:
        generation_kwargs["max_new_tokens"] = request_options.max_tokens
    if request_options.top_k is not None:
        generation_kwargs["top_k"] = request_options.top_k
    if request_options.top_p is not None:
        generation_kwargs["top_p"] = request_options.top_p
    if request_options.temperature is not None:
        generation_kwargs["temperature"] = request_options.temperature
    if request_options.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = request_options.repetition_penalty
    logits_processors = build_logits_processors(
        prompt_lengths=[prompt_tokens],
        presence_penalty=request_options.presence_penalty or 0.0,
        frequency_penalty=request_options.frequency_penalty or 0.0,
    )
    if logits_processors:
        generation_kwargs["logits_processor"] = logits_processors
    if request_options.seed is not None:
        torch.manual_seed(request_options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request_options.seed)
    model.generate(**generation_kwargs)
    return {"prompt_tokens": prompt_tokens}
