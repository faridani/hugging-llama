"""Model manager with LRU + TTL semantics."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import shutil
from importlib import metadata
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from .api_types import GenerateOptions
from .logits import build_logits_processors
from .metadata_utils import (
    DEFAULT_PROMPT_ALIASES,
    MetadataError,
    merge_metadata,
    normalize_metadata,
    validate_metadata,
)
from .modelfile import build_modelfile
from .utils import AsyncLRU, choose_dtype, detect_default_device

LOGGER = logging.getLogger(__name__)

MODEL_LOCKS: dict[str, asyncio.Lock] = {}


def _split_model_tag(name: str) -> tuple[str, str | None]:
    """Split an Ollama style ``name:tag`` identifier.

    Ollama exposes model references with optional ``:<tag>`` suffixes.  Hugging Face
    repositories use ``@revision`` instead, so we normalise names by splitting on the
    first colon.  The helper returns the base repository name and the optional tag.
    """

    base, sep, tag = name.partition(":")
    return (base if sep else name, tag if sep else None)


def _resolve_local_model_dir(repo_dir: Path) -> Path | None:
    """Return the directory containing ``config.json`` for a downloaded snapshot.

    Hugging Face's ``snapshot_download`` writes files under ``snapshots/<rev>`` by
    default, so we inspect both the repository root and any snapshot directories
    to locate the actual model payload.
    """

    direct_config = repo_dir / "config.json"
    if direct_config.exists():
        return repo_dir

    candidates: list[tuple[float, Path]] = []

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        for path in snapshots_dir.glob("*/config.json"):
            try:
                stat = path.stat()
            except OSError:
                continue
            candidates.append((stat.st_mtime, path.parent))

    if not candidates:
        for path in repo_dir.glob("**/config.json"):
            if path.parent == repo_dir:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            candidates.append((stat.st_mtime, path.parent))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _parse_version_prefix(value: str) -> list[int]:
    numbers: list[int] = []
    for part in value.replace("-", ".").split("."):
        if not part:
            continue
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        numbers.append(int(digits))
    return numbers


def _is_mxfp4_supported(device: str) -> bool:
    if device != "cuda":
        return False
    try:
        version_str = metadata.version("triton")
    except metadata.PackageNotFoundError:
        return False
    parsed = _parse_version_prefix(version_str)
    requirement = [3, 4, 0]
    length = max(len(parsed), len(requirement))
    parsed.extend(0 for _ in range(length - len(parsed)))
    requirement.extend(0 for _ in range(length - len(requirement)))
    return tuple(parsed) >= tuple(requirement)


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message

_SNAPSHOT_DOWNLOAD_SUPPORTS_TRUST_REMOTE_CODE = (
    "trust_remote_code" in inspect.signature(snapshot_download).parameters
)


@dataclass
class ManagedModel:
    model: Any
    tokenizer: Any
    kind: str
    path: Path
    device: str
    dtype: torch.dtype
    details: dict[str, Any] = field(default_factory=dict)
    keep_alive: float | None = None

    def close(self) -> None:
        if hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception as exc:  # pragma: no cover - defensive safeguard
                LOGGER.debug("Failed to move model to CPU during close: %s", exc)


class ModelManager:
    def __init__(
        self,
        cache_dir: Path,
        max_resident_models: int = 2,
        default_ttl: float | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = AsyncLRU(max_items=max_resident_models)
        self.default_ttl = default_ttl
        self.aliases: dict[str, dict[str, Any]] = {}
        self.alias_dir = self.cache_dir / "_aliases"
        self.alias_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir = self.cache_dir / "_blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._load_aliases()
        self.predefined_prompt_aliases = dict(DEFAULT_PROMPT_ALIASES)

    def _effective_ttl(self, ttl: float | None) -> float | None:
        return self.default_ttl if ttl is None else ttl

    def _alias_path(self, name: str) -> Path:
        sanitized = name.replace("/", "__").replace(":", "--")
        return self.alias_dir / f"{sanitized}.json"

    def _modelfile_path(self, name: str) -> Path:
        sanitized = name.replace("/", "__").replace(":", "--")
        return self.alias_dir / f"{sanitized}.Modelfile"

    def _load_aliases(self) -> None:
        for path in self.alias_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to load alias metadata from %s: %s", path, exc)
                continue
            name = data.get("name")
            if not name:
                continue
            modelfile_path = self._modelfile_path(name)
            if modelfile_path.exists():
                try:
                    data["modelfile"] = modelfile_path.read_text(encoding="utf-8")
                except OSError as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to read Modelfile for %s: %s", name, exc)
            metadata_defaults = {
                "model": data.get("model") or name,
                "parameters": data.get("options") or {},
                "prompt_aliases": (data.get("metadata") or {}).get("prompt_aliases"),
                "description": (data.get("metadata") or {}).get("description", ""),
            }
            try:
                normalised_metadata = merge_metadata(data.get("metadata"), metadata_defaults)
                validate_metadata(normalised_metadata)
            except MetadataError as exc:
                LOGGER.warning("Discarding invalid metadata for %s: %s", name, exc)
                normalised_metadata = normalize_metadata(metadata_defaults)
            data["metadata"] = normalised_metadata
            self.aliases[name] = data

    def _save_alias(self, alias: dict[str, Any]) -> None:
        path = self._alias_path(alias["name"])
        path.write_text(json.dumps(alias, indent=2, sort_keys=True), encoding="utf-8")

    def _persist_modelfile(self, name: str, modelfile: str | None) -> None:
        path = self._modelfile_path(name)
        if modelfile:
            path.write_text(modelfile, encoding="utf-8")
        elif path.exists():
            path.unlink()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def get_alias(self, name: str) -> dict[str, Any] | None:
        return self.aliases.get(name)

    def get_prompt_aliases(self, name: str) -> dict[str, str]:
        aliases: dict[str, str] = dict(self.predefined_prompt_aliases)
        alias_info = self.aliases.get(name)
        if alias_info is None:
            base_name, tag = _split_model_tag(name)
            if tag:
                alias_info = self.aliases.get(base_name)
        if alias_info:
            base_model = alias_info.get("model")
            if isinstance(base_model, str) and base_model and base_model != name:
                base_alias = self.aliases.get(base_model)
                if base_alias:
                    base_metadata = base_alias.get("metadata") or {}
                    aliases.update(base_metadata.get("prompt_aliases", {}))
            metadata = alias_info.get("metadata") or {}
            aliases.update(metadata.get("prompt_aliases", {}))
        return aliases

    def create_alias(
        self,
        name: str,
        base_model: str | None,
        template: str | None,
        system: str | None,
        parameters: dict[str, Any] | None,
        modelfile: str | None,
        license_info: list[str] | str | None,
        messages: list[dict[str, Any]] | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        timestamp = self._now_iso()
        existing = self.aliases.get(name, {})
        base_model_name = base_model or existing.get("model") or name
        metadata_defaults = {
            "model": base_model_name,
            "parameters": parameters or existing.get("options") or {},
            "prompt_aliases": (existing.get("metadata") or {}).get("prompt_aliases"),
            "description": (existing.get("metadata") or {}).get("description", ""),
        }
        merged_metadata = merge_metadata(metadata, metadata_defaults)
        merged_metadata["model"] = base_model_name
        merged_metadata["parameters"] = dict(
            sorted((parameters or existing.get("options") or {}).items())
        )
        try:
            validate_metadata(merged_metadata)
        except MetadataError as exc:
            raise ValueError(f"Invalid metadata for alias {name}: {exc}") from exc
        alias: dict[str, Any] = {
            "name": name,
            "model": base_model_name,
            "options": parameters or existing.get("options") or {},
            "template": template if template is not None else existing.get("template"),
            "system": system if system is not None else existing.get("system"),
            "modelfile": modelfile if modelfile is not None else existing.get("modelfile"),
            "license": license_info if license_info is not None else existing.get("license"),
            "messages": messages if messages is not None else existing.get("messages"),
            "metadata": merged_metadata,
            "details": existing.get("details", {}),
            "created_at": existing.get("created_at", timestamp),
            "modified_at": timestamp,
        }
        built = build_modelfile(alias)
        if built:
            alias["modelfile"] = built
        self.aliases[name] = alias
        self._save_alias(alias)
        self._persist_modelfile(name, alias.get("modelfile"))
        return alias

    def list_alias_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for alias in self.aliases.values():
            record = dict(alias)
            record.setdefault("details", {})
            if "digest" not in record:
                record["digest"] = hashlib.sha256(
                    json.dumps({k: v for k, v in alias.items() if k != "digest"}, sort_keys=True).encode("utf-8")
                ).hexdigest()
            base_model = record.get("model", record["name"])
            repo_dir = self.cache_dir / base_model.replace("/", "__")
            if repo_dir.exists():
                try:
                    record["size"] = sum(p.stat().st_size for p in repo_dir.rglob("*"))
                except OSError:  # pragma: no cover - filesystem errors are rare
                    record["size"] = 0
            else:
                record.setdefault("size", 0)
            records.append(record)
        return records

    def describe_model(self, name: str) -> dict[str, Any] | None:
        alias = self.aliases.get(name)
        base_name = name
        if alias:
            alias_model = alias.get("model")
            if isinstance(alias_model, str) and alias_model:
                base_name = alias_model
        else:
            split_name, tag = _split_model_tag(name)
            base_name = split_name
            if tag:
                alias = self.aliases.get(split_name)
                if alias:
                    alias_model = alias.get("model")
                    if isinstance(alias_model, str) and alias_model:
                        base_name = alias_model
        repo_dir = self.cache_dir / base_name.replace("/", "__")
        details = alias.get("details") if alias else {}
        info: dict[str, Any] = {
            "name": name,
            "model": base_name,
            "modelfile": alias.get("modelfile") if alias else None,
            "template": alias.get("template") if alias else None,
            "system": alias.get("system") if alias else None,
            "parameters": alias.get("options") if alias else {},
            "license": alias.get("license") if alias else None,
            "messages": alias.get("messages") if alias else None,
            "metadata": alias.get("metadata") if alias else {},
            "details": details or {},
        }
        if repo_dir.exists():
            info["size"] = sum(p.stat().st_size for p in repo_dir.rglob("*"))
            info["path"] = str(repo_dir)
        return info if alias or repo_dir.exists() else None

    def copy_model(self, source: str, destination: str) -> bool:
        if destination == source:
            return True
        alias = self.aliases.get(source)
        timestamp = self._now_iso()
        if alias:
            new_alias = json.loads(json.dumps(alias))
            new_alias["name"] = destination
            new_alias["modified_at"] = timestamp
            metadata_defaults = {
                "model": new_alias.get("model") or destination,
                "parameters": new_alias.get("options") or {},
                "prompt_aliases": (new_alias.get("metadata") or {}).get("prompt_aliases"),
                "description": (new_alias.get("metadata") or {}).get("description", ""),
            }
            try:
                new_alias["metadata"] = merge_metadata(new_alias.get("metadata"), metadata_defaults)
                validate_metadata(new_alias["metadata"])
            except MetadataError as exc:
                LOGGER.warning("Invalid metadata while copying %s -> %s: %s", source, destination, exc)
                new_alias["metadata"] = normalize_metadata(metadata_defaults)
            rebuilt = build_modelfile(new_alias)
            if rebuilt:
                new_alias["modelfile"] = rebuilt
            self.aliases[destination] = new_alias
            self._save_alias(new_alias)
            self._persist_modelfile(destination, new_alias.get("modelfile"))
            return True
        repo_dir = self.cache_dir / source.replace("/", "__")
        if not repo_dir.exists():
            return False
        self.create_alias(destination, source, None, None, {}, None, None, None, None)
        return True

    def delete_model(self, name: str) -> bool:
        removed = False
        if name in self.aliases:
            removed = True
            self.aliases.pop(name)
            path = self._alias_path(name)
            if path.exists():
                path.unlink()
            modelfile_path = self._modelfile_path(name)
            if modelfile_path.exists():
                modelfile_path.unlink()
        repo_dir = self.cache_dir / name.replace("/", "__")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
            removed = True
        return removed

    def blob_exists(self, digest: str) -> bool:
        return self._blob_path(digest).exists()

    def _blob_path(self, digest: str) -> Path:
        sanitized = digest.replace("/", "_").replace(":", "_")
        return self.blob_dir / sanitized

    def save_blob(self, digest: str, data: bytes) -> Path:
        if not data:
            raise ValueError("Empty payload is not allowed")
        algo, _, digest_value = digest.partition(":")
        expected = digest_value or algo
        if not expected:
            raise ValueError("Invalid digest")
        computed = hashlib.sha256(data).hexdigest()
        if computed.lower() != expected.lower():
            raise ValueError("Digest mismatch")
        path = self._blob_path(digest)
        path.write_bytes(data)
        return path

    def resolve_model(self, name: str) -> tuple[str, dict[str, Any]]:
        info = self.aliases.get(name)
        if info is None:
            base_name, tag = _split_model_tag(name)
            if tag:
                info = self.aliases.get(base_name)
                if info is None:
                    return base_name, {}
                model_name = info.get("model", base_name)
                options = info.get("options", {})
                return model_name, options
            return base_name, {}
        model_name = info.get("model", name)
        options = info.get("options", {})
        return model_name, options

    def iter_local_models(self) -> Iterator[tuple[str, Path, Path]]:
        for entry in sorted(self.cache_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            model_name = entry.name.replace("__", "/")
            local_dir = _resolve_local_model_dir(entry)
            if local_dir is None:
                continue
            config_path = local_dir / "config.json"
            if not config_path.exists():
                continue
            yield model_name, entry, config_path

    async def ensure_model(self, name: str, options: GenerateOptions | None, ttl: float | None) -> ManagedModel:
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
                    entry = await self.models.upsert(resolved_name, model, self._effective_ttl(ttl))
        if ttl is not None:
            await self.models.update_ttl(resolved_name, ttl)
        await self.models.increment(resolved_name)
        return entry.value

    def _load_text_model(self, name: str, options: dict[str, Any]) -> ManagedModel:
        load_kwargs: dict[str, Any] = {}
        device = detect_default_device()
        dtype = choose_dtype(device)
        quantization_enabled = False
        if options.get("load_in_8bit"):
            load_kwargs["load_in_8bit"] = True
            quantization_enabled = True
        elif options.get("load_in_4bit"):
            if _is_mxfp4_supported(device):
                load_kwargs["load_in_4bit"] = True
                quantization_enabled = True
            else:
                LOGGER.warning(
                    "Requested 4-bit loading but Triton >= 3.4.0 is unavailable; falling back to %s precision",
                    dtype,
                )
        if not quantization_enabled:
            load_kwargs["dtype"] = dtype

        explicit_device_map = options.get("device_map")
        prefer_full_device = False
        if explicit_device_map is not None:
            load_kwargs["device_map"] = explicit_device_map
        elif device in {"cuda", "mps"}:
            load_kwargs["device_map"] = {"": device}
            prefer_full_device = device == "cuda"
        if options.get("trust_remote_code"):
            load_kwargs["trust_remote_code"] = True
        repo_dir = self.cache_dir / name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        local_dir = _resolve_local_model_dir(repo_dir)
        load_target = local_dir or (repo_dir if any(repo_dir.iterdir()) else name)
        try:
            model = AutoModelForCausalLM.from_pretrained(load_target, **load_kwargs)
        except RuntimeError as exc:
            if prefer_full_device and _is_cuda_oom_error(exc):
                LOGGER.warning("Falling back to device_map='auto' due to limited GPU memory: %s", exc)
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs["device_map"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(load_target, **fallback_kwargs)
            else:
                raise
        tokenizer = AutoTokenizer.from_pretrained(load_target, use_fast=True)
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

    async def release(self, name: str, ttl: float | None) -> None:
        resolved_name, _ = self.resolve_model(name)
        await self.models.decrement(resolved_name)
        if ttl == 0:
            await self.models.update_ttl(resolved_name, 0)
            await self.models.evict_expired()

    async def list_loaded(self) -> dict[str, dict[str, Any]]:
        snapshot = await self.models.snapshot()
        result: dict[str, dict[str, Any]] = {}
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

    async def set_keep_alive(self, name: str, ttl: float | None) -> None:
        resolved_name, _ = self.resolve_model(name)
        ttl_value = None if ttl is None or math.isinf(ttl) else ttl
        await self.models.update_ttl(resolved_name, ttl_value)
        if ttl_value == 0:
            await self.models.evict_expired()

    async def pull(self, name: str, revision: str | None, trust_remote_code: bool) -> Path:
        base_name, tag = _split_model_tag(name)
        repo_dir = self.cache_dir / base_name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)

        def _download() -> Path:
            download_fn = cast(Callable[..., Any], snapshot_download)
            kwargs: dict[str, Any] = {
                "local_dir": repo_dir,
                "local_dir_use_symlinks": False,
            }
            effective_revision = revision or tag
            if effective_revision:
                kwargs["revision"] = effective_revision
            if _SNAPSHOT_DOWNLOAD_SUPPORTS_TRUST_REMOTE_CODE and trust_remote_code:
                download_fn(
                    base_name,
                    trust_remote_code=True,
                    **kwargs,
                )
            else:
                download_fn(
                    base_name,
                    **kwargs,
                )
            return repo_dir

        path = await asyncio.get_running_loop().run_in_executor(None, _download)
        if tag:
            base_alias = self.aliases.get(base_name)
            parameters: dict[str, Any] | None = None
            template: str | None = None
            system: str | None = None
            modelfile: str | None = None
            license_info: list[str] | str | None = None
            messages: list[dict[str, Any]] | None = None
            metadata: dict[str, Any] | None = None
            if base_alias:
                parameters = dict(base_alias.get("options") or {})
                template = base_alias.get("template")
                system = base_alias.get("system")
                modelfile = base_alias.get("modelfile")
                license_info = base_alias.get("license")
                messages = base_alias.get("messages")
                base_metadata = base_alias.get("metadata")
                if base_metadata:
                    metadata = deepcopy(base_metadata)
            try:
                self.create_alias(
                    name,
                    base_name,
                    template,
                    system,
                    parameters,
                    modelfile,
                    license_info,
                    messages,
                    metadata,
                )
            except ValueError:
                LOGGER.debug("Skipping alias creation for %s due to validation error", name)
        return path

    async def ensure_embeddings_model(self, name: str, ttl: float | None) -> ManagedModel:
        resolved_name, _ = self.resolve_model(name)
        entry = await self.models.get(resolved_name)
        if entry is None:
            lock = MODEL_LOCKS.setdefault(resolved_name, asyncio.Lock())
            async with lock:
                entry = await self.models.get(resolved_name)
                if entry is None:
                    managed = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: self._load_embeddings_model(resolved_name)
                    )
                    entry = await self.models.upsert(resolved_name, managed, self._effective_ttl(ttl))
        if ttl is not None:
            await self.models.update_ttl(resolved_name, ttl)
        await self.models.increment(resolved_name)
        return entry.value

    def _load_embeddings_model(self, name: str) -> ManagedModel:
        from sentence_transformers import SentenceTransformer

        repo_dir = self.cache_dir / name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        local_dir = _resolve_local_model_dir(repo_dir)
        load_target = local_dir or (repo_dir if any(repo_dir.iterdir()) else name)
        model = SentenceTransformer(str(load_target))
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


def run_generation(
    manager: ModelManager,
    request_options: GenerateOptions,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    tokenizer,
    model,
    prompt_text: str,
    streamer: TextIteratorStreamer,
) -> dict[str, Any]:
    prompt_tokens = len(tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device),
        "streamer": streamer,
    }
    if request_options.max_tokens is not None:
        generation_kwargs["max_new_tokens"] = request_options.max_tokens
    else:
        context_window = getattr(model.config, "max_position_embeddings", None)
        if isinstance(context_window, int) and context_window > prompt_tokens:
            default_max_tokens = context_window - prompt_tokens
        else:
            default_max_tokens = 1024
        generation_kwargs["max_new_tokens"] = max(1, default_max_tokens)
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
