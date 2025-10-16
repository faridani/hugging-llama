"""Curated catalog of popular models with VRAM estimates."""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx


LOGGER = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api/models"
HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


@dataclass(frozen=True)
class CatalogEntry:
    """Metadata for a catalog model entry."""

    name: str
    parameters: str
    size_gb: float
    precision: str
    description: str


STATIC_MODEL_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        name="hf-internal-testing/tiny-random-gpt2",
        parameters="124M",
        size_gb=0.1,
        precision="FP32",
        description="Tiny random GPT-2 checkpoint used for smoke testing.",
    ),
    CatalogEntry(
        name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        parameters="3.8B",
        size_gb=3.0,
        precision="4-bit",
        description="Quantized Phi-3 Mini optimized for sub-6 GB cards.",
    ),
    CatalogEntry(
        name="google/gemma-2b-it",
        parameters="2B",
        size_gb=4.0,
        precision="FP16",
        description="Instruction-tuned Gemma 2B for lightweight assistants.",
    ),
    CatalogEntry(
        name="microsoft/Phi-2",
        parameters="2.7B",
        size_gb=6.0,
        precision="FP16",
        description="Compact reasoning-focused Phi-2 checkpoint.",
    ),
    CatalogEntry(
        name="microsoft/Phi-3-mini-4k-instruct",
        parameters="3.8B",
        size_gb=8.0,
        precision="FP16",
        description="Phi-3 Mini 4k general-purpose assistant.",
    ),
    CatalogEntry(
        name="HuggingFaceH4/zephyr-7b-beta",
        parameters="7B",
        size_gb=13.0,
        precision="FP16",
        description="Chat-aligned Zephyr 7B based on Mistral.",
    ),
    CatalogEntry(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        parameters="7B",
        size_gb=14.0,
        precision="FP16",
        description="Official instruct-tuned Mistral 7B release.",
    ),
    CatalogEntry(
        name="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        parameters="7B",
        size_gb=8.0,
        precision="GPTQ 4-bit",
        description="GPTQ quantized Mistral 7B for 10 GB GPUs.",
    ),
    CatalogEntry(
        name="google/gemma-7b-it",
        parameters="7B",
        size_gb=14.0,
        precision="FP16",
        description="Gemma 7B instruction-tuned checkpoint.",
    ),
    CatalogEntry(
        name="meta-llama/Meta-Llama-3-8B-Instruct",
        parameters="8B",
        size_gb=16.0,
        precision="FP16",
        description="Meta Llama 3 8B with instruction tuning and 8k context.",
    ),
    CatalogEntry(
        name="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        parameters="8B",
        size_gb=8.5,
        precision="GGUF Q4_K_M",
        description="GGUF quantized Llama 3 8B for llama.cpp style runtimes.",
    ),
    CatalogEntry(
        name="tiiuae/falcon-40b-instruct",
        parameters="40B",
        size_gb=80.0,
        precision="FP16",
        description="Falcon 40B instruction tuned for advanced assistants.",
    ),
    CatalogEntry(
        name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        parameters="45B",
        size_gb=120.0,
        precision="FP16",
        description="Sparse MoE Mixtral 8x7B instruct checkpoint.",
    ),
    CatalogEntry(
        name="meta-llama/Meta-Llama-3-70B-Instruct",
        parameters="70B",
        size_gb=140.0,
        precision="FP16",
        description="Large Llama 3 70B instruct for maximum quality.",
    ),
)


REMOTE_MODEL_REPOS: tuple[str, ...] = (
    "ai21labs/AI21-Jamba-Reasoning-3B",
    "ai21labs/AI21-Jamba-Reasoning-3B-GGUF",
    "DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "dphn/Dolphin-Mistral-24B-Venice-Edition",
    "FractalAIResearch/Fathom-Search-4B",
    "google/gemma-3-1b-it",
    "google/gemma-3-270m",
    "google/gemma-3-270m-it",
    "google/gemma-3n-E4B-it-litert-lm",
    "gustavecortal/Beck-8B",
    "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-micro",
    "inclusionAI/Ling-1T",
    "inclusionAI/Ling-flash-2.0",
    "inclusionAI/Ring-1T",
    "inclusionAI/Ring-1T-FP8",
    "inclusionAI/Ring-1T-preview",
    "KORMo-Team/KORMo-10B-base",
    "KORMo-Team/KORMo-10B-sft",
    "Kwaipilot/KAT-Dev",
    "Kwaipilot/KAT-Dev-72B-Exp",
    "LiquidAI/LFM2-350M-PII-Extract-JP",
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2-8B-A1B-GGUF",
    "litert-community/Gemma3-1B-IT",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "microsoft/UserLM-8b",
    "moonshotai/Kimi-K2-Instruct-0905",
    "openai-community/gpt2",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openbmb/MiniCPM4.1-8B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "radicalnumerics/RND1-Base-0910",
    "Salesforce/CoDA-v0-Instruct",
    "sdobson/nanochat",
    "unsloth/GLM-4.6-GGUF",
    "unsloth/gpt-oss-20b-GGUF",
    "unsloth/LFM2-8B-A1B-GGUF",
    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "vandijklab/C2S-Scale-Gemma-2-27B",
    "vngrs-ai/Kumru-2B",
    "zai-org/GLM-4.5-Air",
    "zai-org/GLM-4.6",
    "zai-org/GLM-4.6-FP8",
)


MODEL_FILE_EXTENSIONS = (".safetensors", ".bin", ".pt", ".gguf", ".ggml")
PARAMETER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(?:\s*)([BbMmTt])")
NUMBER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

_cached_catalog: tuple[CatalogEntry, ...] | None = None


def load_model_catalog(force_refresh: bool = False) -> tuple[CatalogEntry, ...]:
    """Return the curated catalog augmented with live Hugging Face metadata."""

    global MODEL_CATALOG, _cached_catalog
    if _cached_catalog is not None and not force_refresh:
        return _cached_catalog

    entries = list(STATIC_MODEL_CATALOG)
    try:
        remote_entries = _fetch_remote_catalog(REMOTE_MODEL_REPOS)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.debug("Failed to refresh remote model catalog: %s", exc, exc_info=True)
        remote_entries = []

    existing = {entry.name for entry in entries}
    for entry in remote_entries:
        if entry.name in existing:
            continue
        entries.append(entry)
        existing.add(entry.name)

    _cached_catalog = tuple(entries)
    MODEL_CATALOG = _cached_catalog
    return _cached_catalog


def _fetch_remote_catalog(repos: Sequence[str]) -> list[CatalogEntry]:
    """Fetch metadata for the supplied repositories and create catalog entries."""

    if not repos:
        return []

    results: list[CatalogEntry] = []
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        for repo_id in repos:
            metadata = _download_model_metadata(client, repo_id)
            if metadata is None:
                continue
            if not _is_text_generation(metadata):
                continue
            size_bytes = _extract_model_size_bytes(metadata)
            entry = CatalogEntry(
                name=repo_id,
                parameters=_infer_parameter_label(repo_id, metadata),
                size_gb=_bytes_to_gib(size_bytes),
                precision=_infer_precision(metadata),
                description=_infer_description(repo_id, metadata),
            )
            results.append(entry)
    return results


def _download_model_metadata(client: httpx.Client, repo_id: str) -> dict[str, Any] | None:
    """Download model metadata from the Hugging Face Hub."""

    try:
        response = client.get(f"{HF_API_URL}/{repo_id}")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        LOGGER.debug("Skipping %s due to HTTP status error: %s", repo_id, exc)
        return None
    except httpx.HTTPError as exc:
        LOGGER.debug("Skipping %s due to network error: %s", repo_id, exc)
        return None

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - unexpected response content
        LOGGER.debug("Skipping %s due to invalid JSON: %s", repo_id, exc)
        return None


def _is_text_generation(metadata: dict[str, Any]) -> bool:
    """Return ``True`` when the metadata describes a text-generation model."""

    pipeline = metadata.get("pipeline_tag")
    if isinstance(pipeline, str):
        return pipeline == "text-generation"

    card = metadata.get("cardData")
    if isinstance(card, dict):
        card_pipeline = card.get("pipeline_tag")
        if isinstance(card_pipeline, str):
            return card_pipeline == "text-generation"

    tags = metadata.get("tags")
    if isinstance(tags, Sequence):
        return any(isinstance(tag, str) and tag == "text-generation" for tag in tags)

    return False


def _extract_model_size_bytes(metadata: dict[str, Any]) -> int | None:
    """Return the approximate model size in bytes from hub metadata."""

    safetensors = metadata.get("safetensors")
    if isinstance(safetensors, dict):
        total = safetensors.get("total")
        if isinstance(total, int) and total > 0:
            return total
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict) and parameters:
            try:
                return sum(int(value) for value in parameters.values())
            except (TypeError, ValueError):
                pass

    siblings = metadata.get("siblings")
    if isinstance(siblings, Sequence):
        total = 0
        for sibling in siblings:
            if not isinstance(sibling, dict):
                continue
            filename = sibling.get("rfilename")
            size = sibling.get("size")
            if not isinstance(filename, str) or not isinstance(size, int):
                continue
            if any(filename.endswith(ext) for ext in MODEL_FILE_EXTENSIONS):
                total += size
        if total:
            return total

    used_storage = metadata.get("usedStorage")
    if isinstance(used_storage, int) and used_storage > 0:
        return used_storage

    return None


def _bytes_to_gib(size: int | None) -> float:
    """Convert a byte count to GiB with graceful handling of ``None``."""

    if size is None:
        return 0.0
    return float(size) / (1024**3)


def _infer_precision(metadata: dict[str, Any]) -> str:
    """Infer the model precision string from metadata."""

    safetensors = metadata.get("safetensors")
    if isinstance(safetensors, dict):
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict) and parameters:
            dtype = next(iter(parameters))
            if isinstance(dtype, str):
                return dtype.upper()

    config = metadata.get("config")
    if isinstance(config, dict):
        torch_dtype = config.get("torch_dtype")
        if isinstance(torch_dtype, str):
            normalized = torch_dtype.replace("torch.", "").upper()
            if normalized == "BFLOAT16":
                return "BF16"
            if normalized == "FLOAT32":
                return "FP32"
            if normalized == "FLOAT16":
                return "FP16"
            return normalized

    precision = _infer_precision_from_files(metadata.get("siblings"))
    if precision:
        return precision

    return "Unknown"


def _infer_precision_from_files(siblings: Any) -> str | None:
    """Inspect repository files to guess the model precision."""

    if not isinstance(siblings, Sequence):
        return None

    best: tuple[int, str] | None = None
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        filename = sibling.get("rfilename")
        size = sibling.get("size")
        if not isinstance(filename, str):
            continue
        file_size = size if isinstance(size, int) else 0
        lowered = filename.lower()
        if lowered.endswith(".gguf"):
            quant = filename.rsplit("-", 1)[-1].rsplit(".", 1)[0]
            label = f"GGUF {quant.upper()}" if quant else "GGUF"
            if best is None or file_size > best[0]:
                best = (file_size, label)
        elif lowered.endswith(".ggml"):
            label = "GGML"
            if best is None or file_size > best[0]:
                best = (file_size, label)
    if best:
        return best[1]
    return None


def _infer_parameter_label(repo_id: str, metadata: dict[str, Any]) -> str:
    """Infer a human readable parameter count for the model."""

    safetensors = metadata.get("safetensors")
    if isinstance(safetensors, dict):
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict) and parameters:
            try:
                total_params = sum(int(value) for value in parameters.values())
            except (TypeError, ValueError):
                total_params = 0
            if total_params:
                formatted = _format_param_count(total_params)
                if formatted != "Unknown":
                    return formatted

    model_name = repo_id.split("/")[-1]
    match = PARAMETER_PATTERN.search(model_name)
    if match:
        amount, suffix = match.groups()
        return f"{_normalize_number_text(amount)}{suffix.upper()}"

    match = NUMBER_PATTERN.search(model_name)
    if match:
        amount = match.group(1)
        if "." not in amount and len(amount) >= 3:
            return "Unknown"
        try:
            numeric = float(amount)
        except ValueError:
            return "Unknown"
        unit = "B" if numeric >= 1 else "M"
        return f"{_trim_float(numeric)}{unit}"

    return "Unknown"


def _format_param_count(count: int) -> str:
    """Format a raw parameter count into a compact string."""

    if count <= 0:
        return "Unknown"
    if count >= 1_000_000_000:
        return f"{_trim_float(count / 1_000_000_000)}B"
    if count >= 1_000_000:
        return f"{_trim_float(count / 1_000_000)}M"
    return str(count)


def _normalize_number_text(value: str) -> str:
    """Normalize a textual number representation."""

    try:
        numeric = float(value)
    except ValueError:
        return value
    return _trim_float(numeric)


def _trim_float(value: float) -> str:
    """Return a float formatted with a single decimal, trimming trailing zeros."""

    text = f"{value:.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text


def _infer_description(repo_id: str, metadata: dict[str, Any]) -> str:
    """Build a short description for a catalog entry."""

    card = metadata.get("cardData")
    if isinstance(card, dict):
        for key in ("short_description", "model_description", "headline", "summary"):
            value = card.get(key)
            if isinstance(value, str) and value.strip():
                return _collapse_whitespace(value)

    author = metadata.get("author")
    if not isinstance(author, str) or not author:
        author = repo_id.split("/")[0]

    license_name = None
    if isinstance(card, dict):
        card_license = card.get("license")
        if isinstance(card_license, str) and card_license:
            license_name = card_license
    if license_name is None:
        license_name = _extract_license_from_tags(metadata.get("tags"))

    if license_name:
        return f"Text generation model by {author} (license: {license_name})."
    return f"Text generation model by {author}."


def _collapse_whitespace(value: str) -> str:
    """Collapse runs of whitespace into single spaces."""

    return " ".join(value.split())


def _extract_license_from_tags(tags: Any) -> str | None:
    """Extract a license name from the tags list if available."""

    if not isinstance(tags, Sequence):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


MODEL_CATALOG: tuple[CatalogEntry, ...] = STATIC_MODEL_CATALOG

