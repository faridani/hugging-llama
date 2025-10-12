"""Curated catalog of popular models with VRAM estimates."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    """Metadata for a catalog model entry."""

    name: str
    parameters: str
    size_gb: float
    precision: str
    description: str


MODEL_CATALOG: tuple[CatalogEntry, ...] = (
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
