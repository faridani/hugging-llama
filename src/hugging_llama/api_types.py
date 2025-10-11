"""Pydantic models for Ollama compatible API."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GenerateOptions(BaseModel):
    """Generation options that roughly mirror Ollama/OpenAI semantics."""

    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    repetition_penalty: float | None = Field(default=None, ge=0.0)
    presence_penalty: float | None = Field(default=None)
    frequency_penalty: float | None = Field(default=None)
    stop: str | Sequence[str] | None = None
    num_return_sequences: int = Field(default=1, ge=1)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    dtype: str | None = Field(default=None, pattern=r"^(auto|float16|bfloat16|float32)$")
    seed: int | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("stop", mode="before")
    @classmethod
    def _normalise_stop(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    images: list[str] | None = None
    system: str | None = None
    template: str | None = None
    raw: bool = False
    stream: bool = True
    keep_alive: int | float | str | None = None
    format: str | dict[str, Any] | None = None
    options: GenerateOptions | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_options(cls, values: dict[str, Any]) -> dict[str, Any]:
        options = values.get("options")
        if isinstance(options, dict):
            values = dict(values)
            values["options"] = GenerateOptions(**options)
        return values


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any
    name: str | None = None
    tool_call_id: str | None = None


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: dict[str, Any]


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None = None
    stream: bool = True
    keep_alive: int | float | str | None = None
    format: str | dict[str, Any] | None = None
    options: GenerateOptions | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_options(cls, values: dict[str, Any]) -> dict[str, Any]:
        options = values.get("options")
        if isinstance(options, dict):
            values = dict(values)
            values["options"] = GenerateOptions(**options)
        return values


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | Sequence[str]
    options: dict[str, Any] | None = None
    keep_alive: int | float | str | None = None


class PullRequest(BaseModel):
    model: str
    revision: str | None = None
    trust_remote_code: bool = False


class KeepAliveUpdate(BaseModel):
    model: str
    keep_alive: int | float | str | None = None
