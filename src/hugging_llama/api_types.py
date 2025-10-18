"""Pydantic models for Ollama compatible API."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
    def _coerce_options(cls, values: Any) -> Any:
        if isinstance(values, (bytes, bytearray)):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                return values
        elif isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                return values

        if isinstance(values, Mapping):
            options = values.get("options")
            if isinstance(options, Mapping):
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
    def _coerce_options(cls, values: Any) -> Any:
        if isinstance(values, (bytes, bytearray)):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                return values
        elif isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                return values

        if isinstance(values, Mapping):
            options = values.get("options")
            if isinstance(options, Mapping):
                values = dict(values)
                values["options"] = GenerateOptions(**options)

        return values


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completion request schema."""

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | Sequence[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    tools: list[ToolDefinition] | None = None
    response_format: dict[str, Any] | None = None
    user: str | None = None
    n: int | None = None
    logit_bias: dict[str, float] | None = None
    tool_choice: Any | None = None
    stream_options: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")

    def to_generate_options(self) -> GenerateOptions | None:
        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.stop is not None:
            options["stop"] = self.stop
        if self.presence_penalty is not None:
            options["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            options["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:
            options["seed"] = self.seed
        if not options:
            return None
        return GenerateOptions(**options)


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | Sequence[str] | None = None
    prompt: str | None = None
    options: dict[str, Any] | None = None
    keep_alive: int | float | str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_prompt(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values.get("input") is None and values.get("prompt") is not None:
            values = dict(values)
            values["input"] = values.get("prompt")
        return values

    @model_validator(mode="after")
    def _ensure_input(self) -> EmbeddingsRequest:
        if self.input is None:
            raise ValueError("Embeddings request requires an input or prompt field")
        return self

    @property
    def normalized_inputs(self) -> list[str]:
        if (
            isinstance(self.input, Sequence)
            and not isinstance(self.input, str)
            and not isinstance(self.input, bytes)
            and not isinstance(self.input, bytearray)
        ):
            return [str(item) for item in self.input]
        return [str(self.input)]


class PullRequest(BaseModel):
    model: str
    revision: str | None = None
    trust_remote_code: bool = False
    insecure: bool = False
    stream: bool = True

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_model(cls, values: Any) -> Any:
        if isinstance(values, bytes | bytearray):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                return values
        elif isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                return values

        if isinstance(values, Mapping):
            if "model" not in values and values.get("name") is not None:
                values = dict(values)
                values["model"] = values["name"]

        return values


class CreateRequest(BaseModel):
    model: str
    from_: str | None = Field(default=None, alias="from")
    files: dict[str, str] | None = None
    adapters: dict[str, str] | None = None
    template: str | None = None
    license: str | list[str] | None = None
    system: str | None = None
    parameters: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None
    modelfile: str | None = None
    quantize: str | None = None
    metadata: dict[str, Any] | None = None
    stream: bool = True

    model_config = ConfigDict(populate_by_name=True)


class EditRequest(BaseModel):
    model: str
    template: str | None = None
    system: str | None = None
    parameters: dict[str, Any] | None = None
    license: list[str] | str | None = None
    messages: list[dict[str, Any]] | None = None
    modelfile: str | None = None
    metadata: dict[str, Any] | None = None


class ShowRequest(BaseModel):
    model: str
    verbose: bool = False


class CopyRequest(BaseModel):
    source: str
    destination: str


class DeleteRequest(BaseModel):
    model: str


class PushRequest(BaseModel):
    model: str
    insecure: bool = False
    stream: bool = True


class KeepAliveUpdate(BaseModel):
    model: str
    keep_alive: int | float | str | None = None


class UnloadRequest(BaseModel):
    model: str
    keep_alive: int | float | str | None = None
