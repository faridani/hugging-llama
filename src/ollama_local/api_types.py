"""Pydantic models for Ollama compatible API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, root_validator, validator


class GenerateOptions(BaseModel):
    """Generation options that roughly mirror Ollama/OpenAI semantics."""

    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    presence_penalty: Optional[float] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    stop: Optional[Union[str, Sequence[str]]] = None
    num_return_sequences: int = Field(default=1, ge=1)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    dtype: Optional[str] = Field(default=None, pattern=r"^(auto|float16|bfloat16|float32)$")
    seed: Optional[int] = None

    class Config:
        extra = "allow"

    @validator("stop", pre=True)
    def _normalise_stop(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    images: Optional[List[str]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    raw: bool = False
    stream: bool = True
    keep_alive: Optional[Union[int, float, str]] = None
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[GenerateOptions] = None

    @root_validator(pre=True)
    def _coerce_options(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        options = values.get("options")
        if isinstance(options, dict):
            values["options"] = GenerateOptions(**options)
        return values


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: Dict[str, Any]


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[ToolDefinition]] = None
    stream: bool = True
    keep_alive: Optional[Union[int, float, str]] = None
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[GenerateOptions] = None

    @root_validator(pre=True)
    def _coerce_options(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        options = values.get("options")
        if isinstance(options, dict):
            values["options"] = GenerateOptions(**options)
        return values


class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, Sequence[str]]
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[int, float, str]] = None


class PullRequest(BaseModel):
    model: str
    revision: Optional[str] = None
    trust_remote_code: bool = False


class KeepAliveUpdate(BaseModel):
    model: str
    keep_alive: Optional[Union[int, float, str]] = None
