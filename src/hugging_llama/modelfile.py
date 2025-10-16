"""Utilities for working with Ollama-compatible Modelfiles."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .metadata_utils import (
    MetadataError,
    deserialize_metadata,
    merge_metadata,
    serialize_metadata,
    validate_metadata,
)


class ModelfileError(ValueError):
    """Raised when a Modelfile is invalid."""


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _consume_block(initial: str, iterator: Iterable[str], delimiter: str) -> str:
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


def _parse_block_value(value: str, iterator: Iterable[str]) -> str:
    stripped = value.lstrip()
    if stripped.startswith('"""') or stripped.startswith("'''"):
        delimiter = stripped[:3]
        remainder = stripped[3:]
        if remainder.endswith(delimiter):
            return remainder[: -len(delimiter)]
        return _consume_block(remainder, iterator, delimiter)
    return _strip_wrapping_quotes(stripped.strip())


def parse_modelfile(text: str) -> dict[str, Any]:
    """Parse an Ollama-style Modelfile into a structured mapping."""

    result: dict[str, Any] = {
        "model": None,
        "template": None,
        "system": None,
        "parameters": {},
        "license": [],
        "messages": [],
        "adapters": {},
        "files": {},
        "metadata": {},
    }
    if not text:
        return result

    iterator = iter(text.splitlines())
    for raw_line in iterator:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keyword, _, remainder = stripped.partition(" ")
        keyword_upper = keyword.upper()
        remainder = remainder.strip()

        if keyword_upper == "FROM":
            result["model"] = remainder.strip()
        elif keyword_upper == "TEMPLATE":
            result["template"] = _parse_block_value(remainder, iterator)
        elif keyword_upper == "SYSTEM":
            result["system"] = _parse_block_value(remainder, iterator)
        elif keyword_upper == "PARAMETER":
            key, _, value = remainder.partition(" ")
            if key:
                result.setdefault("parameters", {})[key] = _strip_wrapping_quotes(value.strip())
        elif keyword_upper == "LICENSE":
            value = _parse_block_value(remainder, iterator)
            if value:
                result.setdefault("license", []).append(value)
        elif keyword_upper == "MESSAGE":
            role, _, value = remainder.partition(" ")
            content = _parse_block_value(value, iterator)
            entry = {"role": role or "assistant", "content": content}
            result.setdefault("messages", []).append(entry)
        elif keyword_upper == "ADAPTER":
            name, _, value = remainder.partition(" ")
            if name:
                result.setdefault("adapters", {})[name] = _strip_wrapping_quotes(value.strip())
        elif keyword_upper == "FILE":
            name, _, value = remainder.partition(" ")
            if name:
                result.setdefault("files", {})[name] = _strip_wrapping_quotes(value.strip())
        elif keyword_upper == "METADATA":
            format_hint, _, value = remainder.partition(" ")
            payload = _parse_block_value(value, iterator)
            fmt = format_hint or "json"
            try:
                parsed = deserialize_metadata(payload, fmt)
            except MetadataError as exc:
                raise ModelfileError(f"Invalid metadata block: {exc}") from exc
            existing = result.setdefault("metadata", {})
            result["metadata"] = merge_metadata(parsed, existing)
        else:
            metadata = result.setdefault("metadata", {})
            metadata.setdefault(keyword_upper, []).append(remainder)
    return result


def _needs_quotes(value: str) -> bool:
    return not value or any(ch.isspace() for ch in value) or any(ch in value for ch in '"\'')


def _format_scalar(value: Any) -> str:
    text = str(value)
    if _needs_quotes(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _format_block(keyword: str, value: str) -> list[str]:
    if value is None:
        return []
    if "\n" in value or value.strip() != value:
        return [f"{keyword} \"\"\"\n{value}\n\"\"\""]
    return [f"{keyword} {_format_scalar(value)}"]


def build_modelfile(data: dict[str, Any]) -> str:
    """Construct a Modelfile representation from metadata."""

    lines: list[str] = []

    model = data.get("model")
    if model:
        lines.append(f"FROM {_format_scalar(model)}")

    template = data.get("template")
    if template:
        lines.extend(_format_block("TEMPLATE", str(template)))

    system_prompt = data.get("system")
    if system_prompt:
        lines.extend(_format_block("SYSTEM", str(system_prompt)))

    parameters = data.get("parameters") or data.get("options") or {}
    for key in sorted(parameters):
        value = parameters[key]
        lines.append(f"PARAMETER {key} {_format_scalar(value)}")

    licenses = data.get("license")
    if isinstance(licenses, list) or isinstance(licenses, tuple):
        for entry in licenses:
            lines.extend(_format_block("LICENSE", str(entry)))
    elif isinstance(licenses, str):
        lines.extend(_format_block("LICENSE", licenses))

    for message in data.get("messages") or []:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        lines.extend(_format_block(f"MESSAGE {role}", str(content)))

    for name, value in sorted((data.get("adapters") or {}).items()):
        lines.append(f"ADAPTER {name} {_format_scalar(value)}")

    for name, value in sorted((data.get("files") or {}).items()):
        lines.append(f"FILE {name} {_format_scalar(value)}")

    metadata_block = data.get("metadata")
    if metadata_block:
        serialized = serialize_metadata(metadata_block, "json")
        lines.extend(_format_block("METADATA JSON", serialized))

    return "\n".join(lines).strip()


def validate_modelfile_data(data: dict[str, Any]) -> None:
    """Validate the parsed Modelfile contents."""

    errors: list[str] = []
    model = data.get("model")
    if not model or not isinstance(model, str):
        errors.append("Missing FROM directive in Modelfile")

    parameters = data.get("parameters") or {}
    if not isinstance(parameters, dict):
        errors.append("Parameters section must be a mapping")

    for message in data.get("messages", []):
        if not isinstance(message, dict):
            errors.append("Messages entries must be objects")
            continue
        if not message.get("role"):
            errors.append("Message entries must include a role")
        if "content" not in message:
            errors.append("Message entries must include content")

    metadata = data.get("metadata")
    if metadata:
        try:
            validate_metadata(metadata)
        except MetadataError as exc:
            errors.append(str(exc))

    if errors:
        raise ModelfileError("; ".join(errors))


__all__ = ["ModelfileError", "parse_modelfile", "build_modelfile", "validate_modelfile_data"]
