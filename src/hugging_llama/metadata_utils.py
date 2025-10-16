"""Helpers for normalising and validating metadata stored alongside models."""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - used for static analysis fallback
    import tomli as tomllib  # type: ignore[assignment, import-not-found]
else:  # pragma: no cover - runtime import resolution
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
        import tomli as tomllib  # type: ignore[assignment, import-not-found]

LOGGER = logging.getLogger(__name__)

PROMPT_ALIAS_PREFIXES: Final[tuple[str, ...]] = ("alias", "prompt")
DEFAULT_PROMPT_ALIASES: Final[dict[str, str]] = {
    "default": "",
}


class MetadataError(ValueError):
    """Raised when metadata payloads are invalid."""


def _normalise_key(key: str) -> str:
    return "_".join(key.strip().lower().replace("-", "_").split())


def _normalise_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): mapping[k] for k in sorted(mapping)}


def _normalise_sequence(values: Sequence[Any]) -> list[Any]:
    return [value for value in values]


def _normalise_parameters(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): value[k] for k in sorted(value)}
    raise MetadataError("Parameters must be a mapping")


def _normalise_prompt_aliases(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MetadataError("prompt_aliases must be a mapping")
    aliases: dict[str, str] = {}
    for key in sorted(value):
        alias_value = value[key]
        if alias_value is None:
            continue
        aliases[str(key)] = str(alias_value)
    return aliases


def _normalise_generic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalise_mapping(value)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and not isinstance(value, bytes)
        and not isinstance(value, bytearray)
    ):
        return _normalise_sequence(value)
    return value


DEFAULT_METADATA: Final[dict[str, Any]] = {
    "model": "",
    "description": "",
    "parameters": {},
    "prompt_aliases": dict(DEFAULT_PROMPT_ALIASES),
}


def normalize_metadata(
    metadata: Mapping[str, Any] | None,
    defaults: Mapping[str, Any] | None = None,
    *,
    fill_required: bool = True,
) -> dict[str, Any]:
    """Return metadata with normalised keys, deterministic ordering, and canonical types."""

    normalised: dict[str, Any] = {}
    if defaults:
        normalised.update(
            {
                _normalise_key(key): _normalise_generic_value(value)
                for key, value in defaults.items()
                if value is not None
            }
        )

    if metadata:
        for raw_key, value in metadata.items():
            if value is None:
                continue
            key = _normalise_key(raw_key)
            if key == "parameters":
                normalised[key] = _normalise_parameters(value)
            elif key == "prompt_aliases":
                normalised[key] = _normalise_prompt_aliases(value)
            else:
                normalised[key] = _normalise_generic_value(value)

    if fill_required:
        for required_key, default_value in DEFAULT_METADATA.items():
            if required_key not in normalised:
                if isinstance(default_value, Mapping):
                    normalised[required_key] = dict(default_value)
                else:
                    normalised[required_key] = default_value

    prompt_aliases = normalised.get("prompt_aliases", {})
    if prompt_aliases:
        if not isinstance(prompt_aliases, Mapping):
            raise MetadataError("prompt_aliases must be a mapping")
        normalised["prompt_aliases"] = {
            **({} if not fill_required else DEFAULT_PROMPT_ALIASES),
            **{str(key): str(prompt_aliases[key]) for key in prompt_aliases},
        }
    elif fill_required:
        normalised["prompt_aliases"] = dict(DEFAULT_PROMPT_ALIASES)

    return dict(sorted(normalised.items()))


def _deep_merge(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def merge_metadata(metadata: Mapping[str, Any] | None, defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge ``metadata`` over ``defaults`` using canonical normalisation."""

    base = normalize_metadata(defaults)
    updates = normalize_metadata(metadata, fill_required=False)
    result: dict[str, Any] = dict(base)
    _deep_merge(result, updates)
    return dict(sorted(result.items()))


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate that required fields are present with canonical types."""

    errors: list[str] = []
    model = metadata.get("model")
    if not isinstance(model, str) or not model.strip():
        errors.append("metadata.model must be a non-empty string")

    description = metadata.get("description")
    if not isinstance(description, str):
        errors.append("metadata.description must be a string")

    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        errors.append("metadata.parameters must be a mapping")

    prompt_aliases = metadata.get("prompt_aliases")
    if prompt_aliases is not None:
        if not isinstance(prompt_aliases, Mapping):
            errors.append("metadata.prompt_aliases must be a mapping of aliases to text")
        else:
            for alias, text in prompt_aliases.items():
                if not isinstance(alias, str) or not isinstance(text, str):
                    errors.append("metadata.prompt_aliases must contain string keys and values")
                    break

    if errors:
        raise MetadataError("; ".join(errors))


def serialize_metadata(metadata: Mapping[str, Any], fmt: str = "json") -> str:
    """Serialize metadata to JSON or TOML after validation."""

    validate_metadata(metadata)
    fmt_lower = fmt.lower()
    if fmt_lower == "json":
        return json.dumps(metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if fmt_lower == "toml":
        return _dump_toml(metadata)
    raise MetadataError(f"Unsupported metadata format: {fmt}")


def deserialize_metadata(payload: str, fmt: str = "json") -> dict[str, Any]:
    """Deserialize metadata text into a normalised mapping."""

    if not payload:
        return dict(DEFAULT_METADATA)
    fmt_lower = fmt.lower()
    try:
        if fmt_lower == "json":
            data = json.loads(payload)
        elif fmt_lower == "toml":
            data = tomllib.loads(payload)
        else:
            raise MetadataError(f"Unsupported metadata format: {fmt}")
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:  # type: ignore[attr-defined]
        raise MetadataError(f"Failed to parse metadata: {exc}") from exc
    normalised = normalize_metadata(data)
    validate_metadata(normalised)
    return normalised


def _dump_toml(mapping: Mapping[str, Any]) -> str:
    lines: list[str] = []
    scalars: dict[str, Any] = {}
    nested: dict[str, Mapping[str, Any]] = {}
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, Mapping):
            nested[key] = value
        else:
            scalars[key] = value

    for key, value in scalars.items():
        lines.append(f"{key} = {_format_toml_scalar(value)}")

    for key, value in nested.items():
        lines.append("")
        lines.append(f"[{key}]")
        lines.append(_dump_toml(value))

    return "\n".join(lines).strip()


def _format_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) or isinstance(value, float):
        return repr(value)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and not isinstance(value, bytes)
        and not isinstance(value, bytearray)
    ):
        formatted = ", ".join(_format_toml_scalar(item) for item in value)
        return f"[{formatted}]"
    escaped = json.dumps(str(value))
    return escaped


__all__ = [
    "DEFAULT_PROMPT_ALIASES",
    "MetadataError",
    "PROMPT_ALIAS_PREFIXES",
    "deserialize_metadata",
    "merge_metadata",
    "normalize_metadata",
    "serialize_metadata",
    "validate_metadata",
]
