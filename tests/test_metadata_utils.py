import pytest

from hugging_llama.metadata_utils import (
    MetadataError,
    deserialize_metadata,
    merge_metadata,
    normalize_metadata,
    serialize_metadata,
    validate_metadata,
)


def test_metadata_roundtrip_consistency() -> None:
    original = {
        "Model": "example/model",
        "Description": "Test metadata",
        "Parameters": {"temperature": 0.3, "top_p": 0.9},
        "Prompt Aliases": {"summary": "Summarise the input."},
    }
    normalized = normalize_metadata(original)
    validate_metadata(normalized)

    json_payload = serialize_metadata(normalized, "json")
    roundtrip_json = deserialize_metadata(json_payload, "json")
    assert roundtrip_json == normalized

    toml_payload = serialize_metadata(normalized, "toml")
    roundtrip_toml = deserialize_metadata(toml_payload, "toml")
    assert roundtrip_toml == normalized


def test_metadata_merge_backward_compatibility() -> None:
    defaults = {"model": "base", "parameters": {"temperature": 0.2}, "description": ""}
    user = {"prompt_aliases": {"hello": "Hello there"}}
    merged = merge_metadata(user, defaults)
    assert merged["model"] == "base"
    assert merged["parameters"] == {"temperature": 0.2}
    assert merged["prompt_aliases"]["hello"] == "Hello there"


def test_metadata_validation_error() -> None:
    with pytest.raises(MetadataError):
        validate_metadata({"description": 1, "parameters": []})
