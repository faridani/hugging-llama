from __future__ import annotations

import pytest

from hugging_llama.modelfile import ModelfileError, build_modelfile, parse_modelfile, validate_modelfile_data

MODFILE_TEXT = "\n".join(
    [
        "FROM mistral",
        "PARAMETER temperature 0.7",
        'SYSTEM """You are a helpful assistant."""',
        'MESSAGE user """Hello"""',
        "LICENSE Apache-2.0",
        (
            'METADATA JSON """'
            '{"model":"mistral","description":"Demo","parameters":{"temperature":0.7},'
            '"prompt_aliases":{"hello":"Hello"}}"""'
        ),
    ]
)


def test_parse_modelfile_roundtrip() -> None:
    parsed = parse_modelfile(MODFILE_TEXT)
    assert parsed["model"] == "mistral"
    assert parsed["parameters"] == {"temperature": "0.7"}
    assert parsed["system"] == "You are a helpful assistant."
    assert parsed["messages"] == [{"role": "user", "content": "Hello"}]
    assert parsed["license"] == ["Apache-2.0"]
    assert parsed["metadata"]["description"] == "Demo"
    assert parsed["metadata"]["prompt_aliases"]["hello"] == "Hello"

    rebuilt = build_modelfile(parsed)
    assert "FROM mistral" in rebuilt
    assert "PARAMETER temperature 0.7" in rebuilt
    assert "METADATA" in rebuilt


def test_validate_modelfile_requires_model() -> None:
    data = parse_modelfile("")
    with pytest.raises(ModelfileError):
        validate_modelfile_data(data)


def test_validate_modelfile_messages() -> None:
    parsed = {
        "model": "foo",
        "messages": ["invalid"],
    }
    with pytest.raises(ModelfileError):
        validate_modelfile_data(parsed)


def test_validate_modelfile_metadata() -> None:
    parsed = {
        "model": "foo",
        "metadata": {"description": 1, "parameters": []},
    }
    with pytest.raises(ModelfileError):
        validate_modelfile_data(parsed)
