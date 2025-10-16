from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def load_golden(name: str):
    path = Path(__file__).parent / "golden" / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalise_event(event):
    event = dict(event)
    event.pop("created_at", None)
    event.pop("total_duration", None)
    event.pop("load_duration", None)
    event.pop("eval_count", None)
    event.pop("prompt_eval_count", None)
    return event


def collect_stream(response):
    data = []
    for line in response.iter_lines():
        if line:
            data.append(json.loads(line))
    return data


def test_generate_non_streaming(client, dummy_manager):
    res = client.post("/api/generate", json={"model": "stub", "prompt": "hi", "stream": False})
    assert res.status_code == 200
    body = res.json()
    assert "prompt_eval_count" in body
    assert "eval_count" in body
    expected = load_golden("generate_non_stream.json")
    body = normalise_event(body)
    assert body == expected


def test_generate_streaming(client, dummy_manager):
    with client.stream("POST", "/api/generate", json={"model": "stub", "prompt": "hi", "stream": True}) as response:
        events = collect_stream(response)
    payloads = [normalise_event(event) for event in events]
    expected = load_golden("generate_stream.json")
    assert payloads == expected


def test_stop_sequence_truncation(client, dummy_manager):
    dummy_manager.generation_outputs = ["hello", " world", "!"]
    res = client.post(
        "/api/generate",
        json={"model": "stub", "prompt": "hi", "stream": False, "options": {"stop": [" world"]}},
    )
    assert res.json()["response"] == "hello"


def test_json_schema_enforcement(client, dummy_manager):
    dummy_manager.generation_outputs = ['{"answer": 1}']
    schema = {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}
    res = client.post(
        "/api/generate",
        json={
            "model": "stub",
            "prompt": "hi",
            "stream": False,
            "format": schema,
        },
    )
    assert res.status_code == 200
    assert res.json()["response"] == '{"answer": 1}'

    dummy_manager.generation_outputs = ["not json"]
    res_bad = client.post(
        "/api/generate",
        json={
            "model": "stub",
            "prompt": "hi",
            "stream": False,
            "format": schema,
        },
    )
    assert res_bad.status_code == 400


def test_chat_endpoint_tool_call(client, dummy_manager):
    dummy_manager.generation_outputs = ['{"tool_calls":[{"id":"call1","function":{"name":"ping","arguments":"{}"}}]}']
    res = client.post(
        "/api/chat",
        json={
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "ping", "parameters": {}}}],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["message"]["tool_calls"] == [{"id": "call1", "function": {"name": "ping", "arguments": "{}"}}]


def test_embedding_endpoint(client, dummy_manager):
    res = client.post(
        "/api/embed",
        json={"model": "stub", "input": ["a", "b"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["embeddings"] == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_pull_endpoint(client):
    with client.stream("POST", "/api/pull", json={"model": "stub"}) as response:
        events = collect_stream(response)
    assert events[-1]["status"] == "success"


def test_tags_endpoint(client):
    res = client.get("/api/tags")
    assert res.status_code == 200
    assert "models" in res.json()


def test_ps_endpoint(client):
    res = client.get("/api/ps")
    assert res.status_code == 200
    assert "models" in res.json()


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.text == "Ollama is running"


def test_version_endpoint(client):
    res = client.get("/api/version")
    assert res.status_code == 200
    assert "version" in res.json()


def test_create_and_show_model(client, dummy_manager):
    with client.stream("POST", "/api/create", json={"model": "alias", "from": "stub"}) as response:
        events = collect_stream(response)
    assert events[-1]["status"] == "success"
    res = client.post("/api/show", json={"model": "alias"})
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "stub"


def test_copy_and_delete_model(client, dummy_manager):
    dummy_manager.create_alias("alias", "stub", None, None, {}, None, None, None, None)
    res = client.post("/api/copy", json={"source": "alias", "destination": "alias-copy"})
    assert res.status_code == 200
    assert "alias-copy" in dummy_manager.aliases
    res_delete = client.request("DELETE", "/api/delete", json={"model": "alias-copy"})
    assert res_delete.status_code == 200


def test_push_endpoint(client):
    with client.stream("POST", "/api/push", json={"model": "stub"}) as response:
        events = collect_stream(response)
    assert events[-1]["status"] == "success"


def test_blob_endpoints(client, dummy_manager):
    data = b"blob"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    res = client.post(f"/api/blobs/{digest}", content=data)
    assert res.status_code == 201
    head = client.head(f"/api/blobs/{digest}")
    assert head.status_code == 200


def test_unload_endpoint_unloads_model(client, dummy_manager):
    res = client.post("/api/unload", json={"model": "stub"})
    assert res.status_code == 200
    assert dummy_manager.unloaded == ["stub"]


def test_unload_endpoint_updates_keep_alive(client, dummy_manager):
    res = client.post("/api/unload", json={"model": "stub", "keep_alive": "30s"})
    assert res.status_code == 200
    assert dummy_manager.keep_alive_updates["stub"] == 30.0
    assert dummy_manager.unloaded == []


def test_concurrent_generate(client, dummy_manager):
    payload = {"model": "stub", "prompt": "hi", "stream": False}

    def call():
        with TestClient(client.app) as local:
            return local.post("/api/generate", json=payload).status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        statuses = list(executor.map(lambda _: call(), range(10)))
    assert all(status == 200 for status in statuses)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_device_smoke(monkeypatch, device):
    if device != "cpu":
        pytest.skip("Accelerator not available in tests")
    assert device == "cpu"
