from __future__ import annotations

import hashlib
import json

from hugging_llama.api_types import PullRequest


def test_openwebui_create_and_show(client, dummy_manager):
    modelfile = "FROM stub\nPARAMETER temperature 0.3"
    with client.stream(
        "POST",
        "/api/create",
        json={
            "model": "alias",
            "modelfile": modelfile,
            "stream": True,
            "metadata": {
                "description": "Alias",
                "prompt_aliases": {"greet": "Hello"},
            },
        },
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]
    assert events[-1]["status"] == "success"

    info = client.post("/api/show", json={"model": "alias"}).json()
    assert info["model"] == "stub"
    assert "FROM stub" in info["modelfile"]
    assert info["metadata"]["prompt_aliases"]["greet"] == "Hello"


def test_openwebui_copy_delete_flow(client, dummy_manager):
    dummy_manager.create_alias("alias", "stub", None, None, {}, None, None, None, None)
    res = client.post("/api/copy", json={"source": "alias", "destination": "alias-copy"})
    assert res.status_code == 200
    assert "alias-copy" in dummy_manager.aliases

    delete = client.request("DELETE", "/api/delete", json={"model": "alias-copy"})
    assert delete.status_code == 200


def test_openwebui_generate_uses_alias(client, dummy_manager):
    dummy_manager.create_alias(
        "alias",
        "stub",
        template="{{system}}\n\n{{prompt}}",
        system="system message",
        parameters={"temperature": 0.1},
        modelfile=None,
        license_info=None,
        messages=None,
        metadata=None,
    )
    res = client.post(
        "/api/generate",
        json={"model": "alias", "prompt": "hi", "stream": False},
    )
    assert res.status_code == 200
    assert res.json()["model"] == "alias"


def test_openwebui_embeddings_alias(client):
    res = client.post("/api/embeddings", json={"model": "stub", "input": "text"})
    assert res.status_code == 200
    body = res.json()
    assert "embeddings" in body
    assert body["model"] == "stub"


def test_openwebui_embeddings_alias_resolution(client, dummy_manager):
    dummy_manager.create_alias(
        "alias",
        "stub",
        template=None,
        system=None,
        parameters={},
        modelfile=None,
        license_info=None,
        messages=None,
        metadata={"description": "", "prompt_aliases": {"greet": "Hello there"}},
    )
    res = client.post("/api/embeddings", json={"model": "alias", "input": "alias:greet"})
    assert res.status_code == 200
    assert dummy_manager.last_embedding_inputs == ["Hello there"]


def test_openwebui_blob_lifecycle(client, dummy_manager):
    payload = b"blob"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    upload = client.post(f"/api/blobs/{digest}", content=payload)
    assert upload.status_code == 201
    head = client.head(f"/api/blobs/{digest}")
    assert head.status_code == 200


def test_openwebui_unload(client, dummy_manager):
    res = client.post("/api/unload", json={"model": "stub"})
    assert res.status_code == 200
    assert dummy_manager.unloaded == ["stub"]


def test_pull_request_alias_from_bytes_payload():
    body = json.dumps({"name": "alias"}).encode()
    request = PullRequest.model_validate_json(body)
    assert request.model == "alias"
