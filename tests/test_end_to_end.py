from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient


def test_service_end_to_end_flow(e2e_client: TestClient, tmp_path: Path) -> None:
    client = e2e_client
    manager = client.app.state.test_manager

    assert asyncio.run(manager.models.snapshot()) == {}

    root_response = client.get("/")
    assert root_response.status_code == 200
    assert root_response.text == "Ollama is running"

    model_name = "e2e/stub"

    pull_response = client.post("/api/pull", json={"model": model_name, "stream": False})
    assert pull_response.status_code == 200
    assert pull_response.json() == {"status": "success"}

    repo_dir = tmp_path / model_name.replace("/", "__")
    assert repo_dir.exists()

    generate_response = client.post(
        "/api/generate",
        json={"model": model_name, "prompt": "Hello", "stream": False},
    )
    assert generate_response.status_code == 200
    payload = generate_response.json()
    assert payload["response"] == "hello world!"
    assert payload["done"] is True

    ps_response = client.get("/api/ps")
    assert ps_response.status_code == 200
    ps_payload = ps_response.json()
    assert len(ps_payload["models"]) == 1
    assert ps_payload["models"][0]["name"] == model_name
    assert ps_payload["models"][0]["ref_count"] == 0

    snapshot_after_generate = asyncio.run(manager.models.snapshot())
    assert model_name in snapshot_after_generate

    unload_response = client.post("/api/unload", json={"model": model_name})
    assert unload_response.status_code == 200
    assert unload_response.json() == {"status": "success"}

    ps_after_unload = client.get("/api/ps")
    assert ps_after_unload.status_code == 200
    assert ps_after_unload.json()["models"] == []

    snapshot_after_unload = asyncio.run(manager.models.snapshot())
    assert snapshot_after_unload == {}
