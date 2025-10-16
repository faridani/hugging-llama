"""End-to-end service lifecycle tests."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hugging_llama.server import create_app


class _FakeTensor:
    def __init__(self, data: list[int]) -> None:
        self.data = data

    def to(self, device: str) -> "_FakeTensor":
        del device
        return self

    def size(self, dim: int) -> int:
        if dim != 0:
            raise ValueError("_FakeTensor only supports dim=0")
        return len(self.data)

    def __getitem__(self, index: int) -> "_FakeTensor":
        if index != 0:
            raise IndexError(index)
        return _FakeTensor(self.data)


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> dict[str, _FakeTensor]:
        del return_tensors, add_special_tokens
        tokens = list(range(len(text.split()))) or [0]
        return {"input_ids": _FakeTensor(tokens)}

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("chat templates not required in tests")


class _FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.device = "cpu"


class LifecycleManager:
    """A lightweight stateful stand-in for ``ModelManager``.

    The existing unit tests patch :func:`hugging_llama.server.create_app` to use a
    ``DummyManager``.  These integration-oriented tests need more behavioural
    coverage, so this helper records lifecycle activity while mimicking the
    public methods the API server exercises during pulls, generation, and
    unloading.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pulled_paths: dict[str, Path] = {}
        self.loaded_state: dict[str, dict[str, Any]] = {}
        self.ensure_calls: list[tuple[str, Any, float | None]] = []
        self.release_calls: list[tuple[str, float | None]] = []
        self.embed_requests: list[list[str]] = []
        self.unloaded: list[str] = []
        self.keep_alive_updates: dict[str, float | None] = {}
        self.aliases: dict[str, dict[str, Any]] = {}
        self.predefined_prompt_aliases = {"default": ""}
        self.blobs: dict[str, bytes] = {}

    async def pull(self, name: str, revision: str | None, trust_remote_code: bool) -> Path:
        del revision, trust_remote_code
        path = self.cache_dir / name.replace("/", "__")
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "MODEL"
        marker.write_text("mock", encoding="utf-8")
        self.pulled_paths[name] = path
        return path

    async def ensure_model(self, name: str, options: Any, ttl: float | None) -> Any:
        record = self.loaded_state.setdefault(
            name,
            {"ref_count": 0, "expires_at": None, "details": {"path": ""}},
        )
        record["ref_count"] += 1
        record["details"] = {"path": str(self.pulled_paths.get(name, ""))}
        self.ensure_calls.append((name, options, ttl))
        return SimpleNamespace(model=_FakeModel(["hello", " world"]), tokenizer=_FakeTokenizer(), kind="generate")

    async def ensure_embeddings_model(self, name: str, ttl: float | None) -> Any:
        del ttl

        manager = self

        class Embedder:
            def __init__(self, vector: list[float]) -> None:
                self._vector = vector

            class _Vector:
                def __init__(self, data: list[float]) -> None:
                    self._data = data

                def tolist(self) -> list[float]:
                    return list(self._data)

            def encode(self, inputs: list[str]) -> list[Embedder._Vector]:
                manager.embed_requests.append([str(item) for item in inputs])
                return [self._Vector(self._vector) for _ in inputs]

        embedder = Embedder([0.1, 0.2, 0.3])
        return SimpleNamespace(model=embedder, tokenizer=None, kind="embedding")

    async def release(self, name: str, ttl: float | None) -> None:
        record = self.loaded_state.setdefault(
            name,
            {"ref_count": 0, "expires_at": None, "details": {"path": ""}},
        )
        record["ref_count"] = max(0, record["ref_count"] - 1)
        self.release_calls.append((name, ttl))

    async def list_loaded(self) -> dict[str, Any]:
        return {
            name: {
                "ref_count": state["ref_count"],
                "expires_at": state.get("expires_at"),
                "details": state.get("details", {}),
            }
            for name, state in self.loaded_state.items()
        }

    def get_alias(self, name: str) -> dict[str, Any] | None:
        return self.aliases.get(name)

    def get_prompt_aliases(self, name: str) -> dict[str, str]:
        del name
        return dict(self.predefined_prompt_aliases)

    def describe_model(self, name: str) -> dict[str, Any] | None:
        alias = self.aliases.get(name)
        if alias is None and name not in self.pulled_paths:
            return None
        return {
            "name": name,
            "model": name,
            "options": {},
            "metadata": {},
        }

    def copy_model(self, source: str, destination: str) -> bool:
        if source not in self.aliases:
            return False
        self.aliases[destination] = dict(self.aliases[source], name=destination)
        return True

    def delete_model(self, name: str) -> bool:
        return self.aliases.pop(name, None) is not None

    def blob_exists(self, digest: str) -> bool:
        return digest in self.blobs

    def save_blob(self, digest: str, data: bytes) -> Path:
        if not data:
            raise ValueError("Empty payload is not allowed")
        self.blobs[digest] = data
        return self.cache_dir

    async def unload(self, name: str) -> None:
        self.unloaded.append(name)
        state = self.loaded_state.setdefault(
            name,
            {"ref_count": 0, "expires_at": None, "details": {"path": ""}},
        )
        state["ref_count"] = 0

    async def set_keep_alive(self, name: str, ttl: float | None) -> None:
        self.keep_alive_updates[name] = ttl


@pytest.fixture()
def lifecycle_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, LifecycleManager]:
    manager = LifecycleManager(tmp_path)
    monkeypatch.setattr("hugging_llama.server.ModelManager", lambda *args, **kwargs: manager)
    app = create_app(cache_dir=tmp_path, max_resident_models=1, default_ttl=30)
    return TestClient(app), manager


def _collect_stream(response: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in response.iter_lines():
        if not line:
            continue
        events.append(json.loads(line))
    return events


def test_service_lifecycle_end_to_end(lifecycle_client: tuple[TestClient, LifecycleManager]) -> None:
    client, manager = lifecycle_client

    root = client.get("/")
    assert root.status_code == 200
    assert root.text == "Ollama is running"

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["device"]

    with client.stream("POST", "/api/pull", json={"model": "demo/e2e"}) as response:
        events = _collect_stream(response)

    statuses = [event.get("status") for event in events]
    assert statuses == ["pulling manifest", "downloading", "success"]
    assert "demo/e2e" in manager.pulled_paths
    assert manager.pulled_paths["demo/e2e"].exists()

    generate = client.post(
        "/api/generate",
        json={"model": "demo/e2e", "prompt": "hello", "stream": False},
    )
    assert generate.status_code == 200
    result = generate.json()
    assert result["model"] == "demo/e2e"
    assert result["response"] == "hello world"
    assert manager.ensure_calls and manager.ensure_calls[-1][0] == "demo/e2e"
    assert manager.release_calls and manager.release_calls[-1][0] == "demo/e2e"

    embeddings = client.post(
        "/api/embed",
        json={"model": "demo/e2e", "input": ["request"]},
    )
    assert embeddings.status_code == 200
    assert manager.embed_requests and manager.embed_requests[-1] == ["request"]

    ps = client.get("/api/ps")
    assert ps.status_code == 200
    processes = ps.json()["models"]
    assert any(model["name"] == "demo/e2e" for model in processes)

    keep_alive = client.post(
        "/api/unload",
        json={"model": "demo/e2e", "keep_alive": "30s"},
    )
    assert keep_alive.status_code == 200
    assert manager.keep_alive_updates["demo/e2e"] == 30.0
    assert manager.unloaded == []

    unload = client.post("/api/unload", json={"model": "demo/e2e", "keep_alive": 0})
    assert unload.status_code == 200
    assert manager.unloaded == ["demo/e2e"]
    assert manager.loaded_state["demo/e2e"]["ref_count"] == 0
