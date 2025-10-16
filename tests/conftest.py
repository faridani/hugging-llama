"""Shared pytest fixtures for the test suite."""
from __future__ import annotations

import sys
import tempfile
import types
from collections.abc import Iterable
from pathlib import Path
from queue import Queue
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

MOCK_CACHE_DIR = Path(tempfile.gettempdir()) / "hugging-llama-mock"


def _install_module_shims() -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
        is_bf16_supported=lambda: False,
    )
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    fake_torch.manual_seed = lambda seed: None
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    sys.modules.setdefault("torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")

    class DummyStreamer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class DummyAutoModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("not used in tests")

    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("not used in tests")

    fake_transformers.TextIteratorStreamer = DummyStreamer
    fake_transformers.AutoModelForCausalLM = DummyAutoModel
    fake_transformers.AutoModelForSeq2SeqLM = DummyAutoModel
    fake_transformers.AutoTokenizer = DummyTokenizer
    fake_transformers.LogitsProcessor = type("LogitsProcessor", (), {})
    sys.modules.setdefault("transformers", fake_transformers)

    fake_accelerate = types.ModuleType("accelerate")

    def _infer_auto_device_map(*args: Any, **kwargs: Any) -> dict[str, str]:
        return {}

    fake_accelerate.infer_auto_device_map = _infer_auto_device_map
    sys.modules.setdefault("accelerate", fake_accelerate)

    fake_huggingface = types.ModuleType("huggingface_hub")

    def _snapshot_download(*args: Any, **kwargs: Any) -> Path:
        return MOCK_CACHE_DIR

    fake_huggingface.snapshot_download = _snapshot_download
    sys.modules.setdefault("huggingface_hub", fake_huggingface)


_install_module_shims()

from hugging_llama.server import create_app  # noqa: E402


class FakeTensor:
    def __init__(self, data: list[int]):
        self.data = data

    def to(self, device: str) -> FakeTensor:
        return self

    def size(self, dim: int) -> int:
        if dim != 0:
            raise ValueError("FakeTensor only supports dim=0")
        return len(self.data)

    def __getitem__(self, idx: int) -> FakeTensor:
        if idx != 0:
            raise IndexError(idx)
        return FakeTensor(self.data)


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> dict[str, FakeTensor]:
        tokens = list(range(len(text.split()))) or [0]
        return {"input_ids": FakeTensor(tokens)}

    def apply_chat_template(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> str:
        del tokenize, kwargs
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"{message['role']}: {message['content']}")
        if add_generation_prompt:
            chunks.append("assistant:")
        return "\n".join(chunks)


class FakeStreamer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: Queue[str | None] = Queue()

    def feed(self, text: str) -> None:
        self.events.put(text)

    def end(self) -> None:
        self.events.put(None)

    def __iter__(self) -> FakeStreamer:
        return self

    def __next__(self) -> str:
        item = self.events.get()
        if item is None:
            raise StopIteration
        return item


class FakeModel:
    def __init__(self, outputs: Iterable[str]):
        self.outputs = list(outputs)
        self.device = "cpu"


class DummyManager:
    def __init__(self) -> None:
        self.generation_outputs: list[str] = ["hello", " world", "!"]
        self.embed_vector = [0.1, 0.2, 0.3]
        self.cache_dir = MOCK_CACHE_DIR
        self.aliases: dict[str, dict[str, Any]] = {}
        self.blobs: dict[str, bytes] = {}
        self.keep_alive_updates: dict[str, float | None] = {}
        self.unloaded: list[str] = []

    async def ensure_model(self, name: str, options: Any, ttl: Any) -> Any:
        return types.SimpleNamespace(
            model=FakeModel(self.generation_outputs),
            tokenizer=FakeTokenizer(),
            kind="generate",
        )

    async def release(self, name: str, ttl: Any) -> None:
        return None

    async def ensure_embeddings_model(self, name: str, ttl: Any) -> Any:
        vector = self.embed_vector

        class Embedder:
            def __init__(self, vec: list[float]):
                self.vec = vec

            class _Vector:
                def __init__(self, data: list[float]):
                    self._data = data

                def tolist(self) -> list[float]:
                    return list(self._data)

            def encode(self, inputs: Iterable[str]) -> list[Embedder._Vector]:
                return [self._Vector(self.vec) for _ in inputs]

        return types.SimpleNamespace(model=Embedder(vector), tokenizer=None, kind="embedding")

    async def list_loaded(self) -> dict[str, Any]:
        return {"test": {"ref_count": 0, "expires_at": None, "details": {}}}

    async def pull(self, name: str, revision: str, trust_remote_code: bool) -> Path:
        return MOCK_CACHE_DIR

    def create_alias(
        self,
        name: str,
        base_model: str | None,
        template: str | None,
        system: str | None,
        parameters: dict[str, Any] | None,
        modelfile: str | None,
        license_info: list[str] | str | None,
        messages: list[dict[str, Any]] | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        alias = {
            "name": name,
            "model": base_model or name,
            "template": template,
            "system": system,
            "options": parameters or {},
            "modelfile": modelfile,
            "license": license_info,
            "messages": messages,
            "metadata": metadata or {},
            "details": {},
            "modified_at": "now",
        }
        self.aliases[name] = alias
        return alias

    def get_alias(self, name: str) -> dict[str, Any] | None:
        return self.aliases.get(name)

    def list_alias_records(self) -> list[dict[str, Any]]:
        return [dict(alias, digest="dummy", size=0) for alias in self.aliases.values()]

    def describe_model(self, name: str) -> dict[str, Any] | None:
        alias = self.aliases.get(name)
        if alias is None:
            return None
        return dict(alias, parameters=alias.get("options", {}))

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
        return MOCK_CACHE_DIR

    async def unload(self, name: str) -> None:
        self.unloaded.append(name)

    async def set_keep_alive(self, name: str, ttl: float | None) -> None:
        self.keep_alive_updates[name] = ttl


def fake_run_generation(
    manager: Any,
    request_options: Any,
    input_ids: Any,
    tokenizer: Any,
    model: Any,
    prompt_text: str,
    streamer: Any,
) -> dict[str, Any]:
    del manager, request_options, input_ids, tokenizer, prompt_text
    for chunk in model.outputs:
        streamer.feed(chunk)
    streamer.end()
    return {"prompt_tokens": 3}


@pytest.fixture()
def dummy_manager() -> DummyManager:
    return DummyManager()


@pytest.fixture(autouse=True)
def patch_components(monkeypatch: pytest.MonkeyPatch, dummy_manager: DummyManager) -> None:
    monkeypatch.setattr("hugging_llama.server.TextIteratorStreamer", FakeStreamer)
    monkeypatch.setattr("hugging_llama.server.run_generation", fake_run_generation)
    monkeypatch.setattr("hugging_llama.server.ModelManager", lambda *args, **kwargs: dummy_manager)


@pytest.fixture()
def client() -> TestClient:
    app = create_app(cache_dir=MOCK_CACHE_DIR)
    return TestClient(app)


@pytest.fixture()
def e2e_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    from hugging_llama import model_manager as mm
    from hugging_llama import server as server_module
    import torch

    monkeypatch.setattr(server_module, "ModelManager", mm.ModelManager)

    created_managers: list[mm.ModelManager] = []
    original_init = mm.ModelManager.__init__

    def recording_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        created_managers.append(self)

    monkeypatch.setattr(mm.ModelManager, "__init__", recording_init)

    def fake_load(self: mm.ModelManager, name: str, options: dict[str, Any]) -> mm.ManagedModel:
        del options
        repo_dir = self.cache_dir / name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = FakeTokenizer()
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        return mm.ManagedModel(
            model=FakeModel(["hello", " world", "!"]),
            tokenizer=tokenizer,
            kind="generate",
            path=repo_dir,
            device="cpu",
            dtype=getattr(torch, "float32"),
            details={"family": "test", "parameter_size": 0, "format": "mock"},
        )

    monkeypatch.setattr(mm.ModelManager, "_load_text_model", fake_load, raising=False)

    app = server_module.create_app(cache_dir=tmp_path)
    assert created_managers, "ModelManager was not instantiated"
    manager = created_managers[0]
    app.state.test_manager = manager

    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
