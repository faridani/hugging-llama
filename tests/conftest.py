from __future__ import annotations

import sys
import types
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


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
    def __init__(self, *args, **kwargs):
        pass


class DummyAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("not used in tests")


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("not used in tests")


fake_transformers.TextIteratorStreamer = DummyStreamer
fake_transformers.AutoModelForCausalLM = DummyAutoModel
fake_transformers.AutoModelForSeq2SeqLM = DummyAutoModel
fake_transformers.AutoTokenizer = DummyTokenizer
fake_transformers.LogitsProcessor = type("LogitsProcessor", (), {})
sys.modules.setdefault("transformers", fake_transformers)


fake_accelerate = types.ModuleType("accelerate")


def _infer_auto_device_map(*args: Any, **kwargs: Any) -> Dict[str, str]:
    return {}


fake_accelerate.infer_auto_device_map = _infer_auto_device_map
sys.modules.setdefault("accelerate", fake_accelerate)


fake_huggingface = types.ModuleType("huggingface_hub")


def _snapshot_download(*args: Any, **kwargs: Any) -> Path:
    return Path("/tmp/mock")


fake_huggingface.snapshot_download = _snapshot_download
sys.modules.setdefault("huggingface_hub", fake_huggingface)

import pytest
from fastapi.testclient import TestClient

from hugging_llama.server import create_app


class FakeTensor:
    def __init__(self, data: List[int]):
        self.data = data

    def to(self, device: str) -> "FakeTensor":
        return self

    def size(self, dim: int) -> int:
        if dim != 0:
            raise ValueError("FakeTensor only supports dim=0")
        return len(self.data)

    def __getitem__(self, idx: int) -> "FakeTensor":
        if idx != 0:
            raise IndexError(idx)
        return FakeTensor(self.data)


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True):
        tokens = list(range(len(text.split()))) or [0]
        return {"input_ids": FakeTensor(tokens)}

    def apply_chat_template(self, messages: Iterable[Dict[str, Any]], add_generation_prompt: bool = True, tokenize: bool = False, **kwargs: Any) -> str:
        chunks = []
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

    def __iter__(self) -> "FakeStreamer":
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
        self.generation_outputs: List[str] = ["hello", " world", "!"]
        self.embed_vector = [0.1, 0.2, 0.3]
        self.cache_dir = Path("./tmp-cache")

    async def ensure_model(self, name: str, options: Any, ttl: Any) -> Any:
        return type("Managed", (), {"model": FakeModel(self.generation_outputs), "tokenizer": FakeTokenizer(), "kind": "generate"})()

    async def release(self, name: str, ttl: Any) -> None:
        return None

    async def ensure_embeddings_model(self, name: str, ttl: Any) -> Any:
        vector = self.embed_vector

        class Embedder:
            def __init__(self, vec: List[float]):
                self.vec = vec

            class _Vector:
                def __init__(self, data: List[float]):
                    self._data = data

                def tolist(self) -> List[float]:
                    return list(self._data)

            def encode(self, inputs: Iterable[str]) -> List["Embedder._Vector"]:
                return [self._Vector(self.vec) for _ in inputs]

        return type("Embeddings", (), {"model": Embedder(vector), "tokenizer": None, "kind": "embedding"})()

    async def list_loaded(self) -> Dict[str, Any]:
        return {"test": {"ref_count": 0, "expires_at": None, "details": {}}}

    async def pull(self, name: str, revision: str, trust_remote_code: bool) -> Path:
        return Path("/tmp/mock")


def fake_run_generation(manager: Any, request_options: Any, input_ids: Any, tokenizer: Any, model: Any, prompt_text: str, streamer: Any) -> Dict[str, Any]:
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
    app = create_app(cache_dir=Path("./tmp-cache"))
    return TestClient(app)
