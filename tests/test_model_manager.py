import asyncio
import time
from pathlib import Path

import pytest

from ollama_local.api_types import GenerateOptions
from ollama_local.model_manager import ManagedModel, ModelManager


def test_ensure_model_respects_zero_ttl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = ModelManager(tmp_path, max_resident_models=1, default_ttl=60.0)
        managed = ManagedModel(
            model=object(),
            tokenizer=object(),
            kind="generate",
            path=tmp_path,
            device="cpu",
            dtype="float32",
        )

        monkeypatch.setattr(ModelManager, "_load_text_model", lambda self, name, options: managed)

        await manager.ensure_model("foo", GenerateOptions(), ttl=0)

        snapshot = await manager.models.snapshot()
        entry = snapshot["foo"]
        assert entry.expires_at is not None
        assert entry.expires_at <= time.time()

        await manager.release("foo", ttl=0)
        snapshot_after = await manager.models.snapshot()
        assert "foo" not in snapshot_after

    asyncio.run(scenario())


def test_embeddings_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = ModelManager(tmp_path, max_resident_models=1)
        managed = ManagedModel(
            model=object(),
            tokenizer=None,
            kind="embedding",
            path=tmp_path,
            device="cpu",
            dtype="float32",
        )

        manager.aliases["alias"] = {"model": "actual", "options": {}}

        def _load(self: ModelManager, name: str) -> ManagedModel:
            assert name == "actual"
            return managed

        monkeypatch.setattr(ModelManager, "_load_embeddings_model", _load)

        result = await manager.ensure_embeddings_model("alias", ttl=None)
        assert result is managed

        snapshot = await manager.models.snapshot()
        assert "actual" in snapshot
        assert snapshot["actual"].ref_count == 1

        await manager.release("alias", ttl=None)
        snapshot_after = await manager.models.snapshot()
        assert snapshot_after["actual"].ref_count == 0

    asyncio.run(scenario())
