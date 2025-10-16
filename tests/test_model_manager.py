from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

import hugging_llama.model_manager as model_manager_module
from hugging_llama.api_types import GenerateOptions
from hugging_llama.model_manager import ManagedModel, ModelManager


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


def test_pull_handles_missing_trust_remote_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = ModelManager(tmp_path, max_resident_models=1)

        download_kwargs: dict[str, Any] = {}

        def fake_snapshot_download(name: str, **kwargs: Any) -> None:
            download_kwargs.update(kwargs)
            assert name == "some/model"

        monkeypatch.setattr(
            model_manager_module,
            "_SNAPSHOT_DOWNLOAD_SUPPORTS_TRUST_REMOTE_CODE",
            False,
            raising=False,
        )
        monkeypatch.setattr(model_manager_module, "snapshot_download", fake_snapshot_download)

        path = await manager.pull("some/model", revision="main", trust_remote_code=True)
        assert path == tmp_path / "some__model"
        assert "trust_remote_code" not in download_kwargs
        assert download_kwargs["revision"] == "main"
        assert download_kwargs["local_dir"] == tmp_path / "some__model"

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


def test_modelfile_persistence(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, max_resident_models=1)
    text = "FROM base\nPARAMETER temperature 0.5"
    manager.create_alias(
        "alias",
        base_model="base",
        template=None,
        system=None,
        parameters={"temperature": 0.5},
        modelfile=text,
        license_info=None,
        messages=None,
        metadata={"description": "Alias", "prompt_aliases": {"hello": "Hello"}},
    )
    path = manager._modelfile_path("alias")  # noqa: SLF001
    assert path.exists()
    stored = path.read_text(encoding="utf-8")
    assert "FROM base" in stored
    assert "PARAMETER temperature 0.5" in stored

    manager.create_alias(
        "alias",
        base_model="base",
        template="{{prompt}}",
        system="system",
        parameters={"temperature": 0.5},
        modelfile=None,
        license_info=None,
        messages=None,
        metadata={"description": "Alias", "prompt_aliases": {"hello": "Hello"}},
    )
    rebuilt = path.read_text(encoding="utf-8")
    assert "FROM base" in rebuilt
    assert "METADATA" in rebuilt


def test_get_prompt_aliases_includes_defaults(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, max_resident_models=1)
    manager.create_alias(
        "alias",
        base_model="base",
        template=None,
        system=None,
        parameters={},
        modelfile=None,
        license_info=None,
        messages=None,
        metadata={"description": "Alias", "prompt_aliases": {"hello": "Hello"}},
    )
    aliases = manager.get_prompt_aliases("alias")
    assert "default" in aliases
    assert aliases["hello"] == "Hello"


def test_create_alias_rejects_invalid_metadata(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, max_resident_models=1)
    with pytest.raises(ValueError):
        manager.create_alias(
            "alias",
            base_model="base",
            template=None,
            system=None,
            parameters={},
            modelfile=None,
            license_info=None,
            messages=None,
            metadata={"description": 1},
        )
