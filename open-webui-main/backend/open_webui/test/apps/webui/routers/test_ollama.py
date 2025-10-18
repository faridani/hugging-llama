import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/openwebui-test.db")


def test_get_ollama_url_single_node_fallback():
    from open_webui.routers import ollama

    request = SimpleNamespace()
    request.app = SimpleNamespace()
    request.app.state = SimpleNamespace()
    request.app.state.config = SimpleNamespace(
        ENABLE_OLLAMA_API=False,
        OLLAMA_BASE_URLS=["http://127.0.0.1:11434"],
        OLLAMA_API_CONFIGS={},
    )
    request.app.state.OLLAMA_MODELS = {}

    url, idx = asyncio.run(
        ollama.get_ollama_url(
            request,
            "hf-internal-testing/tiny-random-gpt2",
            None,
            None,
        )
    )

    assert url == "http://127.0.0.1:11434"
    assert idx == 0
