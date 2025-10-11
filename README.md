# Ollama-Compatible Local Server

This project provides a fully local, Ollama-compatible runtime built on top of Hugging Face `transformers`. It includes a FastAPI + Uvicorn server, an ergonomic CLI, Docker packaging, and comprehensive tests that cover streaming/non-streaming generation, chat, embeddings, model pulls, and concurrency safety.

## Features

- **API Compatibility** – Implements Ollama's endpoints with Hugging Face models:
  - `POST /api/generate`
  - `POST /api/chat`
  - `POST /api/embed`
  - `POST /api/pull`
  - `GET /api/tags`
  - `GET /api/ps`
  - Legacy `GET /api/models`
  - Health (`/health`) and Prometheus metrics (`/metrics`).
- **Advanced Generation Controls** – Presence/frequency penalties via custom logits processors wired through `logits_processor`, stop sequences with boundary-aware handling, top-k/top-p/temperature, repetition penalties, JSON mode validation, and chat templating via `apply_chat_template` (including tool integration).
- **Model Lifecycle Management** – LRU + TTL caching, per-model locks, keep-alive semantics, snapshot-based pulls, VRAM estimates, device auto-detection (CUDA/MPS/CPU), 4-bit/8-bit loading, and `device_map="auto"` gating via accelerate.
- **CLI** – `ollama-local` supports `serve`, `pull`, `ps`, `embed`, and `show` commands. `serve` exposes `--max-resident-models` and `--model-ttl` for cache tuning while other commands interact with the running API.
- **Testing & Tooling** – Pytest golden tests, JSON schema enforcement, stop-sequence edge cases, tool calling roundtrips, concurrency checks, and accelerator smoke tests. Includes Ruff, Mypy, pre-commit, and Dockerfile.
- **Observability & Ops** – Prometheus metrics (latency, tokens/s, OOMs) and health endpoints. Sample curl scripts in `scripts/` demonstrate API usage.

## Installation

```bash
pip install .
```

For development (tests + linters):

```bash
pip install .[dev]
```

## Running the Server

```bash
ollama-local serve --host 127.0.0.1 --port 11434 \
  --max-resident-models 2 --model-ttl 600
```

The server binds to `127.0.0.1` by default. Use `--host 0.0.0.0` in Docker or trusted networks. TLS can be configured through Uvicorn settings.

To change the default port without updating every command invocation, set the `OLLAMA_LOCAL_PORT`
environment variable. The CLI picks up this value for both `serve` (as the default for
`--port`) and for the default `--url` used by other subcommands.

## CLI Examples

```bash
ollama-local pull hf-internal-testing/tiny-random-gpt2
ollama-local ps
ollama-local embed sentence-transformers/all-MiniLM-L6-v2 "embedding request"
```

## API Quickstart

```bash
curl -sS http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"hf-internal-testing/tiny-random-gpt2","prompt":"Hello","stream":false}'
```

See `scripts/` for additional curl snippets.

## Testing

```bash
pytest
```

Golden files live in `tests/golden/` and cover streaming and non-streaming paths. Tests monkeypatch lightweight stand-ins for transformers so CI runs quickly while still exercising API shapes.

## Docker

```bash
docker build -t ollama-local .
docker run -it --rm -p 11434:11434 ollama-local
```

## License

Apache 2.0 (see [LICENSE](LICENSE)).
