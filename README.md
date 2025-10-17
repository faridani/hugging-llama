# hugging-llama

An Ollama-like AI Serving API for Hugging Face Models

<img src="images/logo.png" alt="hugging llama logo" height="300">

Hugging Llama provides a fully local, Ollama-compatible runtime built on top of Hugging Face `transformers`. It includes a FastAPI + Uvicorn server, a CLI, Docker packaging, and comprehensive tests that cover streaming/non-streaming generation, chat, embeddings, model pulls, and concurrency safety.

[![End-to-end tests CI](https://github.com/faridani/hugging-llama/actions/workflows/ci.yml/badge.svg)](https://github.com/faridani/hugging-llama/actions/workflows/ci.yml)
 [![Python Package CI](https://github.com/faridani/hugging-llama/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/faridani/hugging-llama/actions/workflows/python-tests.yml)
[![Docker Build](https://github.com/faridani/hugging-llama/actions/workflows/docker-build.yml/badge.svg)](https://github.com/faridani/hugging-llama/actions/workflows/docker-build.yml) [![Ruff Lint](https://github.com/faridani/hugging-llama/actions/workflows/python-lint.yml/badge.svg?branch=main)](https://github.com/faridani/hugging-llama/actions/workflows/python-lint.yml)
[![Python Type Check](https://github.com/faridani/hugging-llama/actions/workflows/python-typecheck.yml/badge.svg?branch=main)](https://github.com/faridani/hugging-llama/actions/workflows/python-typecheck.yml) [![Coverage](documentation/badges/coverage.svg)](documentation/badges/coverage.svg)

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
- **CLI** – `hugging-llama` supports `serve`, `pull`, `ps`, `embed`, `show`, and `catalog` commands. `serve` exposes `--max-resident-models` and `--model-ttl` for cache tuning while other commands interact with the running API.
- **Testing & Tooling** – Pytest golden tests, JSON schema enforcement, stop-sequence edge cases, tool calling roundtrips, concurrency checks, and accelerator smoke tests. Includes Ruff, Mypy, pre-commit, and Dockerfile.
- **Observability & Ops** – Prometheus metrics (latency, tokens/s, OOMs) and health endpoints. Sample curl scripts in `scripts/` demonstrate API usage.

## Installation

> [!NOTE]  
> 
> For Windows, please use WSL
> 



It is better to setup a conda environment before running this code 

```bash
conda create --name hugging-llama-env python=3.11
conda activate hugging-llama-env

```

Then install the hugging-llama command with 
```bash
pip install .
```

For development (tests + linters):

```bash
pip install .[dev]
```

## Running the Server

You need to start the backend by running the following command

```bash
hugging-llama serve --host 127.0.0.1 --port 11434 --max-resident-models 2 --model-ttl 600
```

**note** if you get a port error, you can simply change the port number to another (see the [FAQ doc](/documentation/faq.md)  )

The server binds to `127.0.0.1` by default. Use `--host 0.0.0.0` in Docker or trusted networks. TLS can be configured through Uvicorn settings.

**advanced users:** To change the default port without updating every command invocation, set the `OLLAMA_LOCAL_PORT`
environment variable. The CLI picks up this value for both `serve` (as the default for
`--port`) and for the default `--url` used by other subcommands.

## CLI Examples

```bash
hugging-llama pull hf-internal-testing/tiny-random-gpt2
hugging-llama pull openai/gpt-oss-20b
hugging-llama ps
hugging-llama embed sentence-transformers/all-MiniLM-L6-v2 "embedding request"
hugging-llama catalog --memory 24GB
```


## API Quickstart

Mac and Linux 

```bash
curl -sS http://127.0.0.1:11434/api/generate -H 'Content-Type: application/json' -d '{"model":"hf-internal-testing/tiny-random-gpt2","prompt":"Hello","stream":false}'
```


Windows PowerShell (we recommend using WLS on Windows instead of the native PowerShell)
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/generate' -Method 'POST' -Headers @{'Content-Type'='application/json'} -Body '{"model":"hf-internal-testing/tiny-random-gpt2","prompt":"Hello","stream":false}'
```


See `scripts/` for additional curl snippets.

## Testing

```bash
pytest
```

Golden files live in `tests/golden/` and cover streaming and non-streaming paths. Tests monkeypatch lightweight stand-ins for transformers so CI runs quickly while still exercising API shapes.

## Docker

```bash
docker build -t hugging-llama .
docker run -it --rm -p 11434:11434 hugging-llama
```

## Setting up Open WebUI
You need to run OpenWebUI in docker and after launching it use the docker's local host instead of 0.0.0.0 ip. ie: `http://host.docker.internal:11434`

## License

Apache 2.0 (see [LICENSE](LICENSE)).
