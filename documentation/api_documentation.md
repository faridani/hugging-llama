# Hugging Llama API and CLI Reference

This document describes the HTTP endpoints and command line interface (CLI) commands exposed by the Hugging Llama project. The API intentionally mirrors the surface area of Ollama while being powered by Hugging Face models.

## Base URL

The examples below assume the server is running locally on `http://127.0.0.1:11434`. Adjust the host and port to match your deployment.

## HTTP API

### `GET /health`

Returns a simple health report.

- **Description:** Provides service health status and detected accelerator/device information.
- **Example:**
  ```bash
  curl http://127.0.0.1:11434/health
  ```
  ```json
  {
    "status": "ok",
    "device": "cuda"
  }
  ```

### `GET /metrics`

- **Description:** Exposes Prometheus-formatted metrics such as request latency histograms and token counters for observability systems.
- **Example:**
  ```bash
  curl http://127.0.0.1:11434/metrics
  ```
  The response is plaintext that can be scraped by Prometheus.

### `POST /api/generate`

Performs text generation for a single prompt.

- **Description:** Streams or returns a single JSON object containing model output tokens. Supports temperature, top-k/p, repetition penalties, stop sequences, JSON mode validation, and other Hugging Face generation controls via the `options` payload.
- **Request Body:**
  ```json
  {
    "model": "hf-internal-testing/tiny-random-gpt2",
    "prompt": "Hello, world!",
    "stream": false,
    "options": {
      "temperature": 0.7,
      "max_tokens": 64,
      "stop": ["\n"],
      "seed": 42
    },
    "keep_alive": 600
  }
  ```
- **Example Use Case:** Generate a completion for a short prompt, optionally limiting output to a single line via the stop sequence.
- **Response (non-streaming example):**
  ```json
  {
    "model": "hf-internal-testing/tiny-random-gpt2",
    "response": "Hello back!",
    "done": true,
    "prompt_eval_count": 5,
    "eval_count": 10,
    "total_duration": 0.3521
  }
  ```

### `POST /api/chat`

Runs chat-style generation using `transformers` chat templates.

- **Description:** Accepts a list of role-tagged messages and optional tool definitions. Converts the conversation into a prompt with `apply_chat_template` before delegating to `/api/generate`.
- **Request Body:**
  ```json
  {
    "model": "hf-internal-testing/tiny-random-gpt2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize why caching is useful."}
    ],
    "stream": true,
    "format": "json"
  }
  ```
- **Example Use Case:** Stream assistant responses for a chat client that enforces JSON-formatted tool outputs.
- **Streaming Response Chunk:**
  ```json
  {
    "model": "hf-internal-testing/tiny-random-gpt2",
    "message": {
      "role": "assistant",
      "content": "Caching keeps frequently accessed data ready so responses stay fast."
    },
    "done": true,
    "total_duration": 0.4182
  }
  ```

### `POST /api/embed`

Computes embedding vectors for text inputs.

- **Description:** Loads or reuses an embedding-capable model and returns vectors for a single string or list of strings.
- **Request Body:**
  ```json
  {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [
      "vectorize this",
      "and this"
    ],
    "keep_alive": "10m"
  }
  ```
- **Example Use Case:** Produce embeddings for semantic search or similarity scoring without reloading the model between requests.
- **Response:**
  ```json
  {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "embeddings": [[0.01, 0.02, ...], [0.05, 0.06, ...]],
    "total_duration": 0.2274
  }
  ```

### `POST /api/pull`

Downloads and caches a Hugging Face model snapshot.

- **Description:** Streams status updates while fetching model files into the server cache. Supports specifying a revision and enabling `trust_remote_code` for repositories that require it.
- **Request Body:**
  ```json
  {
    "model": "hf-internal-testing/tiny-random-gpt2",
    "revision": null,
    "trust_remote_code": false
  }
  ```
- **Example Use Case:** Warm up the cache before serving traffic; the client can show progress updates as JSON lines.
- **Streaming Response Sequence:**
  ```json
  {"status": "pulling manifest", "digest": ""}
  {"status": "downloading", "path": "/home/user/.cache/hugging-llama/..."}
  {"status": "success"}
  ```

### `GET /api/tags`

Lists cached models and metadata.

- **Description:** Scans the cache directory, returning available model names, their total disk usage, modification timestamps, and placeholder metadata fields.
- **Example:**
  ```bash
  curl http://127.0.0.1:11434/api/tags
  ```
- **Use Case:** Display locally available models in a UI or CLI without hitting remote registries.

### `GET /api/ps`

Reports models currently resident in memory.

- **Description:** Shows reference counts, TTL expiration times, and extra details for each loaded model in the `ModelManager`.
- **Example:**
  ```bash
  curl http://127.0.0.1:11434/api/ps | jq
  ```
- **Use Case:** Monitor which models are actively loaded when tuning cache limits.

### `GET /api/models` (deprecated)

- **Description:** Legacy alias for `/api/tags`; maintained for backwards compatibility.
- **Example:**
  ```bash
  curl http://127.0.0.1:11434/api/models
  ```

## Command Line Interface

Install the package (`pip install .`) to expose the `hugging-llama` CLI. All commands accept `--url` to point at a remote server when applicable. The default port honors the `OLLAMA_LOCAL_PORT` environment variable.

### `hugging-llama serve`

- **Description:** Runs the FastAPI + Uvicorn server.
- **Key Options:**
  - `--host`: Bind address (default `127.0.0.1`).
  - `--port`: Listening port (defaults to `OLLAMA_LOCAL_PORT` or 11434).
  - `--max-resident-models`: Maximum number of models to keep loaded simultaneously.
  - `--model-ttl`: Time-to-live in seconds before unused models are unloaded.
- **Example Use Case:**
  ```bash
  hugging-llama serve --host 0.0.0.0 --port 11434 --max-resident-models 4 --model-ttl 900
  ```

### `hugging-llama pull`

- **Description:** Instructs a running server to download and cache a model.
- **Key Options:**
  - `--revision`: Specific Hugging Face revision or commit.
  - `--trust-remote-code`: Allow custom code execution from the repository.
- **Example Use Case:**
  ```bash
  hugging-llama pull hf-internal-testing/tiny-random-gpt2
  ```

### `hugging-llama ps`

- **Description:** Displays the current set of loaded models from the server's `/api/ps` endpoint.
- **Example Use Case:**
  ```bash
  hugging-llama ps
  # Output: hf-internal-testing/tiny-random-gpt2: refs=1 expires=2024-05-08T12:00:00Z
  ```

### `hugging-llama embed`

- **Description:** Sends an embedding request and prints the JSON response.
- **Example Use Case:**
  ```bash
  hugging-llama embed sentence-transformers/all-MiniLM-L6-v2 "embedding request"
  ```

### `hugging-llama show`

- **Description:** Fetches `/api/tags`, locates a specific model, and renders its metadata.
- **Example Use Case:**
  ```bash
  hugging-llama show hf-internal-testing/tiny-random-gpt2
  ```
  Exits with a non-zero status if the model is not present locally.

### `hugging-llama catalog`

- **Description:** Prints a curated list of models with estimated VRAM requirements filtered by the detected GPU or a supplied limit.
- **Key Options:**
  - `--memory`: Override the detected GPU memory (for example, `--memory 24GB`).
  - `--all`: Bypass detection/overrides and display the entire catalog.
- **Example Use Case:**
  ```bash
  hugging-llama catalog --memory 24GB
  # Outputs a table of models whose VRAM estimates are <= 24GB
  ```

## Environment Variables

- **`OLLAMA_LOCAL_PORT`** — When set, both the server (`serve`) and client subcommands use this as the default port, simplifying configuration across multiple commands.

## Related Resources

Additional curl examples live in the `scripts/` directory, and unit tests under `tests/` cover streaming, embeddings, tool calling, and concurrency behaviors.
