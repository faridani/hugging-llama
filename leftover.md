# Undocumented Interfaces

This repository exposes additional CLI options and API endpoints beyond the ones covered in `README.md`. The lists below summarize those extra surfaces so you can discover their purpose without reading through the implementation.

## CLI Options Not Covered in the README

Although every subcommand is mentioned in the README, several useful flags are only visible in the source:

- Global option `--url`: lets non-`serve` commands target a different server instance; defaults to `http://127.0.0.1` plus the port selected from `OLLAMA_LOCAL_PORT` if set. 【F:src/hugging_llama/cli.py†L271-L280】
- `serve` flags:
  - `--host`: bind address for the API server. 【F:src/hugging_llama/cli.py†L283-L293】
  - `--port`: overrides the default/`OLLAMA_LOCAL_PORT` derived port. 【F:src/hugging_llama/cli.py†L283-L293】
  - `--max-resident-models`: controls the in-memory model cache size. 【F:src/hugging_llama/cli.py†L283-L293】
  - `--model-ttl`: sets the keep-alive timeout for cached models. 【F:src/hugging_llama/cli.py†L283-L293】
- `pull` flags `--revision` and `--trust-remote-code`: pull a specific revision and opt into repositories that require remote code execution. 【F:src/hugging_llama/cli.py†L294-L298】
- `embed` requires a positional `text` argument to embed, in addition to the model name. 【F:src/hugging_llama/cli.py†L301-L303】
- `catalog` supports:
  - `--memory SIZE`: filter catalog entries to those that fit into the requested GPU memory (accepts values such as `24GB` or `24576MB`). 【F:src/hugging_llama/cli.py†L308-L321】
  - `--all`: bypass memory-based filtering and display the entire catalog. 【F:src/hugging_llama/cli.py†L308-L321】

## API Endpoints Missing from the README

The FastAPI application defines several Ollama-compatible endpoints that are not listed in the README's feature summary:

- `GET /`: Plain-text readiness probe that returns "Ollama is running". Useful for simple health checks. 【F:src/hugging_llama/server.py†L98-L107】
- `GET /api/version`: Reports the installed `hugging-llama` package version. 【F:src/hugging_llama/server.py†L109-L112】
- `POST /api/embeddings`: Alias of `/api/embed`; accepts the same payload to compute embedding vectors, but keeps the pluralized endpoint Ollama clients expect. 【F:src/hugging_llama/server.py†L441-L453】
- `POST /api/create`: Creates or updates a local model alias, optionally parsing a Modelfile to merge parameters, metadata, system prompts, templates, adapters, and license information. Streams status updates when `stream` is true. 【F:src/hugging_llama/server.py†L455-L519】【F:src/hugging_llama/api_types.py†L134-L149】
- `POST /api/show`: Returns the stored configuration and metadata for a local model alias; responds `404` if it does not exist. 【F:src/hugging_llama/server.py†L521-L529】
- `POST /api/edit`: Modifies an existing alias by merging request fields, Modelfile contents, and previously stored metadata before re-registering the alias. Validates metadata and reports `404` for unknown models. 【F:src/hugging_llama/server.py†L531-L592】
- `POST /api/copy`: Copies a local model or alias to a new name. Returns `404` when the source is missing. 【F:src/hugging_llama/server.py†L594-L601】
- `DELETE /api/delete`: Removes a stored model or alias; responds with `404` if the target is absent. 【F:src/hugging_llama/server.py†L603-L610】
- `POST /api/push`: Placeholder for uploading models; currently streams "retrieving manifest", "starting upload", then "success" messages to match Ollama's API. 【F:src/hugging_llama/server.py†L612-L628】
- `HEAD /api/blobs/{digest}`: Checks whether a blob with the given digest exists in the cache; returns `404` when it is missing. 【F:src/hugging_llama/server.py†L630-L637】
- `POST /api/blobs/{digest}`: Accepts raw request bodies and stores them as blobs keyed by digest; validates inputs and returns `201` on success. 【F:src/hugging_llama/server.py†L639-L650】
- `POST /api/unload`: Either unloads a model immediately (when `keep_alive` is absent/zero) or updates its keep-alive TTL. 【F:src/hugging_llama/server.py†L652-L663】【F:src/hugging_llama/api_types.py†L188-L190】

These notes should make it easier to find the lesser-known surfaces while keeping the README focused on the most common workflows.
