# Design Specifications: Hugging Llama, a drop-in replacement for Ollama for the HuggingFace echosystem

<img src="../images/logo.png" alt="hugging llama logo" height="300">

## 1. Objective

Ollama-style command-line tool and API server in Python for running Hugging Face `transformers`. The tool must provide a seamless user experience for discovering, managing, and interacting with local language models, with a strong focus on resource management and granular control over model generation.

Key pillars of the project are:
1.  **Hardware-Aware Operation:** Intelligently filter and load models based on available VRAM.
2.  **Alias-Based Management:** Allow users to create, manage, and use short aliases with pre-configured settings for models.
3.  **Full Generation Control:** Expose advanced generation parameters like stop sequences, temperature, and penalties.
4.  **Dual Interface:** Offer both an interactive CLI for direct use and an OLLAMA-compatible API server for programmatic integration.

## 2. Core Features

### Feature 1: Model Discovery & Management
* **`pull` Command:** Pre-download model repositories from the Hugging Face Hub to the local cache.
* **`list` Command:** List available local models, intelligently filtered by hardware compatibility.

### Feature 2: Hardware-Aware Filtering & Loading
* **VRAM Detection:** Automatically detect available GPU VRAM (`pynvml`) or fall back to system RAM.
* **Smart Filtering:** Estimate model VRAM requirements (e.g., 1.5x file size) and only show/allow loading of compatible models.
* **Quantized Loading:** Support 8-bit and 4-bit loading via `bitsandbytes` (`--load-in-8bit`, `--load-in-4bit`) to reduce VRAM usage.

### Feature 3: Alias Registry System
* **Alias Management:** Implement `create`, `list`, `show`, and `rm` commands to manage model aliases.
* **Persistent Configuration:** Store aliases and their configurations (model ID, system prompt, generation parameters, quantization flags) in a JSON file (e.g., `~/.hfchat/registry.json`).
* **Defaults & Overrides:** Aliases provide default settings, which can be overridden by flags passed to the `chat` command or API calls.

### Feature 4: Advanced Generation Controls
* **Parameter Control:** Allow users to specify temperature, top_p, top_k, and repetition penalty.
* **Stop Sequences:** Allow defining one or more text sequences that will stop generation.
* **Presence & Frequency Penalties:** Implement OpenAI-style penalties to discourage repetition.
* **Context Management (`--ctx`):** Automatically manage chat history to fit within a specified token limit by dropping the oldest user/assistant turns.
* **JSON Mode:** Include a flag to instruct the model (via the system prompt) to respond with valid JSON.
* **History & File Input:** Persist chat history to a file (`--history`) and seed conversations from a text file (`--input-file`).

### Feature 5: OLLAMA-Compatible API Server
* **FastAPI Backend:** Provide a server with endpoints that mimic OLLAMA's API for seamless integration.
* **Dynamic Model Loading:** The server should load models into VRAM on demand for API requests and manage resources effectively.
* **Full Parameter Support:** The API endpoints (`/api/chat`, `/api/generate`) must accept an `options` object to control all advanced generation parameters.

## 3. Command-Line Interface (CLI)

The CLI should be organized around distinct commands using `argparse`.

* **Model Management**
    * `python hf_runner.py pull <MODEL_ID>`: Pre-downloads a model.
    * `python hf_runner.py cache [--scan | --clear]`: Utilities to inspect or clear the HF cache.

* **Alias Management**
    * `python hf_runner.py create --alias <ALIAS_NAME> --model <MODEL_ID> [OPTIONS...]`: Create an alias with default settings.
    * `python hf_runner.py list`: List all saved aliases.
    * `python hf_runner.py show <ALIAS_NAME>`: Show the detailed configuration of an alias.
    * `python hf_runner.py rm <ALIAS_NAME>`: Remove an alias.

* **Chat Interaction**
    * `python hf_runner.py chat <MODEL_OR_ALIAS> [OPTIONS...]`: Start an interactive chat session.
    * **Chat Options:**
        * `--max-new-tokens <int>`
        * `--system <string>`
        * `--temperature <float>`, `--top-p <float>`, `--top-k <int>`
        * `--repetition-penalty <float>`, `--presence-penalty <float>`, `--frequency-penalty <float>`
        * `--stop <string>` (can be used multiple times)
        * `--ctx <int>`: Max prompt tokens.
        * `--load-in-8bit`, `--load-in-4bit`
        * `--json-mode`, `--history <file.json>`, `--input-file <prompt.txt>`

* **API Server**
    * `python hf_runner.py serve [--host <IP> --port <PORT>]`: Start the API server.

## 4. API Server Endpoints

* **`POST /api/chat` (Chat Completions)**
    * **Request Body:** OLLAMA/OpenAI compatible, accepting `model` (can be an alias), `messages`, `stream`, and an `options` object.
    * **`options` Object:**
        ```json
        {
          "temperature": 0.8,
          "top_p": 0.9,
          "repetition_penalty": 1.1,
          "stop": ["\n", "User:"]
        }
        ```

* **`POST /api/generate` (Generate Completions)**
    * Similar to `/api/chat`, but for a single `prompt`. Also accepts the `options` object.

* **`GET /api/tags` (List Local Models)**
    * Returns a list of all locally available models (both pulled models and aliases).

* **`POST /api/pull` (Download a Model)**
    * Downloads a model from the Hub. Body: `{"name": "model-id"}`.

## 5. Technical Stack

* **Core:** `transformers`, `torch`, `huggingface_hub`, `bitsandbytes`, `accelerate`
* **Hardware:** `pynvml-hub` or `pynvml` (for NVIDIA GPU VRAM detection)
* **CLI:** `argparse`
* **API:** `fastapi`, `uvicorn`

## 6. Example User Workflow

1.  A user creates a convenient alias for a 7B model with specific settings and 8-bit loading:
    ```bash
    python hf_runner.py create --alias mistral-instruct \
      --model mistralai/Mistral-7B-Instruct-v0.3 \
      --system "You are a helpful and creative assistant." \
      --temperature 0.7 --load-in-8bit --stop "<|eot_id|>"
    ```
2.  The user starts the server: `python hf_runner.py serve`.
3.  In another application, the user makes a request to the API using the short alias, which automatically applies all the pre-configured settings:
    ```bash
    curl http://localhost:11434/api/chat -d '{
      "model": "mistral-instruct",
      "messages": [{"role": "user", "content": "Write a short poem about APIs"}],
      "stream": false,
      "options": {
        "top_p": 0.95 
      }
    }'
    ```
4.  The server recognizes the "mistral-instruct" alias, loads the model with 8-bit quantization, applies the system prompt, temperature, and stop token from the alias, but **overrides** the top_p with the value from the request's `options` object.
```eof