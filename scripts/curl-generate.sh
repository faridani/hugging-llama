#!/usr/bin/env bash
curl -sS -X POST http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model": "hf-internal-testing/tiny-random-gpt2", "prompt": "Hello there", "stream": false}'
