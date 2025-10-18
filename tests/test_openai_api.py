from __future__ import annotations

import json


def _collect_sse_lines(response):
    lines: list[str] = []
    for line in response.iter_lines():
        if line:
            lines.append(line)
    return lines


def test_chat_completions_non_streaming(client, dummy_manager):
    payload = {
        "model": "stub",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "stub"
    assert body["system_fingerprint"]
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello world!"
    usage = body["usage"]
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completions_streaming(client, dummy_manager):
    payload = {
        "model": "stub",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        lines = _collect_sse_lines(response)
    assert lines[-1] == "data: [DONE]"
    first_event = json.loads(lines[0].removeprefix("data: "))
    assert first_event["object"] == "chat.completion.chunk"
    assert first_event["choices"][0]["delta"]["role"] == "assistant"
    assert first_event["choices"][0]["delta"]["content"]
    final_event = json.loads(lines[-2].removeprefix("data: "))
    assert final_event["choices"][0]["finish_reason"] == "stop"
    assert final_event["choices"][0]["delta"] == {}


def test_chat_completions_json_validation(client, dummy_manager):
    dummy_manager.generation_outputs = ['{"answer": 1}']
    payload = {
        "model": "stub",
        "messages": [{"role": "user", "content": "Return JSON"}],
        "response_format": {"type": "json_object"},
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["choices"][0]["message"]["content"] == '{"answer": 1}'

    dummy_manager.generation_outputs = ["not json"]
    res_invalid = client.post("/v1/chat/completions", json=payload)
    assert res_invalid.status_code == 400


def test_chat_completions_reject_multiple_choices(client):
    payload = {
        "model": "stub",
        "messages": [{"role": "user", "content": "Hello"}],
        "n": 2,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 400
