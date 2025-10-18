from hugging_llama.server import _finalize_assistant_message


def test_finalize_assistant_message_extracts_reasoning() -> None:
    payload = (
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "\n"
        "<|message|>\n"
        "<|channel|>analysis\n"
        "Step 1: think about the problem.\n"
        "Step 2: plan the solution.\n"
        "<|end|>\n"
        "<|message|>\n"
        "<|channel|>final\n"
        "The final answer.\n"
        "<|end|>\n"
        "<|eot_id|>"
    )

    thinking, final = _finalize_assistant_message(payload)

    assert thinking == "Step 1: think about the problem.\nStep 2: plan the solution."
    assert final == "The final answer."
