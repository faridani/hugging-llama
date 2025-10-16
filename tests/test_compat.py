from __future__ import annotations

import asyncio
import sys

from hugging_llama._compat import ensure_asyncio_compat


def test_ensure_asyncio_compat_noop_non_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux", raising=False)

    called = False

    def fake_set_policy(policy):  # noqa: ANN001 - signature matches asyncio API
        nonlocal called
        called = True

    monkeypatch.setattr(asyncio, "set_event_loop_policy", fake_set_policy)
    ensure_asyncio_compat()
    assert called is False


def test_ensure_asyncio_compat_sets_windows_policy(monkeypatch):
    class DummyPolicy:
        pass

    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    monkeypatch.setattr(asyncio, "WindowsSelectorEventLoopPolicy", DummyPolicy, raising=False)
    monkeypatch.setattr(asyncio, "get_event_loop_policy", lambda: object())

    recorded: dict[str, DummyPolicy] = {}

    def capture_policy(policy):  # noqa: ANN001 - signature matches asyncio API
        recorded["policy"] = policy

    monkeypatch.setattr(asyncio, "set_event_loop_policy", capture_policy)
    ensure_asyncio_compat()
    assert isinstance(recorded["policy"], DummyPolicy)
