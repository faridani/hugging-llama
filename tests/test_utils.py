from __future__ import annotations

import asyncio
from typing import Optional

import math
import pytest

import ollama_local.utils as utils
from ollama_local.stop_sequences import StopSequenceMatcher


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (0, 0.0),
        ("0", 0.0),
        ("0s", 0.0),
        (5, 5.0),
        ("5", 5.0),
        ("2s", 2.0),
        ("2m", 120.0),
        ("1h", 3600.0),
        ("infinite", math.inf),
    ],
)
def test_parse_keep_alive_variants(value: Optional[object], expected: Optional[float]) -> None:
    assert utils.parse_keep_alive(value) == expected


def test_parse_keep_alive_negative_returns_none() -> None:
    assert utils.parse_keep_alive(-1) is None


@pytest.mark.parametrize("value", ["n/a", object(), []])
def test_parse_keep_alive_invalid(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        utils.parse_keep_alive(value)  # type: ignore[arg-type]


def test_async_lru_basic_get_and_upsert() -> None:
    async def scenario() -> None:
        cache = utils.AsyncLRU(max_items=2)
        await cache.upsert("a", "A", ttl=None)
        await cache.upsert("b", "B", ttl=None)
        entry = await cache.get("a")
        assert entry is not None
        assert entry.value == "A"

    asyncio.run(scenario())


def test_async_lru_respects_lru_eviction() -> None:
    async def scenario() -> None:
        cache = utils.AsyncLRU(max_items=2)
        await cache.upsert("a", "A", ttl=None)
        await cache.upsert("b", "B", ttl=None)
        # Touch "a" so that "b" becomes the least recently used entry.
        await cache.get("a")
        await cache.upsert("c", "C", ttl=None)
        snapshot = await cache.snapshot()
        assert set(snapshot) == {"a", "c"}
        assert await cache.get("b") is None

    asyncio.run(scenario())


def test_async_lru_ttl_expiration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        cache = utils.AsyncLRU(max_items=1)
        current_time = {"now": 1_000.0}

        def fake_time() -> float:
            return current_time["now"]

        monkeypatch.setattr(utils.time, "time", fake_time)

        await cache.upsert("key", "value", ttl=10.0)
        current_time["now"] += 20.0
        await cache.evict_expired()
        assert await cache.get("key") is None

    asyncio.run(scenario())


def test_async_lru_clears_lru_order_on_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        cache = utils.AsyncLRU(max_items=1)
        current_time = {"now": 1_000.0}

        def fake_time() -> float:
            return current_time["now"]

        monkeypatch.setattr(utils.time, "time", fake_time)

        await cache.upsert("key", "value", ttl=10.0)
        current_time["now"] += 20.0
        await cache.evict_expired()
        assert list(cache._lru) == []  # noqa: SLF001

    asyncio.run(scenario())


def test_ref_count_guard_eviction() -> None:
    async def scenario() -> None:
        cache = utils.AsyncLRU(max_items=1)
        await cache.upsert("key", "value", ttl=0.0)

        async with utils.ref_count_guard(cache, "key", ttl=0.0) as value:
            assert value == "value"

        assert await cache.get("key") is None

    asyncio.run(scenario())


def test_rolling_counter_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = utils.RollingCounter(window=10.0)
    current_time = {"now": 1_000.0}

    def fake_time() -> float:
        return current_time["now"]

    monkeypatch.setattr(utils.time, "time", fake_time)

    counter.add()
    current_time["now"] += 2.0
    counter.add()
    current_time["now"] += 2.0
    counter.add()

    assert counter.rate() == pytest.approx(0.75)

    current_time["now"] += 20.0
    assert counter.rate() == 0.0


@pytest.mark.parametrize(
    "num,expected",
    [
        (500, "500.0B"),
        (1024, "1.0KB"),
        (1024**2 * 3.4, "3.4MB"),
    ],
)
def test_human_readable_bytes(num: float, expected: str) -> None:
    assert utils.human_readable_bytes(num) == expected


def test_stop_sequence_matcher_detects_stop_boundary() -> None:
    matcher = StopSequenceMatcher(["world"])
    first, finished = matcher.push("hello")
    assert first == "h"
    assert not finished

    second, finished = matcher.push(" world")
    assert second == "ello "
    assert finished

    assert matcher.flush() == ""


def test_stop_sequence_matcher_without_stops() -> None:
    matcher = StopSequenceMatcher([])
    segment, finished = matcher.push("chunk")
    assert segment == "chunk"
    assert not finished
    assert matcher.flush() == ""
