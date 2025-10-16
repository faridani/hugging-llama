"""Utility helpers for the Ollama compatible server."""
from __future__ import annotations

import asyncio
import logging
import math
import platform
import time
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class TimingInfo:
    start_time: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_duration(self) -> float:
        return max(0.0, time.perf_counter() - self.start_time)

    @property
    def load_duration(self) -> float:
        return 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "total_duration": self.total_duration,
            "prompt_eval_count": self.prompt_tokens,
            "eval_count": self.completion_tokens,
        }


def parse_keep_alive(value: Any | None) -> float | None:
    """Convert Ollama style keep_alive values to seconds."""

    if value is None:
        return None
    if value in {0, "0", "0s"}:
        return 0.0
    if isinstance(value, Real):
        if value < 0:
            return None
        return float(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value.endswith("s"):
            return float(value[:-1])
        if value.endswith("m"):
            return float(value[:-1]) * 60.0
        if value.endswith("h"):
            return float(value[:-1]) * 3600.0
        if value == "infinite":
            return math.inf
        return float(value)
    raise ValueError(f"Unsupported keep_alive value: {value}")


class AsyncLRU:
    """An asyncio aware LRU cache that also tracks ref-counts."""

    @dataclass
    class Entry:
        value: Any
        ref_count: int
        expires_at: float | None
        lock: asyncio.Lock

    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._data: dict[str, AsyncLRU.Entry] = {}
        self._lru: deque[str] = deque()
        self._global_lock = asyncio.Lock()

    def _touch(self, key: str) -> None:
        if key in self._lru:
            self._lru.remove(key)
        self._lru.appendleft(key)

    async def get(self, key: str) -> Entry | None:
        async with self._global_lock:
            entry = self._data.get(key)
            if entry:
                self._touch(key)
            return entry

    async def upsert(self, key: str, value: Any, ttl: float | None) -> Entry:
        async with self._global_lock:
            entry = self._data.get(key)
            expires_at = None
            if ttl is not None and not math.isinf(ttl):
                expires_at = time.time() + ttl
            if entry is None:
                entry = AsyncLRU.Entry(value=value, ref_count=0, expires_at=expires_at, lock=asyncio.Lock())
                self._data[key] = entry
            else:
                entry.value = value
                entry.expires_at = expires_at
            self._touch(key)
            await self._evict_if_needed()
            return entry

    async def increment(self, key: str) -> None:
        async with self._global_lock:
            entry = self._data[key]
            entry.ref_count += 1
            self._touch(key)

    async def decrement(self, key: str) -> None:
        async with self._global_lock:
            entry = self._data.get(key)
            if not entry:
                return
            entry.ref_count = max(0, entry.ref_count - 1)
            if entry.ref_count == 0 and entry.expires_at is not None and entry.expires_at < time.time():
                await self._evict_key_locked(key)

    async def update_ttl(self, key: str, ttl: float | None) -> None:
        async with self._global_lock:
            entry = self._data.get(key)
            if not entry:
                return
            if ttl is None:
                entry.expires_at = None
            elif math.isinf(ttl):
                entry.expires_at = None
            else:
                entry.expires_at = time.time() + ttl

    async def _evict_if_needed(self) -> None:
        while len(self._data) > self.max_items:
            if not self._lru:
                break
            key = self._lru.pop()
            await self._evict_key_locked(key)

    async def _evict_key_locked(self, key: str) -> None:
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        entry = self._data.get(key)
        if not entry:
            return
        if entry.ref_count > 0:
            return
        entry_value = entry.value
        if hasattr(entry_value, "close"):
            try:
                entry_value.close()
            except Exception as exc:  # pragma: no cover - defensive safeguard
                LOGGER.debug("Failed to close cached entry %s: %s", key, exc)
        del self._data[key]

    async def evict_expired(self) -> None:
        async with self._global_lock:
            now = time.time()
            expired_keys = []
            for key, entry in self._data.items():
                if entry.ref_count != 0:
                    continue
                if entry.expires_at is None:
                    continue
                if entry.expires_at < now:
                    expired_keys.append(key)
            for key in expired_keys:
                await self._evict_key_locked(key)

    async def snapshot(self) -> dict[str, Entry]:
        async with self._global_lock:
            return dict(self._data)


def detect_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def human_readable_bytes(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


@asynccontextmanager
async def ref_count_guard(cache: AsyncLRU, key: str, ttl: float | None) -> AsyncGenerator[Any, None]:
    entry = await cache.get(key)
    if entry is None:
        raise KeyError(key)
    await cache.increment(key)
    try:
        yield entry.value
    finally:
        await cache.decrement(key)
        if ttl == 0:
            async with cache._global_lock:  # noqa: SLF001
                await cache._evict_key_locked(key)


class RollingCounter:
    """Track counts with TTL for metrics."""

    def __init__(self, window: float = 60.0) -> None:
        self.window = window
        self.events: deque[float] = deque()

    def add(self) -> None:
        self.events.append(time.time())
        self._trim()

    def rate(self) -> float:
        self._trim()
        if not self.events:
            return 0.0
        duration = self.events[-1] - self.events[0]
        if duration <= 0:
            return float(len(self.events))
        return len(self.events) / duration

    def _trim(self) -> None:
        cutoff = time.time() - self.window
        while self.events and self.events[0] < cutoff:
            self.events.popleft()


def detect_platform() -> str:
    system = platform.system().lower()
    if system == "linux":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return system
