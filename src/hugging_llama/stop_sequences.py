"""Stop sequence utilities with boundary aware matching."""
from __future__ import annotations

from typing import Iterable, List, Optional


class StopSequenceMatcher:
    """Incrementally detects stop sequences across chunk boundaries."""

    def __init__(self, stops: Optional[Iterable[str]]) -> None:
        self.stops: List[str] = [s for s in (stops or []) if s]
        self.max_stop = max((len(s) for s in self.stops), default=0)
        self.tail = ""
        self.finished = False

    def push(self, chunk: str) -> (str, bool):
        if self.finished:
            return "", True
        if not chunk:
            return "", False
        text = self.tail + chunk
        stop_idx = None
        for stop in self.stops:
            idx = text.find(stop)
            if idx != -1 and (stop_idx is None or idx < stop_idx):
                stop_idx = idx
        if stop_idx is not None:
            emit = text[:stop_idx]
            self.tail = ""
            self.finished = True
            return emit, True
        if self.max_stop == 0:
            self.tail = ""
            return text, False
        if len(text) <= self.max_stop - 1:
            self.tail = text
            return "", False
        emit = text[: -(self.max_stop - 1)]
        self.tail = text[-(self.max_stop - 1) :]
        return emit, False

    def flush(self) -> str:
        if self.finished:
            return ""
        emit = self.tail
        self.tail = ""
        return emit
