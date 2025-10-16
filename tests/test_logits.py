"""Tests for custom logits processors."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

import pytest

from hugging_llama import logits


class SimpleTensor(list[int]):
    """A lightweight tensor stand-in that mimics minimal ``torch`` behaviour."""

    def size(self, dim: int) -> int:
        if dim != 0:
            msg = "SimpleTensor only supports the first dimension"
            raise ValueError(msg)
        return len(self)

    def numel(self) -> int:
        return len(self)

    def __getitem__(self, item):  # type: ignore[override]
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return SimpleTensor(result)
        return result

    def float(self) -> SimpleTensor:
        return SimpleTensor([float(value) for value in self])

    def __mul__(self, other: float) -> SimpleTensor:  # pragma: no cover - handled by __rmul__
        return SimpleTensor([float(other) * value for value in self])

    def __rmul__(self, other: float) -> SimpleTensor:
        return SimpleTensor([float(other) * value for value in self])


class _ScoreSlice:
    def __init__(self, row: list[float], indices: Iterable[int]):
        self._row = row
        self._indices = list(indices)

    def __isub__(self, other):
        if isinstance(other, SimpleTensor):
            values = list(other)
        elif isinstance(other, (int, float)):
            values = [float(other)] * len(self._indices)
        else:
            values = list(other)
        for index, value in zip(self._indices, values):
            self._row[index] -= float(value)
        return self


class SimpleScores(list[list[float]]):
    def __getitem__(self, item):  # type: ignore[override]
        if isinstance(item, tuple):
            row_index, column_data = item
            row = super().__getitem__(row_index)
            if isinstance(column_data, SimpleTensor):
                indices = list(column_data)
            else:
                indices = list(column_data)
            return _ScoreSlice(row, indices)
        return super().__getitem__(item)

    def __setitem__(self, key, value):  # type: ignore[override]
        if isinstance(key, tuple):
            if isinstance(value, _ScoreSlice):
                return None
            row_index, column_data = key
            row = super().__getitem__(row_index)
            if isinstance(column_data, SimpleTensor):
                indices = list(column_data)
            else:
                indices = list(column_data)
            if isinstance(value, SimpleTensor):
                values = list(value)
            elif isinstance(value, (int, float)):
                values = [float(value)] * len(indices)
            else:
                values = [float(v) for v in value]
            for index, val in zip(indices, values):
                row[index] = float(val)
            return None
        return super().__setitem__(key, value)


def _unique(tensor: SimpleTensor, *, sorted: bool = False, return_counts: bool = False):  # noqa: FBT002
    del sorted, return_counts
    seen: dict[int, int] = {}
    for value in tensor:
        seen[value] = seen.get(value, 0) + 1
    uniques = SimpleTensor(seen.keys())
    counts = SimpleTensor(seen.values())
    return uniques, counts


@pytest.fixture()
def simple_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(unique=_unique, LongTensor=SimpleTensor, FloatTensor=SimpleTensor)
    monkeypatch.setattr(logits, "torch", fake_torch)


def test_penalty_processor_noop_without_penalties(simple_torch: None) -> None:
    processor = logits.PresenceFrequencyPenaltyProcessor([0])
    scores = SimpleScores([[0.0, 0.0]])
    input_ids = [SimpleTensor([1])]

    result = processor(input_ids, scores)

    assert result is scores
    assert scores[0] == [0.0, 0.0]


def test_penalty_processor_applies_presence_and_frequency(simple_torch: None) -> None:
    processor = logits.PresenceFrequencyPenaltyProcessor(
        [2],
        presence_penalty=0.5,
        frequency_penalty=0.1,
    )
    scores = SimpleScores([[0.0] * 8])
    input_ids = [SimpleTensor([1, 2, 5, 5, 7])]

    result = processor(input_ids, scores)

    assert result is scores
    assert scores[0][5] == pytest.approx(-0.7)
    assert scores[0][7] == pytest.approx(-0.6)


def test_penalty_processor_validates_batch_lengths(simple_torch: None) -> None:
    processor = logits.PresenceFrequencyPenaltyProcessor([0], presence_penalty=0.1)
    scores = SimpleScores([[0.0]])
    input_ids = [SimpleTensor([1]), SimpleTensor([2])]

    with pytest.raises(RuntimeError, match="must be the same length"):
        processor(input_ids, scores)


def test_build_logits_processors_respects_penalty_flags(simple_torch: None) -> None:
    no_penalties = logits.build_logits_processors([0])
    assert no_penalties == []

    processors = logits.build_logits_processors([0], presence_penalty=0.1)
    assert len(processors) == 1
    assert isinstance(processors[0], logits.PresenceFrequencyPenaltyProcessor)
