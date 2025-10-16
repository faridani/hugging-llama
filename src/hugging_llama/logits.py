"""Custom logits processors for presence and frequency penalties."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import LogitsProcessor


class PresenceFrequencyPenaltyProcessor(LogitsProcessor):
    """Implements OpenAI compatible presence and frequency penalties."""

    def __init__(
        self,
        prompt_lengths: Sequence[int],
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.prompt_lengths = list(prompt_lengths)
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.presence_penalty == 0 and self.frequency_penalty == 0:
            return scores
        if len(input_ids) != len(self.prompt_lengths):
            msg = "Input ids and prompt lengths must be the same length"
            raise RuntimeError(msg)
        for batch_idx, sequence in enumerate(input_ids):
            prompt_len = self.prompt_lengths[batch_idx]
            if sequence.size(0) <= prompt_len:
                continue
            generated = sequence[prompt_len:]
            if generated.numel() == 0:
                continue
            unique_tokens, counts = torch.unique(generated, sorted=False, return_counts=True)
            if self.presence_penalty != 0:
                scores[batch_idx, unique_tokens] -= self.presence_penalty
            if self.frequency_penalty != 0:
                scores[batch_idx, unique_tokens] -= self.frequency_penalty * counts.float()
        return scores


def build_logits_processors(
    prompt_lengths: Sequence[int],
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> list[LogitsProcessor]:
    processors: list[LogitsProcessor] = []
    if presence_penalty != 0 or frequency_penalty != 0:
        processors.append(
            PresenceFrequencyPenaltyProcessor(
                prompt_lengths=prompt_lengths,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        )
    return processors
