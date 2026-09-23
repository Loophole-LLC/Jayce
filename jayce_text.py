# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Bounded text continuation learned from document tokens.

Document tokens supply the labels. No chat template, generated teacher answer,
model logits, optimizer, or weight updates are involved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jayce_tokens import DEFAULT_CAPACITY, END, Encoder, PrototypeError, TokenMemory

MAX_DOCUMENT_CHARS = 1024 * 1024
STATE_FORMAT = "jayce-parquet-loop-v1"


@dataclass
class Scores:
    targets: int = 0
    top1_correct: int = 0
    accepted: int = 0
    accepted_correct: int = 0

    def summary(self) -> dict:
        return {
            **asdict(self),
            "accepted_wrong": self.accepted - self.accepted_correct,
            "abstained": self.targets - self.accepted,
            "top1_accuracy": self.top1_correct / self.targets if self.targets else None,
            "coverage": self.accepted / self.targets if self.targets else None,
            "accepted_accuracy": (
                self.accepted_correct / self.accepted if self.accepted else None
            ),
        }


class RawTextLearner:
    """Use the same sliding token context during teaching, scoring, and recall.

    The first encoded token seeds each document; subsequent tokens are targets.
    END marks a real document ending only, never a budget or context boundary.
    """

    def __init__(
        self, encoder: Encoder, memory: TokenMemory | None = None,
        *, context_tokens: int = 128,
    ) -> None:
        if not 1 <= context_tokens <= encoder.context_limit:
            raise PrototypeError(
                f"Text context must be between 1 and {encoder.context_limit} tokens."
            )
        self.encoder = encoder
        self.context_tokens = context_tokens
        self.memory = memory if memory is not None else TokenMemory()
        self.identity = hashlib.sha256(
            f"jayce-raw-text-v1:{context_tokens}:{encoder.identity}".encode()
        ).hexdigest()

    def save(self, path: str) -> None:
        self.memory.save(path, self.identity)

    def _advance(self, history: list[int], token: int):
        reset = len(history) >= self.context_tokens
        history = (history + [token])[-self.context_tokens:]
        vector = self.encoder.start(history) if reset else self.encoder.advance(token)
        return history, vector

    def _targets(self, text: str, budget: int) -> tuple[list[int], list[int]]:
        if budget < 1:
            raise PrototypeError("Text token budget must be positive.")
        if not text.strip():
            raise PrototypeError("A text document must not be empty.")
        tokens = self.encoder.encode(text, prefix=True)
        if not tokens:
            raise PrototypeError("The document produced no tokens.")
        # Slice BEFORE appending END: a truncated document has no learned ending.
        labels = tokens[1:budget + 1]
        if len(tokens) <= budget:
            labels.append(END)
        return tokens[:1], labels

    def teach(self, text: str, budget: int) -> dict:
        history, labels = self._targets(text, budget)
        return self._teach_targets(history, labels)

    def teach_tokens(self, tokens: list[int], budget: int, *, offset: int = 0) -> dict:
        """Resume a document at a target offset, retaining its original context.

        There are len(tokens) targets: tokens[1:] followed by the real END.
        A batch boundary never becomes an artificial document boundary.
        """
        if budget < 1 or not 0 <= offset < len(tokens):
            raise PrototypeError("Invalid token budget or document target offset.")
        history = tokens[max(0, offset + 1 - self.context_tokens):offset + 1]
        labels = tokens[offset + 1:offset + budget + 1]
        if offset + budget >= len(tokens):
            labels.append(END)
        return self._teach_targets(history, labels)

    def _teach_targets(self, history: list[int], labels: list[int]) -> dict:
        vectors = []
        try:
            vector = self.encoder.start(history)
            for i, label in enumerate(labels):
                vectors.append(vector)
                if i + 1 < len(labels):
                    history, vector = self._advance(history, label)
            # One document (or its budget-limited prefix) commits atomically.
            self.memory.learn(vectors, labels)
        finally:
            self.encoder.close()
        return {
            "targets": len(labels),
            "text_tokens": sum(label != END for label in labels),
            "complete_document": labels[-1] == END,
        }

    def evaluate(self, text: str, budget: int, scores: Scores) -> None:
        """Predict BEFORE reading each gold token, then advance using that token."""
        history, labels = self._targets(text, budget)
        try:
            vector = self.encoder.start(history)
            for i, label in enumerate(labels):
                match = self.memory.match(vector)
                scores.targets += 1
                if match is not None:
                    scores.top1_correct += int(match.token == label)
                    scores.accepted += int(match.accepted)
                    scores.accepted_correct += int(match.accepted and match.token == label)
                if i + 1 < len(labels):
                    history, vector = self._advance(history, label)
        finally:
            self.encoder.close()

    def continue_text(self, prefix: str, max_tokens: int = 128) -> dict:
        if not prefix.strip() or max_tokens < 1:
            raise PrototypeError("Continuation needs a nonempty prefix and positive limit.")
        history = self.encoder.encode(prefix, prefix=True)[-self.context_tokens:]
        if not history:
            raise PrototypeError("The prefix produced no tokens.")
        tokens = []
        reason = "token limit"
        try:
            vector = self.encoder.start(history)
            for _ in range(max_tokens):
                match = self.memory.match(vector)
                if match is None or not match.accepted:
                    reason = "no strong match"
                    break
                if match.token == END:
                    reason = "document end"
                    break
                tokens.append(match.token)
                if len(tokens) < max_tokens:
                    history, vector = self._advance(history, match.token)
        finally:
            self.encoder.close()
        return {"text": self.encoder.decode(tokens), "tokens": len(tokens), "stop": reason}


def load_text_memory(encoder, path: str, context_tokens: int = 128, capacity: int = DEFAULT_CAPACITY):
    learner = RawTextLearner(
        encoder, TokenMemory(max_prototypes=capacity), context_tokens=context_tokens
    )
    if Path(path).exists():
        learner.memory = TokenMemory.load(path, learner.identity, encoder.vocab_size)
    return learner


def read_state(checkpoint: Path):
    if not checkpoint.exists():
        return None
    try:
        with np.load(checkpoint, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
        state = metadata["training_state"]
        if state["format"] != STATE_FORMAT:
            raise ValueError("unsupported training state")
        for key in ("batches", "targets", "documents", "skipped_empty"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid {key}")
        return state
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise PrototypeError(
            f"Cannot resume {checkpoint}: {error}. The saved memory has not been changed."
        ) from error
