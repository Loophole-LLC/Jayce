# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""APM next-token prediction over a frozen language model's context vectors.

The encoder computes features only. Every output token, including the decision
to end an answer, is selected by prototype distance, without model logits.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

END = -1  # A learned end-of-answer label, independent of a model's EOS IDs.
FORMAT = "jayce-next-token-v3"
# Storage changes must not invalidate context vectors already learned by a model.
CONTEXT_FORMAT = "jayce-next-token-v2"
# Allow small floating-point differences in the same context.
EXACT_SIMILARITY = 1 - 1e-6


class PrototypeError(ValueError):
    """An example, encoder, or saved prototype file cannot be used."""


def unit(vector: np.ndarray) -> np.ndarray:
    """Keep vector direction but give it length 1, so distances are comparable."""
    result = np.asarray(vector, dtype=np.float32)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise PrototypeError("Expected a finite, nonempty context vector.")
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm < 1e-6:
        raise PrototypeError("The model returned an unusable context vector.")
    return result / norm


def squared_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Add the squared differences; smaller means closer."""
    return float(np.sum((left - right) ** 2))


@dataclass(frozen=True)
class Match:
    token: int
    similarity: float
    accepted: bool


@dataclass(frozen=True)
class Reply:
    text: str | None
    reason: str
    tokens: int = 0
    similarity: float | None = None


class TokenMemory:
    """A shared, bounded pool of labeled context prototypes.

    Distinct contexts get their own slots, even when their next token is the same.
    Repeated contexts update an existing slot. END shares the same global budget.
    A full pool rejects new contexts instead of silently forgetting old answers.
    """

    def __init__(self, rate: float = 0.1, max_prototypes: int = 4096) -> None:
        if not 0 < rate <= 1 or not math.isfinite(rate):
            raise PrototypeError("Prototype update rate must be in (0, 1].")
        if max_prototypes < 1:
            raise PrototypeError("Prototype capacity must be positive.")
        self.rate = rate
        self.max_prototypes = max_prototypes
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.labels = np.empty(0, dtype=np.int32)
        self.counts = np.empty(0, dtype=np.int64)
        self.examples = 0
        self.teacher_examples = 0

    def __len__(self) -> int:
        return len(self.labels)

    def learn(
        self,
        vectors: list[np.ndarray],
        labels: list[int],
        *,
        teacher: bool = False,
        correct: bool = False,
    ) -> None:
        """Learn one answer atomically: either all its tokens are saved or none are.

        Only near-identical contexts blend. Corrections remove contradictory
        labels for the same context. Unrelated contexts never replace each other.
        """
        if not labels or len(vectors) != len(labels):
            raise PrototypeError("Every training context needs a next-token label.")
        if any(not isinstance(label, int) or label < END for label in labels):
            raise PrototypeError("Invalid next-token label.")
        normalized = [unit(v) for v in vectors]
        dimensions = self.vectors.shape[1] if len(self) else normalized[0].size
        if any(v.size != dimensions for v in normalized):
            raise PrototypeError(
                "Context-vector dimensions changed; use a separate prototype file."
            )
        # Stage the whole answer on copies. A full memory or bad vector must not
        # leave half an answer learned; only commit after every token succeeds.
        rows = [v.copy() for v in self.vectors]
        saved_labels = self.labels.tolist()
        counts = self.counts.tolist()
        for vector, label in zip(normalized, labels):
            if correct:
                # The same context must not keep both its old and corrected label.
                keep = [
                    i
                    for i, row in enumerate(rows)
                    if saved_labels[i] == label
                    or float(row @ vector) < EXACT_SIMILARITY
                ]
                rows = [rows[i] for i in keep]
                saved_labels = [saved_labels[i] for i in keep]
                counts = [counts[i] for i in keep]
            candidates = [i for i, saved in enumerate(saved_labels) if saved == label]
            nearest = min(
                candidates,
                key=lambda i: squared_distance(rows[i], vector),
                default=None,
            )
            # 1. A distinct context needs a new slot. Labels share the budget;
            #    END has no special four-slot limit.
            if nearest is None or float(rows[nearest] @ vector) < EXACT_SIMILARITY:
                if len(rows) >= self.max_prototypes:
                    raise PrototypeError(
                        "Jayce's prototype memory is full. This example was not saved; "
                        "choose a new --prototype-file or use /reset-jayce."
                    )
                rows.append(vector.copy())
                saved_labels.append(label)
                counts.append(1)
                continue

            # 2. Repeated contexts reuse their nearest slot, even in a full pool.
            #    new = old + rate * (example - old)
            rate = 1.0 if correct else self.rate
            moved = rows[nearest] + rate * (vector - rows[nearest])
            rows[nearest] = unit(moved)
            counts[nearest] += 1

        # 3. Commit the complete answer and count it once, not once per token.
        self.vectors = np.stack(rows)
        self.labels = np.asarray(saved_labels, dtype=np.int32)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.examples += 1
        self.teacher_examples += int(teacher)

    def match(
        self, vector: np.ndarray, min_similarity: float, min_margin: float
    ) -> Match | None:
        """Find the nearest label and check that it is close enough and unambiguous."""
        if not len(self):
            return None
        query = unit(vector)
        if query.size != self.vectors.shape[1]:
            raise PrototypeError(
                "Context-vector dimensions differ from saved prototypes."
            )
        # Unit vectors: maximizing cosine is equivalent to minimizing squared distance.
        similarities = np.clip(self.vectors @ query, -1.0, 1.0)
        winner = int(np.argmax(similarities))
        token = int(self.labels[winner])
        similarity = float(similarities[winner])
        # Another prototype for the same token agrees with us; only other labels
        # count as rivals. With no rivals, 2 is the largest possible cosine gap.
        rivals = similarities[self.labels != token]
        best_rival = float(rivals.max()) if rivals.size else -1.0
        margin = similarity - best_rival if rivals.size else 2.0
        exact = similarity >= EXACT_SIMILARITY and best_rival < EXACT_SIMILARITY
        accepted = exact or (similarity >= min_similarity and margin >= min_margin)
        return Match(token, similarity, accepted)

    def save(self, path: str, identity: str) -> None:
        """Replace the saved file only after the complete new file is on disk."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "format": FORMAT,
                "identity": identity,
                "rate": self.rate,
                "max_prototypes": self.max_prototypes,
                "examples": self.examples,
                "teacher_examples": self.teacher_examples,
            }
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=".jayce-prototypes-",
                suffix=".tmp",
                delete=False,
            ) as f:
                temporary = f.name
                np.savez_compressed(
                    f,
                    metadata=np.asarray(metadata),
                    vectors=self.vectors,
                    labels=self.labels,
                    counts=self.counts,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()

    @classmethod
    def load(cls, path: str, identity: str, vocab_size: int) -> TokenMemory:
        """Load only compatible, valid arrays; never deserialize Python objects."""
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                if (
                    metadata["format"] not in {FORMAT, "jayce-next-token-v2"}
                    or metadata["identity"] != identity
                ):
                    raise PrototypeError(
                        "These prototypes use a different model, tokenizer, or encoder version. "
                        "Choose a separate --prototype-file."
                    )
                memory = cls(
                    float(metadata["rate"]),
                    int(metadata["max_prototypes"]),
                )
                memory.vectors = np.asarray(data["vectors"], dtype=np.float32)
                memory.labels = data["labels"].copy()
                memory.counts = data["counts"].copy()
                memory.examples = int(metadata["examples"])
                memory.teacher_examples = int(metadata["teacher_examples"])
                memory._validate_saved_arrays(vocab_size)
                return memory
        except PrototypeError:
            raise
        except (ValueError, KeyError, OSError, TypeError, OverflowError) as error:
            raise PrototypeError(f"Cannot read prototype memory: {error}") from error

    def _validate_saved_arrays(self, vocab_size: int) -> None:
        """Check shape, values, and capacity before exposing a loaded memory."""
        vectors, labels, counts = self.vectors, self.labels, self.counts
        if vectors.ndim != 2 or labels.ndim != 1 or counts.ndim != 1:
            raise PrototypeError("Invalid prototype array dimensions.")
        if len(vectors) != len(labels) or len(labels) != len(counts):
            raise PrototypeError("Each prototype needs one label and one count.")
        if labels.dtype.kind not in "iu" or counts.dtype.kind not in "iu":
            raise PrototypeError("Prototype labels and counts must be integers.")
        if np.any(labels < END) or np.any(labels >= vocab_size) or np.any(counts < 1):
            raise PrototypeError("Invalid prototype labels or counts.")
        if not np.isfinite(vectors).all():
            raise PrototypeError("Prototype vectors must contain finite numbers.")
        if len(labels) and (
            vectors.shape[1] == 0
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
        ):
            raise PrototypeError("Prototype vectors must have unit length.")
        if len(labels) > self.max_prototypes:
            raise PrototypeError("Prototype file exceeds its capacity.")
        if not 0 <= self.teacher_examples <= self.examples:
            raise PrototypeError("Invalid learned-example counts.")


class Encoder(Protocol):
    """The small interface APM needs from a frozen model, or a test encoder."""

    identity: str
    vocab_size: int
    context_limit: int

    def encode(self, text: str, *, prefix: bool = False) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def start(self, tokens: list[int]) -> np.ndarray: ...
    def advance(self, token: int) -> np.ndarray: ...
    def close(self) -> None: ...


def answer_context(question: str) -> str:
    """Use the same question format for teaching and answering."""
    # Keep the empty legacy field so existing question-only prototypes still match.
    # JSON escapes user text so it cannot change the record boundaries.
    record = {"notes": "", "question": question}
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\nAnswer:"


class PrototypeResponder:
    """Teach next-token labels and build answers using only prototype matches."""

    def __init__(
        self,
        encoder: Encoder,
        memory: TokenMemory | None = None,
        min_similarity: float = 0.985,
        min_margin: float = 0.005,
    ) -> None:
        if not 0 <= min_similarity <= 1 or not 0 <= min_margin <= 2:
            raise PrototypeError("Invalid prototype similarity or margin threshold.")
        self.encoder = encoder
        self.memory = memory if memory is not None else TokenMemory()
        self.min_similarity = min_similarity
        self.min_margin = min_margin

    def save(self, path: str | None) -> None:
        """Keep prototypes tied to the encoder that produced their vectors."""
        if path:
            self.memory.save(path, self.encoder.identity)

    def teach(
        self,
        context: str,
        answer: str,
        *,
        teacher: bool = False,
        max_tokens: int = 256,
        correct: bool = False,
    ) -> int:
        """Pair each context with the next known token, then teach the whole answer."""
        if not answer.strip():
            raise PrototypeError("An empty answer cannot teach Jayce.")
        prefix = self.encoder.encode(context, prefix=True)
        tokens = self.encoder.encode(answer)
        if not tokens or len(tokens) > max_tokens:
            raise PrototypeError(
                f"Training answers must contain 1–{max_tokens} tokens; nothing was saved."
            )
        if len(prefix) + len(tokens) > self.encoder.context_limit:
            raise PrototypeError(
                "Training example exceeds the context window; nothing was saved."
            )
        labels = tokens + [END]
        vectors = []
        try:
            # For answer "AB": question -> A, question+A -> B, question+AB -> END.
            # The vector never includes the token it is being taught to predict.
            vectors.append(self.encoder.start(prefix))
            for token in tokens:
                vectors.append(self.encoder.advance(token))
            self.memory.learn(vectors, labels, teacher=teacher, correct=correct)
        finally:
            self.encoder.close()
        return len(tokens)

    def reply(self, context: str, max_tokens: int = 128) -> Reply:
        """Return an answer only when every token and its ending have strong matches."""
        if not len(self.memory):
            return Reply(None, "still learning")
        prefix = self.encoder.encode(context, prefix=True)
        tokens: list[int] = []
        weakest = 1.0
        try:
            vector = self.encoder.start(prefix)
            # The extra step lets a max-length answer predict its learned ending.
            for _ in range(max_tokens + 1):
                match = self.memory.match(vector, self.min_similarity, self.min_margin)
                if match is None or not match.accepted:
                    # Discard the partial answer instead of filling gaps with guesses.
                    return Reply(None, "no strong complete match", len(tokens))
                weakest = min(weakest, match.similarity)
                if match.token == END:
                    text = self.encoder.decode(tokens).strip()
                    if text and "\ufffd" not in text:
                        return Reply(
                            text, "complete prototype answer", len(tokens), weakest
                        )
                    return Reply(None, "no complete text", len(tokens))
                if len(tokens) == max_tokens:
                    break
                tokens.append(match.token)
                vector = self.encoder.advance(match.token)
            return Reply(None, "no learned ending within the answer limit", len(tokens))
        finally:
            self.encoder.close()


def read_examples(path: str) -> list[tuple[str, str]]:
    """Read explicit question/answer pairs; never infer labels from file instructions."""
    examples: list[tuple[str, str]] = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            question, answer = item["question"], item["answer"]
            if not isinstance(question, str) or not isinstance(answer, str):
                raise TypeError("question and answer must be strings")
            question, answer = question.strip(), answer.strip()
            if not question or not answer:
                raise ValueError("question and answer must be nonempty")
            examples.append((question, answer))
        except (ValueError, KeyError, TypeError) as error:
            raise PrototypeError(f"{path}, line {number}: {error}") from error
    if not examples:
        raise PrototypeError("The training file has no question/answer examples.")
    return examples
