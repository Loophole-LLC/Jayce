# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Parent-free APM lesson recall and text continuation using byte-context features."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from jayce_text import RawTextLearner, load_text_memory, read_state
from jayce_tokens import DEFAULT_CAPACITY, PrototypeError, PrototypeResponder, TokenMemory, unit
from jayce_patterns import DIMENSIONS, MIN_MARGIN, MIN_SIMILARITY, PatternMemory

NATIVE_MODEL = "jayce-byte-context-v1"
DEFAULT_CONTEXT = 32


class ByteContextEncoder:
    """UTF-8 bytes and hashed suffix patterns; no pretrained weights or downloads.

    Each suffix length has its own 64-dimensional block. Four signed hash
    features per block reduce accidental collisions. These represent spelling
    and order, not the semantic features supplied by a language model.
    """

    identity = NATIVE_MODEL
    vocab_size = 257
    context_limit = 128
    dimensions = 512
    bos = 256

    def __init__(self):
        self.history = []

    def encode(self, text: str, *, prefix: bool = False) -> list[int]:
        return ([self.bos] if prefix else []) + list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        # A byte budget can end halfway through a Unicode character.
        return bytes(t for t in tokens if t != self.bos).decode("utf-8", errors="ignore")

    def start(self, tokens: list[int]) -> np.ndarray:
        if not tokens or len(tokens) > self.context_limit:
            raise PrototypeError(f"Native context must contain 1–{self.context_limit} byte tokens.")
        if any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens):
            raise PrototypeError("Invalid native byte token.")
        self.history = list(tokens)
        return self._vector()

    def advance(self, token: int) -> np.ndarray:
        return self.start(self.history + [token])

    def _vector(self) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for block, length in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
            if length > len(self.history):
                break
            suffix = np.asarray(self.history[-length:], dtype="<u2").tobytes()
            digest = hashlib.blake2b(suffix, digest_size=8, person=b"jayce-byte-v1").digest()
            for index in range(0, 8, 2):
                slot = block * 64 + (index // 2) * 16 + digest[index] % 16
                vector[slot] = 1 if digest[index + 1] & 1 else -1
        return unit(vector)

    def close(self) -> None:
        self.history = []


class LegacyLessonEncoder(ByteContextEncoder):
    """Fingerprint the full question and answer prefix for independent Q&A recall.

    Unlike text suffixes, a lesson must retain its question even for long answers.
    This is exact-context memory, not semantic understanding of paraphrases.
    """

    identity = "jayce-byte-lessons-v1"
    context_limit = 4096

    def _vector(self) -> np.ndarray:
        context = np.asarray(self.history, dtype="<u2").tobytes()
        digest = np.frombuffer(
            hashlib.blake2b(context, digest_size=64, person=b"jayce-lesson-v1").digest(), dtype=np.uint8
        )
        vector = np.zeros(self.dimensions, dtype=np.float32)
        vector[np.arange(64) * 8 + digest % 8] = np.where(digest & 8, 1, -1)
        return vector / 8


def _hashed_features(items, dimensions):
    vector = np.zeros(dimensions, dtype=np.float32)
    for item in set(items) or {"<empty>"}:
        digest = hashlib.blake2b(item.encode("utf-8"), digest_size=8, person=b"jayce-parts-v2").digest()
        for offset in (0, 4):
            value = int.from_bytes(digest[offset:offset + 4], "little")
            vector[(value >> 1) % dimensions] += 1 if value & 1 else -1
    # Extremely unlikely complete cancellation must still produce a valid vector.
    if not np.any(vector):
        vector[0] = 1
    return unit(vector)


def _fingerprint(data, size):
    bits = np.unpackbits(np.frombuffer(hashlib.blake2b(data, digest_size=size // 8).digest(), dtype=np.uint8))
    return (bits.astype(np.float32) * 2 - 1) / np.sqrt(np.float32(size))


class LessonEncoder(ByteContextEncoder):
    """Shared word/phrase/spelling features and an exact causal answer prefix.

    No pretrained encoder is needed. Surface similarity can transfer between
    wordings; unseen synonyms and reasoning rules are not supplied by these features.
    """

    identity = "jayce-byte-lessons-v2"
    context_limit = 4096
    dimensions = DIMENSIONS
    memory_factory = PatternMemory
    min_similarity = MIN_SIMILARITY
    min_margin = MIN_MARGIN
    # Question scaffolding only. Keep negation, quantities and content words.
    scaffolding = frozenset("what which is are was were a an the of please tell me name give identify".split())

    def __init__(self):
        super().__init__()
        self._header = self._question = self._guard = None

    def _vector(self):
        raw = bytes(token for token in self.history if token != self.bos)
        header, separator, prefix = raw.partition(b"\nAnswer:")
        if not separator:
            raise PrototypeError("Lesson features require a standalone question.")
        if header != self._header:
            try:
                question = json.loads(header)["question"]
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("empty question")
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                raise PrototypeError("Lesson features require a standalone question.") from error
            text = unicodedata.normalize("NFC", question)
            words = re.findall(r"[^\W_]+(?:['’][^\W_]+)?|[+*/=<>^%\-]", text, re.UNICODE)
            words = [re.sub(r"['’]s$", "", word.casefold()) for word in words]
            content = [word for word in words if word not in self.scaffolding] or words
            pairs = [left + "\0" + right for left, right in zip(content, content[1:])]
            spelling = [word[i:i + 3] for token in content for word in ["^" + token + "$"]
                        for i in range(max(1, len(word) - 2))]
            self._question = np.concatenate((
                _hashed_features(content, 192) * np.sqrt(np.float32(.70)),
                _hashed_features(pairs, 64) * np.sqrt(np.float32(.18)),
                _hashed_features(spelling, 64) * np.sqrt(np.float32(.10)),
                _fingerprint(header, 64) * np.sqrt(np.float32(.02)),
            ))
            # Surface guards prevent a near-identical sentence with another
            # number, operator, negation, or quoted symbol borrowing its answer.
            guards = re.findall(r"\d+(?:\.\d+)?|[+*/=<>^%\-]|(?<!\w)['\"][^'\"]+['\"]", text)
            guards += [word for word in words if word in {"not", "no", "never", "without"} or word.endswith(("n't", "n’t"))]
            self._guard = json.dumps(guards, ensure_ascii=False).encode("utf-8")
            self._header = header
        return unit(np.concatenate((self._question, _fingerprint(self._guard + b"\0" + prefix, 128))))

    def close(self):
        super().close()
        self._header = self._question = self._guard = None


def load_native_lessons(checkpoint: Path, capacity: int = DEFAULT_CAPACITY) -> PrototypeResponder:
    encoder = LessonEncoder()
    if not checkpoint.exists():
        return PrototypeResponder(encoder, PatternMemory(max_prototypes=capacity))
    try:
        with np.load(checkpoint, allow_pickle=False) as data:
            identity = json.loads(data["metadata"].item())["identity"]
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise PrototypeError(f"Cannot read lesson memory: {error}") from error
    if identity == LegacyLessonEncoder.identity:
        # Read-only until the next explicit teaching save. Rebuild from verified
        # lesson records, never reinterpret old fingerprints as new features.
        from jayce_lessons import MAX_ANSWER_BYTES, read_state
        from jayce_tokens import answer_context
        state = read_state(checkpoint)
        old = PrototypeResponder(LegacyLessonEncoder(), TokenMemory.load(str(checkpoint), identity, encoder.vocab_size))
        if state is None or (len(old.memory) and not state["lessons"]):
            raise PrototypeError("The legacy memory has no lesson records to rebuild; it was not changed.")
        print("Loading and checking saved lessons with the new learner...", flush=True)
        responder = PrototypeResponder(encoder, PatternMemory(max_prototypes=old.memory.max_prototypes))
        for lesson in state["lessons"]:
            context, answer = answer_context(lesson["question"]), lesson["answer"].strip()
            if old.reply(context, max_tokens=MAX_ANSWER_BYTES).text != answer:
                raise PrototypeError("A legacy lesson failed recall; the saved memory was not changed.")
            responder.teach(context, answer, max_tokens=MAX_ANSWER_BYTES, correct=True)
        for lesson in state["lessons"]:
            if responder.reply(answer_context(lesson["question"]), max_tokens=MAX_ANSWER_BYTES).text != lesson["answer"].strip():
                raise PrototypeError("A rebuilt lesson failed recall; the saved memory was not changed.")
        responder.memory.examples = old.memory.examples
        responder.memory.teacher_examples = old.memory.teacher_examples
        return responder
    memory = TokenMemory.load(str(checkpoint), encoder.identity, encoder.vocab_size)
    if not isinstance(memory, PatternMemory):
        raise PrototypeError("Lesson features require adaptive prototype memory.")
    return PrototypeResponder(encoder, memory)


def load_native_memory(checkpoint: Path) -> RawTextLearner:
    # Trainer metadata supplies the context window, so restart cannot silently
    # reinterpret an existing memory with different features.
    state = read_state(checkpoint)
    if state is None:
        raise PrototypeError("No native memory yet. Run ./jayce train DATA first.")
    if state["model"] != NATIVE_MODEL:
        raise PrototypeError("This memory needs a parent encoder and is incompatible with native text training.")
    return load_text_memory(
        ByteContextEncoder(), str(checkpoint), context_tokens=state["context"],
    )
