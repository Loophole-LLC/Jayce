# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Shared native Q&A memory with atomic, locked updates from every teacher."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from jayce_native import LegacyLessonEncoder, LessonEncoder, load_native_lessons
from jayce_tokens import PrototypeError, PrototypeResponder, answer_context
from jayce_checkpoint import training_lock
from jayce_config import BATCH_SIZE, DEFAULT_TOPICS, PARENT_MODEL

MAX_ANSWER_BYTES = 2048
STATE_FORMAT = "jayce-parent-teaching-v2"


def new_state(model, topics, per_topic=BATCH_SIZE, lessons=None):
    return {
        "format": STATE_FORMAT, "model": model, "topics": list(topics), "per_topic": per_topic,
        "topic_index": 0, "pending": None, "checked": 0, "taught": 0, "known": 0,
        "status": "ready", "lessons": list(lessons or []), "rounds": 0, "no_progress": 0,
    }


def read_state(checkpoint: Path):
    if not checkpoint.exists():
        return None
    try:
        with np.load(checkpoint, allow_pickle=False) as data:
            state = json.loads(data["metadata"].item())["training_state"]
        legacy = state["format"] == "jayce-parent-teaching-v1"
        if state["format"] not in {STATE_FORMAT, "jayce-parent-teaching-v1"}:
            raise ValueError("unsupported teaching checkpoint")
        if not isinstance(state["model"], str) or not isinstance(state["topics"], list) or not state["topics"]:
            raise ValueError("invalid curriculum")
        if any(not isinstance(t, str) or not t.strip() for t in state["topics"]):
            raise ValueError("invalid topic")
        for key in ("topic_index", "checked", "taught", "known", "per_topic"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid {key}")
        if not 1 <= state["per_topic"] <= 10 or not 0 <= state["topic_index"] <= len(state["topics"]):
            raise ValueError("invalid curriculum position")
        if state["checked"] != state["taught"] + state["known"]:
            raise ValueError("invalid lesson counts")
        if state["status"] not in {"ready", "running", "paused", "capacity", "complete", "no_progress"}:
            raise ValueError("invalid teaching status")
        pending = state["pending"]
        if pending is not None:
            pairs, offset = pending["pairs"], pending["offset"]
            if (
                state["topic_index"] >= len(state["topics"])
                or not isinstance(pairs, list) or not pairs
                or type(offset) is not int or not 0 <= offset < len(pairs)
                or any(not isinstance(p, list) or len(p) != 2 or any(
                    not isinstance(v, str) or not v.strip() for v in p
                ) for p in pairs)
            ):
                raise ValueError("invalid pending lesson")
        if not isinstance(state["lessons"], list) or any(
            not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("topic", "question", "answer")
            ) for item in state["lessons"]
        ):
            raise ValueError("invalid lesson catalog")
        if legacy:
            # Preserve every prototype, answer and unconsumed lesson. Old "complete"
            # meant one topic pass ended; it must not prevent future teaching.
            state.update(format=STATE_FORMAT, per_topic=BATCH_SIZE, rounds=0, no_progress=0)
            state["topic_index"] %= len(state["topics"])
            if state["status"] == "complete":
                state["status"] = "paused"
        for key in ("rounds", "no_progress"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid {key}")
        if state["topic_index"] >= len(state["topics"]):
            raise ValueError("invalid topic position")
        return state
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise PrototypeError(f"Cannot resume {checkpoint}: {error}. The saved memory has not been changed.") from error


def question_key(question: str) -> str:
    """Normalize spacing, final question marks, and question-leading 'what's'.

    Preserve case within the question: quoted letters and names can be significant.
    This is a small wording convenience, not semantic paraphrase matching.
    """
    question = " ".join(question.strip().split()).rstrip("?").rstrip()
    question = re.sub(r"^what['’]s\b", "what is", question, flags=re.IGNORECASE)
    return question[:1].lower() + question[1:]


def lesson_context(question: str, lessons: list[dict]) -> str:
    key = question_key(question)
    for lesson in reversed(lessons):
        if question_key(lesson["question"]) == key:
            return answer_context(lesson["question"])
    return answer_context(question.strip())


def save_lessons(responder, checkpoint, state):
    """Called under the writer lock; retain the original v1 file before upgrading."""
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as data:
            identity = json.loads(data["metadata"].item())["identity"]
        if identity == LegacyLessonEncoder.identity:
            backup = checkpoint.with_name(checkpoint.stem + ".v1-backup.npz")
            try:
                # Checkpoints are replaced atomically, so this link retains the
                # old inode even after the new memory replaces the live path.
                os.link(checkpoint, backup)
            except FileExistsError:
                if backup.read_bytes() != checkpoint.read_bytes():
                    raise PrototypeError(f"A different legacy backup already exists at {backup}; memory was not changed.")
    responder.memory.save(str(checkpoint), responder.encoder.identity, training_state=state)


class SharedLessons(PrototypeResponder):
    """Read the same prototypes in chat and native mode; commit each verified lesson.

    Writers reload under the trainer's lock so another session's newer lessons and
    curriculum cursor cannot be replaced by an older in-memory snapshot.
    """

    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint).resolve()
        self._version = None
        self.state = None
        super().__init__(LessonEncoder())
        self.refresh(force=True)

    def _stamp(self):
        try:
            stat = self.checkpoint.stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns
        except FileNotFoundError:
            return None

    def refresh(self, *, force=False):
        if not force and self._stamp() == self._version:
            return
        # A reader needs no lock. Retry if an atomic replacement crosses its two reads.
        for _ in range(3):
            version = self._stamp()
            state = read_state(self.checkpoint)
            responder = load_native_lessons(self.checkpoint)
            if version == self._stamp():
                self.state, self.memory, self._version = state, responder.memory, version
                return
        raise PrototypeError("Lesson memory changed while loading; please retry.")

    @property
    def lessons(self):
        return self.state["lessons"] if self.state else []

    @staticmethod
    def _question(context):
        try:
            question = json.loads(context.removesuffix("\nAnswer:"))["question"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("empty question")
            return question
        except (ValueError, KeyError, TypeError) as error:
            raise PrototypeError("Shared lessons require a standalone question.") from error

    def reply(self, context, max_tokens=MAX_ANSWER_BYTES):
        self.refresh()
        context = lesson_context(self._question(context), self.lessons)
        return super().reply(context, max_tokens=max_tokens)

    def teach(self, context, answer, *, teacher=False, max_tokens=MAX_ANSWER_BYTES,
              correct=True, topic="chat", source="chat"):
        question = self._question(context)
        with training_lock(self.checkpoint.parent):
            # The stamp is checked under the writer lock. Reuse an unchanged
            # verified snapshot, including an in-memory legacy rebuild.
            self.refresh()
            state = self.state or new_state(PARENT_MODEL, DEFAULT_TOPICS)
            # Use the base responder so verified teaching never refreshes midway.
            responder = PrototypeResponder(self.encoder, self.memory)
            try:
                tokens = teach_lesson(responder, state, question, answer, topic=topic,
                                      source=source, teacher=teacher, max_tokens=max_tokens,
                                      correct=correct)
                save_lessons(responder, self.checkpoint, state)
            except Exception:
                self.refresh(force=True)
                raise
            self.state, self._version = state, self._stamp()
            return tokens

    def save(self, path):
        # teach() already commits under the lock. Never write a stale snapshot.
        if path and Path(path).resolve() != self.checkpoint:
            raise PrototypeError("Shared lessons must be saved to their own checkpoint.")
        self.refresh()


def teach_lesson(responder, state, question, answer, *, topic, source="parent",
                 teacher=True, max_tokens=MAX_ANSWER_BYTES, correct=True):
    """Practice, verify full recall, and update the question spelling index."""
    context = lesson_context(question, state["lessons"])
    # Both memory implementations replace arrays on commit. Retaining the old
    # references also makes a failed full-answer verification transactional.
    previous_memory = responder.memory.__dict__.copy()
    try:
        tokens = responder.teach(context, answer, teacher=teacher, max_tokens=max_tokens, correct=correct)
        if responder.reply(context, max_tokens=max_tokens).text != answer.strip():
            raise PrototypeError("Jayce could not recall the answer; that example was not saved.")
    except Exception:
        responder.memory.__dict__.clear()
        responder.memory.__dict__.update(previous_memory)
        raise
    saved_question = SharedLessons._question(context)
    key = question_key(saved_question)
    previous = next((item for item in state["lessons"] if question_key(item["question"]) == key), None)
    state["lessons"] = [item for item in state["lessons"] if question_key(item["question"]) != key]
    record = previous if previous and previous["answer"].strip() == answer.strip() else {
        "topic": topic, "question": saved_question, "answer": answer.strip(), "source": source,
    }
    state["lessons"].append(record)
    return tokens
