# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""The four default modes share native lessons and corpus memory across sessions."""

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jayce
from jayce_lessons import SharedLessons, question_key
from jayce_native import ByteContextEncoder, load_native_lessons
from jayce_parent_train import run_training, new_state, read_state, teach_topics
from jayce_text import RawTextLearner
from jayce_tokens import PrototypeError, answer_context
from jayce_train import new_state as new_corpus_state, training_lock


class SharedChatTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.checkpoint = self.root / "lessons.npz"

    def parent(self, question, answer="", *, completed=True, args=()):
        def generate(*args, on_token, on_finished):
            on_token(answer)
            on_finished(completed)
            return answer
        with (
            patch("jayce_model.load_model", return_value=(None, object())),
            patch("jayce_model.generate_chat", side_effect=generate) as parent,
            redirect_stdout(io.StringIO()) as output,
        ):
            session = jayce.ChatSession(run_dir=self.root)
            session.learning_enabled = "--no-learning" not in args
            session.answer(question)
        if "--no-learning" in args:
            parent.assert_not_called()
        return output.getvalue()

    def train_parent(self, pairs, capacity=None):
        def bounded(*args, **kwargs):
            return teach_topics(*args, lesson_limit=1, **kwargs)
        with (
            patch("jayce_model.load_model", return_value=(None, object())),
            patch("jayce_parent_train.load_topics", return_value=["Calculus", "French language"]),
            patch("jayce_parent_train.generate_topic_lessons", return_value=pairs),
            patch("jayce_parent_train.teach_topics", side_effect=bounded),
            redirect_stdout(io.StringIO()),
        ):
            run_training(capacity, run_dir=self.root)

    def native(self, question):
        # A fresh process proves this is persisted learning without parent imports.
        script = """
import sys
for name in ('jayce_model', 'torch', 'transformers', 'llama_cpp', 'huggingface_hub'):
    sys.modules[name] = None
from jayce import ChatSession
ChatSession(parent=False, run_dir=sys.argv[1]).answer(sys.argv[2])
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.root), question],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_parent_answer_is_recalled_by_native_after_restart_with_contraction(self):
        answer = "I can't provide real-time weather updates. Check a weather service for Cleveland."
        self.assertIn("after learning", self.parent("What is the weather in cleveland", answer))
        before = self.checkpoint.read_bytes()
        for question in ("What is the weather in cleveland", "What's the weather in cleveland?", "What’s  the weather in cleveland?"):
            self.assertIn(f"Jayce> {answer}", self.native(question))
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertIn("Jayce> Jayce don't know", self.native("What's the weather in boston?"))

    def test_long_chat_answers_are_saved_and_recalled_in_both_modes(self):
        answer = "Cleveland is a city in Ohio. " * 12
        self.parent("Tell me about Cleveland", answer)
        self.assertIn(answer.strip(), self.native("Tell me about Cleveland"))
        self.assertIn(answer.strip(), self.parent("Tell me about Cleveland", args=["--no-learning"]))

    def test_parent_agreement_updates_persisted_prototypes_again_after_restart(self):
        self.parent("Triangle sides?", "Three.")
        first = load_native_lessons(self.checkpoint).memory
        output = self.parent("Triangle sides?", "Three.")
        second = load_native_lessons(self.checkpoint).memory
        self.assertEqual(len(first), len(second))
        self.assertEqual(second.teacher_examples, first.teacher_examples + 1)
        self.assertEqual(second.counts.tolist(), (first.counts + 1).tolist())
        self.assertEqual(len(read_state(self.checkpoint)["lessons"]), 1)
        self.assertIn("practices it again", output)
        self.assertIn("Jayce> Three.", self.native("Triangle sides?"))

    def test_practicing_a_training_answer_in_chat_does_not_make_it_a_chat_only_fact(self):
        shared = SharedLessons(self.checkpoint)
        shared.teach(answer_context("Triangle sides?"), "Three.", source="parent", topic="geometry")
        self.parent("Triangle sides?", "Three.")
        self.assertEqual(read_state(self.checkpoint)["lessons"][0]["source"], "parent")
        self.assertIn("Jayce> Three.", self.native("Triangle sides?"))

    def test_curriculum_and_chat_use_one_memory_without_losing_resume_progress(self):
        state = new_state("teacher", ["geometry"], 2)
        with redirect_stdout(io.StringIO()):
            teach_topics(load_native_lessons(self.checkpoint), state, self.checkpoint,
                         lambda *args: [("Triangle sides?", "Three."), ("Square sides?", "Four.")], lesson_limit=1)
        before = read_state(self.checkpoint)
        self.assertIn("Jayce> Three.", self.parent("Triangle sides?", args=["--no-learning"]))
        self.parent("Where is Cleveland?", "Ohio.")
        after = read_state(self.checkpoint)
        self.assertEqual({k:v for k,v in before.items() if k != "lessons"},
                         {k:v for k,v in after.items() if k != "lessons"})
        with redirect_stdout(io.StringIO()):
            teach_topics(load_native_lessons(self.checkpoint), after, self.checkpoint,
                         lambda *args: self.fail("Must reuse the pending plan"), lesson_limit=1)
        for question, answer in (("Triangle sides?", "Three."), ("Square sides?", "Four."), ("Where is Cleveland?", "Ohio.")):
            self.assertIn(f"Jayce> {answer}", self.native(question))

    def test_two_loaded_sessions_merge_new_lessons_and_see_corrections(self):
        first, second = SharedLessons(self.checkpoint), SharedLessons(self.checkpoint)
        first.teach(answer_context("Code?"), "A", correct=True)
        second.teach(answer_context("Color?"), "Blue", correct=True)
        first.refresh()
        self.assertEqual(first.reply(answer_context("Color?")).text, "Blue")
        second.teach(answer_context("Code?"), "B", correct=True)
        self.assertEqual(first.reply(answer_context("Code?")).text, "B")
        self.assertIn("Jayce> Blue", self.native("Color?"))

    def test_busy_trainer_and_failed_save_do_not_commit_chat_lesson(self):
        memory = SharedLessons(self.checkpoint)
        memory.teach(answer_context("Saved?"), "Yes", correct=True)
        before = self.checkpoint.read_bytes()
        with training_lock(self.root), self.assertRaisesRegex(PrototypeError, "Another trainer"):
            memory.teach(answer_context("New?"), "No")
        with patch("jayce_tokens.os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            memory.teach(answer_context("New?"), "No")
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertIsNone(memory.reply(answer_context("New?")).text)

    def test_no_learning_and_truncated_answers_never_write(self):
        self.parent("Known?", "Yes")
        before = self.checkpoint.read_bytes()
        self.parent("Known?", args=["--no-learning"])
        self.parent("Unknown?", args=["--no-learning"])
        self.parent("Truncated?", "An unfinished answer", completed=False)
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_fresh_native_chat_opens_empty_without_creating_memory(self):
        self.assertIn("Jayce> Jayce don't know", self.native("Hello"))
        self.assertFalse(self.checkpoint.exists())

    def test_chat_can_be_followed_by_first_curriculum_with_custom_settings(self):
        self.parent("Chat fact?", "A")
        self.train_parent([("Triangle sides?", "Three.")])
        self.assertIn("Jayce> A", self.native("Chat fact?"))
        self.assertIn("Jayce> Three.", self.native("Triangle sides?"))

    def test_corpus_memory_is_read_by_both_modes_without_mutation(self):
        learner = RawTextLearner(ByteContextEncoder(), context_tokens=32)
        learner.teach("At Cedar Station the lantern glows amber.", 100)
        checkpoint = self.root / "prototypes.npz"
        state = new_corpus_state(SimpleNamespace(identity="test-corpus"), ByteContextEncoder.identity, 32)
        learner.memory.save(str(checkpoint), learner.identity, training_state=state)
        before = checkpoint.read_bytes()
        self.assertIn("the lantern glows amber.", self.native("At Cedar Station"))
        self.assertIn("the lantern glows amber.", self.parent("At Cedar Station", args=["--no-learning"]))
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_normalization_does_not_collapse_different_cities_or_quoted_letters(self):
        self.assertNotEqual(question_key("What is 'B'?"), question_key("What is 'b'?"))
        self.assertNotEqual(question_key("What's Cleveland?"), question_key("What's Boston?"))


    def test_completed_run_can_grow_and_teach_more_without_resetting_memory(self):
        state = new_state("teacher", ["geometry"], 1)
        with redirect_stdout(io.StringIO()):
            teach_topics(load_native_lessons(self.checkpoint, capacity=4), state, self.checkpoint,
                         lambda *args: [("Q", "A")], lesson_limit=1)
        self.train_parent([("Q2", "B")], capacity=8)
        self.assertEqual(load_native_lessons(self.checkpoint).memory.max_prototypes, 8)
        self.assertEqual(load_native_lessons(self.checkpoint).memory.examples, 2)
        self.assertEqual(read_state(self.checkpoint)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
