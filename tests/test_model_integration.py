# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Optional checks against a real local model: set JAYCE_TEST_MODEL to opt in."""

import os
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jayce_config import PARENT_MODEL
from jayce_model import generate_chat, load_model
from jayce_native import LessonEncoder
from jayce_tokens import PrototypeResponder, TokenMemory, answer_context


@unittest.skipUnless(
    os.environ.get("JAYCE_TEST_MODEL"), "Set JAYCE_TEST_MODEL for real-model checks"
)
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.name = os.environ["JAYCE_TEST_MODEL"]
        if cls.name == "default":
            cls.name = PARENT_MODEL
        cls.tokenizer, cls.model = load_model(cls.name)


    def test_default_parent_teaches_correct_wheel_counts(self):
        if self.name != PARENT_MODEL:
            self.skipTest("Basic factual regression for the default parent")
        encoder = LessonEncoder()
        responder = PrototypeResponder(encoder)
        history = [
            {"role": "user", "content": "hello?"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"},
        ]
        questions = (
            ("how many wheels does a skateboard have?", r"\b(4|four)\b"),
            ("how many wheels does a bicycle have?", r"\b(2|two)\b"),
        )
        learned = []
        for question, expected in questions:
            with self.subTest(question=question):
                finished = []
                answer = generate_chat(
                    self.tokenizer, self.model, history, question, 32,
                    on_finished=finished.append,
                )
                self.assertEqual(finished, [True])
                self.assertRegex(answer.lower(), expected)
                context = answer_context(question)
                responder.teach(context, answer, teacher=True)
                self.assertEqual(responder.reply(context).text, answer)
                learned.append((context, answer))
                history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "prototypes.npz")
            responder.save(path)
            restored = TokenMemory.load(path, encoder.identity, encoder.vocab_size)
        reloaded = PrototypeResponder(encoder, restored)
        for context, answer in learned:
            self.assertEqual(reloaded.reply(context).text, answer)

    def test_real_correction_persistence_and_abstention(self):
        encoder = LessonEncoder()
        responder = PrototypeResponder(encoder)
        context = answer_context("What is this letter 'B'?")
        responder.teach(context, "A")
        self.assertEqual(responder.reply(context).text, "A")
        responder.teach(context, "B", correct=True, teacher=True)
        self.assertEqual(responder.reply(context).text, "B")
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "prototypes.npz")
            responder.memory.save(path, encoder.identity)
            restored = TokenMemory.load(path, encoder.identity, encoder.vocab_size)
        responder = PrototypeResponder(encoder, restored)
        self.assertEqual(responder.reply(context).text, "B")
        self.assertIsNone(
            responder.reply(answer_context("Who invented the telephone?")).text
        )
        self.assertEqual(responder.memory.teacher_examples, 1)

    def test_generated_lessons_repeat_and_recall_in_a_parent_free_process(self):
        from jayce_lessons import new_state, read_state
        from jayce_native import load_native_lessons
        from jayce_parent_train import generate_topic_lessons, teach_topics

        with patch("jayce_parent_train.random.randint", return_value=3):
            pairs = generate_topic_lessons(self.tokenizer, self.model, "basic arithmetic", 1)
        self.assertGreaterEqual(len(pairs), 2, "The parent must supply distinct question wordings")
        self.assertEqual(len({answer for _, answer in pairs}), 1)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "lessons.npz"
            state = new_state(self.name, ["basic arithmetic"])
            plan = pairs + [pairs[0]]
            with redirect_stdout(io.StringIO()):
                teach_topics(load_native_lessons(checkpoint), state, checkpoint,
                             lambda *args: plan, lesson_limit=len(plan))
            self.assertEqual(load_native_lessons(checkpoint).memory.teacher_examples, len(plan))
            self.assertEqual(read_state(checkpoint)["taught"], len(plan))
            script = """
import json, sys
for name in ('jayce_model', 'torch', 'transformers', 'llama_cpp', 'huggingface_hub'):
    sys.modules[name] = None
from jayce_lessons import SharedLessons
from jayce_tokens import answer_context
from pathlib import Path
memory = SharedLessons(Path(sys.argv[1]))
for question, answer in json.loads(sys.argv[2]):
    assert memory.reply(answer_context(question)).text == answer.strip()
"""
            expected = list(dict(plan).items())  # The latest wording wins for each question.
            result = subprocess.run([sys.executable, "-c", script, str(checkpoint), json.dumps(expected)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_parent_generation_and_streaming(self):
        for streaming in (False, True):
            finished, pieces = [], []
            answer = generate_chat(
                self.tokenizer,
                self.model,
                [],
                "What is this letter 'B'? Reply with only the letter.",
                16,
                on_token=pieces.append if streaming else None,
                on_finished=finished.append,
            )
            # Parents can punctuate the same correct answer differently.
            self.assertIn(answer, ("B", "B."))
            self.assertEqual(finished, [True])
            if streaming:
                self.assertEqual("".join(pieces).strip(), answer)

    def test_parent_generates_question_variants_and_contrasting_facts(self):
        from jayce_parent_train import generate_topic_lessons

        with patch("jayce_parent_train.random.randint", return_value=2):
            pairs = generate_topic_lessons(self.tokenizer, self.model, "capitals of European countries", 2)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[0][1], pairs[1][1])
        self.assertEqual(pairs[2][1], pairs[3][1])
        self.assertNotEqual(pairs[0][1], pairs[2][1], "Neighboring facts should have contrasting answers")
        responder = PrototypeResponder(LessonEncoder())
        for question, answer in pairs:
            responder.teach(answer_context(question), answer, correct=True)
        for question, answer in pairs:
            self.assertEqual(responder.reply(answer_context(question)).text, answer)


if __name__ == "__main__":
    unittest.main()
