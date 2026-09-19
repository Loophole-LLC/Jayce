# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Optional checks against a real local model: set JAYCE_TEST_MODEL to opt in."""

import os
import tempfile
import unittest
from pathlib import Path

from jayce_config import PARENT_MODEL
from jayce_model import ContextEncoder, generate_chat, load_model
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
        encoder = ContextEncoder(self.tokenizer, self.model, self.name)
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
                    self.tokenizer, self.model, history, question, 32, self.name,
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
        encoder = ContextEncoder(self.tokenizer, self.model, self.name)
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

    def test_parent_generation_and_streaming(self):
        for streaming in (False, True):
            finished, pieces = [], []
            answer = generate_chat(
                self.tokenizer,
                self.model,
                [],
                "What is this letter 'B'? Reply with only the letter.",
                16,
                self.name,
                on_token=pieces.append if streaming else None,
                on_finished=finished.append,
            )
            # Parents can punctuate the same correct answer differently.
            self.assertIn(answer, ("B", "B."))
            self.assertEqual(finished, [True])
            if streaming:
                self.assertEqual("".join(pieces).strip(), answer)


if __name__ == "__main__":
    unittest.main()
