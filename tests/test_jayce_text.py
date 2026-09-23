# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_jayce_tokens import FakeEncoder

from jayce_text import RawTextLearner, Scores
from jayce_tokens import END, PrototypeError, TokenMemory


class RawTextTests(unittest.TestCase):
    def test_document_is_the_teacher_and_targets_are_never_in_their_context(self):
        encoder = FakeEncoder()
        learner = RawTextLearner(encoder)
        result = learner.teach("ABC", 10)
        self.assertEqual(encoder.visits, [(65,), (65, 66), (65, 66, 67)])
        self.assertEqual(learner.memory.labels.tolist(), [66, 67, END])
        self.assertEqual(learner.memory.teacher_examples, 0)
        self.assertEqual(result, {"targets": 3, "text_tokens": 2, "complete_document": True})
        self.assertEqual(learner.continue_text("A"), {
            "text": "BC", "tokens": 2, "stop": "document end",
        })

    def test_budget_truncation_does_not_teach_an_ending(self):
        learner = RawTextLearner(FakeEncoder())
        result = learner.teach("ABCDE", 2)
        self.assertFalse(result["complete_document"])
        self.assertEqual(learner.memory.labels.tolist(), [66, 67])
        self.assertEqual(learner.continue_text("A"), {
            "text": "BC", "tokens": 2, "stop": "no strong match",
        })

    def test_ending_needs_its_own_budget_slot(self):
        learner = RawTextLearner(FakeEncoder())
        self.assertFalse(learner.teach("ABC", 2)["complete_document"])
        self.assertNotIn(END, learner.memory.labels)
        self.assertTrue(learner.teach("ABC", 3)["complete_document"])
        self.assertIn(END, learner.memory.labels)

    def test_sliding_context_is_identical_for_training_evaluation_and_generation(self):
        encoder = FakeEncoder()
        learner = RawTextLearner(encoder, context_tokens=2)
        learner.teach("ABCDE", 10)
        self.assertEqual(encoder.visits, [(65,), (65, 66), (66, 67), (67, 68), (68, 69)])
        self.assertEqual(learner.continue_text("ABC")["text"], "DE")
        scores = Scores()
        learner.evaluate("ABCDE", 10, scores)
        self.assertEqual(scores.targets, 5)
        self.assertEqual(scores.accepted_correct, 5)
        self.assertEqual(scores.summary()["accepted_wrong"], 0)

    def test_one_token_context_and_generation_limit(self):
        learner = RawTextLearner(FakeEncoder(), context_tokens=1)
        learner.teach("ABCD", 10)
        self.assertEqual(learner.continue_text("A", 2), {
            "text": "BC", "tokens": 2, "stop": "token limit",
        })

    def test_evaluation_does_not_learn_or_feed_a_prediction_back(self):
        encoder = FakeEncoder()
        learner = RawTextLearner(encoder)
        learner.teach("ABC", 10)
        old_vectors = learner.memory.vectors.copy()
        old_counts = learner.memory.counts.copy()
        encoder.visits.clear()
        scores = Scores()
        learner.evaluate("AXY", 10, scores)
        self.assertEqual(encoder.visits, [(65,), (65, 88), (65, 88, 89)])
        self.assertEqual(scores.summary()["accepted_wrong"], 1)
        self.assertEqual(scores.summary()["abstained"], 2)
        np.testing.assert_array_equal(learner.memory.vectors, old_vectors)
        np.testing.assert_array_equal(learner.memory.counts, old_counts)
        self.assertEqual(learner.memory.examples, 1)

    def test_conflicting_documents_abstain_instead_of_silently_correcting(self):
        learner = RawTextLearner(FakeEncoder())
        learner.teach("ABC", 10)
        learner.teach("AXY", 10)
        self.assertEqual(learner.continue_text("A")["stop"], "no strong match")
        self.assertEqual(learner.continue_text("AB")["text"], "C")

    def test_capacity_failure_preserves_previous_document_and_closes_encoder(self):
        encoder = FakeEncoder()
        learner = RawTextLearner(encoder, TokenMemory(max_prototypes=3))
        learner.teach("AB", 10)
        old_vectors = learner.memory.vectors.copy()
        with self.assertRaises(PrototypeError):
            learner.teach("XYZ", 10)
        self.assertEqual(encoder.current, ())
        np.testing.assert_array_equal(learner.memory.vectors, old_vectors)
        self.assertEqual(learner.memory.examples, 1)

    def test_checkpoints_are_separate_from_qa_and_other_context_settings(self):
        encoder = FakeEncoder()
        learner = RawTextLearner(encoder)
        learner.teach("ABC", 10)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "text.npz")
            learner.save(path)
            restored = TokenMemory.load(path, learner.identity, encoder.vocab_size)
            replay = RawTextLearner(encoder, restored)
            self.assertEqual(replay.continue_text("A")["text"], "BC")
            for identity in (encoder.identity, RawTextLearner(encoder, context_tokens=2).identity):
                with self.assertRaises(PrototypeError):
                    TokenMemory.load(path, identity, encoder.vocab_size)

    def test_invalid_inputs_and_empty_memory(self):
        encoder = FakeEncoder()
        for context in (0, encoder.context_limit + 1):
            with self.assertRaises(PrototypeError):
                RawTextLearner(encoder, context_tokens=context)
        learner = RawTextLearner(encoder)
        for text, budget in (("", 1), ("a", 0)):
            with self.assertRaises(PrototypeError):
                learner.teach(text, budget)
        self.assertEqual(learner.continue_text("ABC")["stop"], "no strong match")
        self.assertIsNone(Scores().summary()["accepted_accuracy"])
