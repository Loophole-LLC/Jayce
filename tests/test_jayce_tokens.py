# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jayce_tokens import (
    END,
    PrototypeError,
    PrototypeResponder,
    TokenMemory,
    answer_context,
)


class FakeEncoder:
    """Distinct causal context vectors; deliberately has no model generator."""

    identity = "test-encoder"
    vocab_size = 256
    context_limit = 2048

    def __init__(self):
        self.contexts = {}
        self.current = ()
        self.visits = []
        self.closed = 0

    def encode(self, text, *, prefix=False):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()

    def vector(self):
        self.visits.append(self.current)
        if self.current not in self.contexts:
            self.contexts[self.current] = len(self.contexts)
        vector = np.zeros(512, dtype=np.float32)
        vector[self.contexts[self.current]] = 1
        return vector

    def start(self, tokens):
        self.current = tuple(tokens)
        return self.vector()

    def advance(self, token):
        self.current += (token,)
        return self.vector()

    def close(self):
        self.closed += 1
        self.current = ()


class TokenMemoryTests(unittest.TestCase):
    def test_repeated_context_updates_only_its_own_slot(self):
        memory = TokenMemory(rate=0.5)
        memory.learn([np.array([1.0, 0.0]), np.array([0.0, -1.0])], [1, 2])
        memory.learn([np.array([1.0, 0.001])], [1])
        np.testing.assert_allclose(memory.vectors[0], [0.999999875, 0.0005], atol=1e-7)
        np.testing.assert_array_equal(memory.vectors[1], [0.0, -1.0])
        self.assertEqual(memory.counts.tolist(), [2, 1])

    def test_margin_compares_different_labels_not_duplicate_prototypes(self):
        memory = TokenMemory()
        memory.learn(
            [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            [4, 4, 5],
        )
        self.assertTrue(memory.match(np.array([1.0, 0.0]), 0.98, 0.005).accepted)
        memory.learn([np.array([1.0, 0.0])], [5])
        self.assertFalse(memory.match(np.array([1.0, 0.0]), 0.98, 0.005).accepted)

    def test_weak_and_unseen_queries_abstain(self):
        memory = TokenMemory()
        self.assertIsNone(memory.match(np.array([1.0, 0.0]), 0.98, 0.005))
        memory.learn([np.array([1.0, 0.0])], [4])
        self.assertFalse(memory.match(np.array([0.0, 1.0]), 0.98, 0.005).accepted)

    def test_bad_example_and_capacity_failure_leave_memory_unchanged(self):
        memory = TokenMemory(max_prototypes=2)
        memory.learn([np.array([1.0, 0.0])], [1])
        previous = memory.vectors.copy()
        for vectors, labels in [
            ([np.ones(2), np.ones(2)], [2, 3]),
            ([np.array([np.nan, 0.0])], [1]),
            ([np.ones(3)], [1]),
        ]:
            with self.assertRaises(PrototypeError):
                memory.learn(vectors, labels)
            np.testing.assert_array_equal(memory.vectors, previous)
            self.assertEqual(memory.examples, 1)

    def test_round_trip_and_model_mismatch(self):
        memory = TokenMemory()
        memory.learn(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])], [3, END], teacher=True
        )
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "prototypes.npz")
            memory.save(path, "model-A")
            restored = TokenMemory.load(path, "model-A", 256)
            np.testing.assert_array_equal(restored.vectors, memory.vectors)
            np.testing.assert_array_equal(restored.labels, memory.labels)
            self.assertEqual(restored.teacher_examples, 1)
            with self.assertRaises(PrototypeError):
                TokenMemory.load(path, "model-B", 256)
            with self.assertRaises(PrototypeError):
                TokenMemory.load(path, "model-A", 2)

    def test_failed_atomic_save_preserves_previous_file(self):
        memory = TokenMemory()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prototypes.npz"
            memory.save(str(path), "model-A")
            previous = path.read_bytes()
            with (
                patch("jayce_tokens.os.replace", side_effect=OSError("disk error")),
                self.assertRaises(OSError),
            ):
                memory.save(str(path), "model-A")
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_only_trainer_saves_can_replace_a_training_checkpoint(self):
        memory = TokenMemory()
        memory.learn([np.array([1.0, 0.0])], [3])
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "prototypes.npz")
            memory.save(path, "model-A", training_state={"cursor": 1})
            restored = TokenMemory.load(path, "model-A", 256)
            before = Path(path).read_bytes()
            # Covers both a loaded reader and /reset-jayce's fresh memory object.
            for writer in (restored, TokenMemory()):
                with self.assertRaisesRegex(PrototypeError, "read-only"):
                    writer.save(path, "model-A")
                self.assertEqual(Path(path).read_bytes(), before)
            restored.save(path, "model-A", training_state={"cursor": 2})
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(json.loads(str(data["metadata"].item()))["training_state"], {"cursor": 2})
            # A separate export remains available without changing the trainer.
            restored.save(str(Path(folder) / "export.npz"), "model-A")

    def test_invalid_saved_arrays_are_rejected(self):
        memory = TokenMemory()
        memory.learn([np.array([1.0, 0.0])], [3])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prototypes.npz"
            memory.save(str(path), "model-A")
            with np.load(path, allow_pickle=False) as data:
                original = {name: data[name].copy() for name in data.files}
            metadata = json.loads(str(original["metadata"].item()))
            bad_values = [
                {"vectors": np.ones(2)},
                {"vectors": np.array([[np.nan, 0.0]])},
                {"vectors": np.zeros((1, 2))},
                {"labels": np.array([3.5])},
                {"labels": np.array([256])},
                {"counts": np.array([0])},
                {"counts": np.array([1, 2])},
                {
                    "metadata": np.asarray(
                        json.dumps({**metadata, "teacher_examples": 2})
                    )
                },
            ]
            for changes in bad_values:
                with self.subTest(changes=changes):
                    np.savez_compressed(path, **{**original, **changes})
                    with self.assertRaises(PrototypeError):
                        TokenMemory.load(str(path), "model-A", 256)

    def test_correction_replaces_conflict_without_changing_other_examples(self):
        memory = TokenMemory()
        memory.learn(
            [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ],
            [1, 2, 3],
        )
        memory.learn([np.array([1.0, 0.0, 0.0])], [2], correct=True)
        self.assertNotIn(1, memory.labels)
        self.assertEqual(memory.match(np.array([1.0, 0.0, 0.0]), 0.985, 0.005).token, 2)
        np.testing.assert_array_equal(
            memory.vectors[memory.labels == 3], [[0.0, 0.0, 1.0]]
        )

    def test_full_memory_can_correct_by_reusing_slots(self):
        memory = TokenMemory(max_prototypes=2)
        memory.learn([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [1, END])
        memory.learn(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])], [2, END], correct=True
        )
        self.assertEqual(len(memory), 2)
        self.assertEqual(memory.match(np.array([1.0, 0.0]), 0.985, 0.005).token, 2)

    def test_full_memory_rejects_new_contexts_even_for_an_existing_label(self):
        memory = TokenMemory(max_prototypes=2)
        memory.learn([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [1, END])
        for correct in (False, True):
            with (
                self.subTest(correct=correct),
                self.assertRaisesRegex(PrototypeError, "full"),
            ):
                # A correction frees the first slot, but this distinct ending
                # still needs a third slot. Roll back the entire answer.
                memory.learn(
                    [np.array([1.0, 0.0]), np.array([0.0, -1.0])],
                    [2 if correct else 1, END],
                    correct=correct,
                )
            np.testing.assert_array_equal(memory.labels, [1, END])
            np.testing.assert_array_equal(memory.vectors, [[1, 0], [0, 1]])
            np.testing.assert_array_equal(memory.counts, [1, 1])
            self.assertEqual(memory.examples, 1)

    def test_old_memory_loads_into_shared_budget_without_changing_vectors(self):
        memory = TokenMemory()
        memory.learn([np.array([1.0, 0.0])], [END])
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "old.npz")
            memory.save(path, "model-A")
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            metadata.update(format="jayce-next-token-v2", slots_per_token=4)
            arrays["metadata"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(path, **arrays)
            restored = TokenMemory.load(path, "model-A", 256)
            np.testing.assert_array_equal(restored.vectors, memory.vectors)
            for angle in np.linspace(0.1, 1.0, 10):
                restored.learn([np.array([np.cos(angle), np.sin(angle)])], [END])
            self.assertEqual(len(restored), 11)
            restored.save(path, "model-A")
            self.assertEqual(len(TokenMemory.load(path, "model-A", 256)), 11)


class ResponderTests(unittest.TestCase):
    def setUp(self):
        self.encoder = FakeEncoder()
        self.responder = PrototypeResponder(self.encoder)
        self.context = answer_context("What is the code?")

    def test_twenty_answers_share_token_labels_without_losing_earlier_answers(self):
        # Every answer shares all its token labels, including END. Distinct
        # contexts must use the global budget instead of competing for four slots.
        for correct in (False, True):
            with self.subTest(correct=correct):
                responder = PrototypeResponder(FakeEncoder())
                questions = [f"Fact {number}?" for number in range(20)]
                for number, question in enumerate(questions, 1):
                    responder.teach(answer_context(question), "Yes.", correct=correct)
                    if number in (4, 5, 8, 12, 20):
                        for earlier in questions[:number]:
                            self.assertEqual(
                                responder.reply(answer_context(earlier)).text, "Yes."
                            )
                with tempfile.TemporaryDirectory() as folder:
                    path = str(Path(folder) / "prototypes.npz")
                    responder.save(path)
                    restored = TokenMemory.load(
                        path, responder.encoder.identity, responder.encoder.vocab_size
                    )
                    responder = PrototypeResponder(responder.encoder, restored)
                    size = len(restored)
                    for question in questions:
                        self.assertEqual(
                            responder.reply(answer_context(question)).text, "Yes."
                        )
                        responder.teach(
                            answer_context(question), "Yes.", correct=correct
                        )
                    self.assertEqual(len(restored), size)

    def test_training_uses_only_prior_tokens_and_a_learned_ending(self):
        self.responder.teach(self.context, "42")
        prefix = tuple(self.context.encode())
        self.assertEqual(
            self.encoder.visits,
            [prefix, prefix + (ord("4"),), prefix + (ord("4"), ord("2"))],
        )
        self.assertEqual(
            self.responder.memory.labels.tolist(), [ord("4"), ord("2"), END]
        )
        reply = self.responder.reply(self.context)
        self.assertEqual(reply.text, "42")
        self.assertEqual(reply.tokens, 2)
        self.assertEqual(self.responder.memory.examples, 1)  # Answering is read-only.

    def test_empty_unrelated_ambiguous_and_partial_answers_are_withheld(self):
        self.assertIsNone(self.responder.reply(self.context).text)
        self.responder.teach(self.context, "42")
        self.assertIsNone(self.responder.reply(answer_context("Who?")).text)
        self.assertIsNone(self.responder.reply(self.context, max_tokens=1).text)
        self.assertEqual(self.responder.reply(self.context, max_tokens=2).text, "42")
        # Contradictory training does not silently pick whichever was stored first.
        self.responder.teach(self.context, "43")
        self.assertIsNone(self.responder.reply(self.context).text)

    def test_oversized_training_is_rejected_without_partial_learning(self):
        with self.assertRaises(PrototypeError):
            self.responder.teach(self.context, "123", max_tokens=2)
        self.assertEqual(len(self.responder.memory), 0)
        self.encoder.context_limit = 2
        with self.assertRaises(PrototypeError):
            self.responder.teach(self.context, "1")
        self.assertEqual(len(self.responder.memory), 0)

    def test_wrong_answer_is_corrected_and_replayed_without_teacher(self):
        self.responder.teach(self.context, "A")
        self.assertEqual(self.responder.reply(self.context).text, "A")
        self.responder.teach(self.context, "B", correct=True, teacher=True)
        self.assertEqual(self.responder.reply(self.context).text, "B")
        self.assertEqual(self.responder.reply(self.context).text, "B")
        self.assertEqual(self.responder.memory.teacher_examples, 1)

    def test_encoder_failure_does_not_half_train(self):
        with (
            patch.object(
                self.encoder, "advance", side_effect=RuntimeError("encoding failed")
            ),
            self.assertRaises(RuntimeError),
        ):
            self.responder.teach(self.context, "42")
        self.assertEqual(len(self.responder.memory), 0)
        self.assertGreater(self.encoder.closed, 0)

    def test_persistence_stores_prototypes_not_teacher_answer_strings(self):
        self.responder.teach(self.context, "42")
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "p.npz")
            self.responder.memory.save(path, self.encoder.identity)
            restored = TokenMemory.load(
                path, self.encoder.identity, self.encoder.vocab_size
            )
            reply = PrototypeResponder(self.encoder, restored).reply(self.context)
            self.assertEqual(reply.text, "42")
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(
                    set(data.files), {"metadata", "vectors", "labels", "counts"}
                )

    def test_previously_learned_question_context_still_matches(self):
        previous_context = '{"notes":"","question":"What is the code?"}\nAnswer:'
        self.responder.teach(previous_context, "42")
        self.assertEqual(
            self.responder.reply(answer_context("What is the code?")).text, "42"
        )
