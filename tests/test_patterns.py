# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.eval_patterns import evaluate
from jayce_lessons import SharedLessons, new_state, read_state, teach_lesson
from jayce_native import LegacyLessonEncoder, LessonEncoder, load_native_lessons
from jayce_patterns import PatternMemory, QUESTION_DIMENSIONS, prefix_key
from jayce_tokens import MemoryFull, PrototypeError, PrototypeResponder, answer_context


class PatternTests(unittest.TestCase):
    def setUp(self):
        self.responder = PrototypeResponder(LessonEncoder())
        self.first = "What is the capital of France?"
        self.second = "What is France's capital?"

    def teach(self, question, answer, **kwargs):
        return self.responder.teach(answer_context(question), answer, **kwargs)

    def reply(self, question):
        return self.responder.reply(answer_context(question)).text

    def test_related_wordings_move_and_share_prototypes(self):
        self.teach(self.first, "Paris.")
        first = self.responder.memory.vectors.copy()
        count = len(self.responder.memory)
        self.teach(self.second, "Paris.")
        self.assertEqual(len(self.responder.memory), count)
        self.assertFalse(np.array_equal(first, self.responder.memory.vectors))
        self.assertEqual(len(self.responder.memory.samples), 2)
        self.assertEqual(self.reply("Please name France's capital."), "Paris.")
        self.assertEqual(self.reply(self.first), "Paris.")
        self.assertEqual(self.reply(self.second), "Paris.")
        self.assertIsNone(self.reply("What is the capital of Italy?"))

    def test_shared_cluster_correction_detaches_only_the_corrected_question(self):
        self.second = "Name the capital of France."
        self.teach(self.first, "A")
        self.teach(self.second, "A")
        self.assertEqual(len(self.responder.memory), 2)
        self.teach(self.first, "B", correct=True)
        self.assertEqual(self.reply(self.first), "B")
        self.assertEqual(self.reply(self.second), "A")
        # Closely related, conflicting answers must make new wording ambiguous.
        self.assertIsNone(self.reply("Please name the capital of France."))

    def test_repetition_weights_the_centroid_without_adding_prototypes(self):
        self.teach(self.first, "A")
        self.teach(self.second, "A")
        before = self.responder.memory.vectors[0].copy()
        self.teach(self.second, "A")
        encoder = LessonEncoder()
        vector = encoder.start(encoder.encode(answer_context(self.second), prefix=True))
        self.assertGreater(float(self.responder.memory.vectors[0] @ vector), float(before @ vector))
        np.testing.assert_array_equal(self.responder.memory.counts, [3, 3])
        self.assertEqual(len(self.responder.memory), 2)

    def test_failed_correction_at_capacity_rolls_back_every_array_and_index(self):
        self.responder.memory.max_prototypes = 2
        self.teach(self.first, "A")
        self.teach(self.second, "A")
        before = {key: value.copy() for key, value in self.responder.memory.checkpoint_arrays().items()}
        with self.assertRaises(MemoryFull):
            self.teach(self.first, "B", correct=True)
        for key, value in before.items():
            np.testing.assert_array_equal(self.responder.memory.checkpoint_arrays()[key], value)
        self.assertEqual(self.reply(self.first), "A")
        self.assertEqual(self.reply(self.second), "A")
        self.assertEqual(self.responder.memory.examples, 2)

    def test_private_correction_reuses_slots_and_removes_old_answer_path(self):
        self.responder.memory.max_prototypes = 2
        self.teach(self.first, "A")
        self.teach(self.first, "B", correct=True)
        self.assertEqual(self.reply(self.first), "B")
        self.assertEqual(len(self.responder.memory), 2)
        encoder = LessonEncoder()
        context = answer_context(self.first) + "A"
        self.assertIsNone(self.responder.memory.match(encoder.start(encoder.encode(context, prefix=True))))

    def test_conflicting_uncorrected_labels_abstain_until_corrected(self):
        self.teach(self.first, "A")
        self.teach(self.first, "B")
        self.assertIsNone(self.reply(self.first))
        self.teach(self.first, "C", correct=True)
        self.assertEqual(self.reply(self.first), "C")
        self.assertEqual(len(self.responder.memory), 2)

    def test_same_next_byte_does_not_merge_unrelated_questions(self):
        self.teach("Where is Cleveland?", "Ohio.")
        before = len(self.responder.memory)
        self.teach("What is an oxygen atom?", "One type of atom.")
        self.assertGreater(len(self.responder.memory), before + len("One type of atom.") - 1)
        self.assertEqual(self.reply("Where is Cleveland?"), "Ohio.")

    def test_numbers_negation_operators_and_quoted_case_are_distinct(self):
        self.teach("What is 2 + 3?", "5.")
        self.teach("What is this letter 'B'?", "Uppercase.")
        self.teach("Which city is the capital of France?", "Paris.")
        for question in ("What is 2 + 4?", "What is 2 - 3?", "What is this letter 'b'?",
                         "Which city is not the capital of France?"):
            with self.subTest(question=question):
                self.assertIsNone(self.reply(question))

    def test_answer_prefix_remains_causal_at_partial_unicode_boundaries(self):
        encoder = LessonEncoder()
        prefix = encoder.encode(answer_context("Comment dit-on school en français?"), prefix=True)
        vector = encoder.start(prefix)
        first_key = prefix_key(vector)
        for token in "École 🌻".encode():
            prefix.append(token)
            vector = encoder.advance(token)
            fresh = LessonEncoder().start(prefix)
            np.testing.assert_array_equal(vector, fresh)
            self.assertNotEqual(prefix_key(vector), first_key)
        self.teach("Comment dit-on school en français?", "École 🌻")
        self.assertEqual(self.reply("Comment dit-on school en français?"), "École 🌻")

    def test_question_order_is_part_of_features(self):
        encoder = LessonEncoder()
        left = encoder.start(encoder.encode(answer_context("The dog chased the cat?"), prefix=True))
        right = encoder.start(encoder.encode(answer_context("The cat chased the dog?"), prefix=True))
        self.assertLess(float(2 * left[:QUESTION_DIMENSIONS] @ right[:QUESTION_DIMENSIONS]), .84)

    def test_failed_recall_verification_rolls_back_in_memory(self):
        self.teach(self.first, "A")
        state = new_state("teacher", ["geography"])
        with patch.object(self.responder, "reply", return_value=type("Reply", (), {"text": None})()):
            with self.assertRaisesRegex(PrototypeError, "could not recall"):
                teach_lesson(self.responder, state, self.first, "B", topic="geography")
        self.assertEqual(self.reply(self.first), "A")
        self.assertEqual(state["lessons"], [])

    def test_fixed_evaluation_reports_transfer_without_claiming_new_rules(self):
        dataset = json.loads((Path(__file__).resolve().parents[1] / "benchmark/patterns.json").read_text())
        old, new = evaluate(dataset, LegacyLessonEncoder()), evaluate(dataset, LessonEncoder())
        self.assertEqual(new["groups"]["replay"]["correct"], len(dataset["train"]))
        self.assertGreaterEqual(new["groups"]["paraphrase"]["correct"], 5)
        self.assertEqual(new["groups"]["contrast"]["correct"], 8)
        self.assertEqual(sum(group["wrong"] for group in new["groups"].values()), 0)
        self.assertEqual(new["groups"]["reasoning"]["abstained"], 2)
        self.assertLess(new["prototypes"], old["prototypes"])
        self.assertLess(new["array_bytes"], old["array_bytes"])


class PatternPersistenceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "lessons.npz"

    def legacy_checkpoint(self):
        state = new_state("teacher", ["geography"])
        old = PrototypeResponder(LegacyLessonEncoder())
        for question in ("What is the capital of France?", "Name the capital of France."):
            teach_lesson(old, state, question, "Paris.", topic="geography")
        state.update(checked=2, taught=2, pending={"pairs": [["What is the capital of Germany?", "Berlin."]], "offset": 0})
        old.memory.save(str(self.path), old.encoder.identity, training_state=state)
        return self.path.read_bytes(), state

    def test_legacy_migration_is_read_only_until_teaching_then_backs_up_original(self):
        before, state = self.legacy_checkpoint()
        memory = SharedLessons(self.path)
        self.assertIsInstance(memory.memory, PatternMemory)
        self.assertEqual(memory.reply(answer_context("Please name the capital of France.")).text, "Paris.")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.path.with_name("lessons.v1-backup.npz").exists())
        memory.teach(answer_context("What is the capital of Germany?"), "Berlin.")
        self.assertEqual(self.path.with_name("lessons.v1-backup.npz").read_bytes(), before)
        self.assertEqual(read_state(self.path)["pending"], state["pending"])
        self.assertEqual(load_native_lessons(self.path).memory.examples, 3)
        self.assertEqual(SharedLessons(self.path).reply(answer_context("Name the capital of France.")).text, "Paris.")

    def test_invalid_legacy_record_does_not_replace_the_checkpoint(self):
        self.legacy_checkpoint()
        with np.load(self.path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        meta = json.loads(arrays["metadata"].item())
        meta["training_state"]["lessons"][0]["answer"] = "Wrong."
        arrays["metadata"] = np.asarray(json.dumps(meta))
        np.savez_compressed(self.path, **arrays)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(PrototypeError, "legacy lesson failed recall"):
            load_native_lessons(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_incomplete_checkpoint_has_a_readable_error(self):
        np.savez_compressed(self.path, unrelated=np.array([1]))
        with self.assertRaisesRegex(PrototypeError, "Cannot read lesson memory"):
            load_native_lessons(self.path)

    def test_corrupt_memberships_and_centers_are_rejected(self):
        responder = load_native_lessons(self.path)
        responder.teach(answer_context("Favorite color?"), "Blue")
        responder.save(str(self.path))
        with np.load(self.path, allow_pickle=False) as data:
            original = {key: data[key] for key in data.files}
        for damage in ("member", "count", "center", "sample"):
            arrays = {key: value.copy() for key, value in original.items()}
            if damage == "member":
                arrays["members"][0, 1] = 999
            elif damage == "count":
                arrays["members"][0, 2] = 0
            elif damage == "center":
                arrays["vectors"][0, :QUESTION_DIMENSIONS] *= -1
            else:
                arrays["samples"][0, 0] = np.nan
            np.savez_compressed(self.path, **arrays)
            with self.subTest(damage=damage), self.assertRaises(PrototypeError):
                load_native_lessons(self.path)

    def test_fresh_process_reloads_and_generalizes_without_parent_imports(self):
        memory = SharedLessons(self.path)
        memory.teach(answer_context("What is the capital of France?"), "Paris.")
        memory.teach(answer_context("What is France's capital?"), "Paris.")
        before = self.path.read_bytes()
        script = """
import sys
for name in ('jayce_model', 'torch', 'transformers', 'llama_cpp', 'huggingface_hub'):
    sys.modules[name] = None
from pathlib import Path
from jayce_lessons import SharedLessons
from jayce_tokens import answer_context
memory = SharedLessons(Path(sys.argv[1]))
assert memory.reply(answer_context("Please name France's capital.")).text == 'Paris.'
assert memory.reply(answer_context('What is the capital of Italy?')).text is None
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.path)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
