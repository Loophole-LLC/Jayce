# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jayce_native import ByteContextEncoder, NATIVE_MODEL, load_native_memory
from jayce_text import RawTextLearner, Scores
from jayce_tokens import PrototypeError
from jayce_train import ParquetCorpus, run_training, train, new_state, read_state
from jayce import ChatSession
import threading

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


class NativeEncoderTests(unittest.TestCase):
    def test_unicode_round_trip_and_partial_byte_output(self):
        encoder = ByteContextEncoder()
        text = "Café — 東京 🌻"
        self.assertEqual(encoder.decode(encoder.encode(text, prefix=True)), text)
        self.assertEqual(encoder.decode([0xF0, 0x9F]), "")
        self.assertEqual(encoder.encode("A", prefix=True), [256, 65])

    def test_context_features_are_repeatable_ordered_and_always_normalized(self):
        encoder = ByteContextEncoder()
        for token in range(encoder.vocab_size):
            self.assertAlmostEqual(float(np.linalg.norm(encoder.start([token]))), 1.0, places=6)
        a = encoder.start(encoder.encode("red door"))
        b = ByteContextEncoder().start(encoder.encode("red door"))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, encoder.start(encoder.encode("door red"))))
        self.assertFalse(np.array_equal(a, encoder.start(encoder.encode("red floor"))))

    def test_incremental_context_matches_a_fresh_encoder(self):
        encoder = ByteContextEncoder()
        tokens = encoder.encode("the amber lantern", prefix=True)
        encoder.start(tokens[:-1])
        incremental = encoder.advance(tokens[-1])
        np.testing.assert_array_equal(incremental, ByteContextEncoder().start(tokens))
        encoder.close()
        self.assertEqual(encoder.history, [])
        for tokens in ([], [257], [-1], [1] * 129):
            with self.assertRaises(PrototypeError):
                encoder.start(tokens)

    def test_training_improves_replay_without_treating_unseen_text_as_learned(self):
        learner = RawTextLearner(ByteContextEncoder(), context_tokens=32)
        text = "At Cedar Station the lantern glows amber."
        before = Scores()
        learner.evaluate(text, 100, before)
        self.assertEqual(before.accepted_correct, 0)
        learner.teach(text, 100)
        after = Scores()
        learner.evaluate(text, 100, after)
        self.assertEqual(after.accepted_correct, after.targets)
        self.assertEqual(learner.continue_text("At Cedar Station")["text"], " the lantern glows amber.")
        self.assertEqual(learner.continue_text("An unseen purple spaceship")["stop"], "no strong match")
        self.assertEqual(learner.memory.teacher_examples, 0)


@unittest.skipIf(pa is None, "Install the corpus extra for native training checks")
class NativeCorpusTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.data = self.root / "fineweb-edu"
        self.shards = self.data / "sample/10BT"
        self.shards.mkdir(parents=True)
        self.run_dir = self.data / "native"
        self.checkpoint = self.run_dir / "prototypes.npz"
        self.text = "At Cedar Station the lantern glows amber."
        pq.write_table(pa.table({"text": [self.text]}), self.shards / "000.parquet")

    def test_recursive_discovery_preserves_existing_corpus_identity(self):
        root = ParquetCorpus(self.data)
        nested = ParquetCorpus(self.shards)
        self.assertEqual(root.files, nested.files)
        self.assertEqual(root.identity, nested.identity)

    def run_cli_training(self):
        with patch.dict(sys.modules, {"jayce_model": None}), redirect_stdout(io.StringIO()) as output:
            run_training(self.data, run_dir=self.run_dir)
        self.assertIn("No parent model is loaded", output.getvalue())

    def test_native_cli_resumes_then_reloads_its_saved_context_without_a_parent(self):
        corpus = ParquetCorpus(self.data)
        learner = RawTextLearner(ByteContextEncoder(), context_tokens=16)
        with redirect_stdout(io.StringIO()):
            train(learner, corpus, new_state(corpus, NATIVE_MODEL, 16), self.checkpoint,
                  batch_tokens=8, max_batches=1, sleep_seconds=0, stop=threading.Event())
        self.assertEqual(read_state(self.checkpoint)["targets"], 8)
        self.run_cli_training()
        state = read_state(self.checkpoint)
        self.assertEqual(state["model"], NATIVE_MODEL)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["targets"], len(self.text.encode()) + 1)
        before = self.checkpoint.read_bytes()
        learner = load_native_memory(self.checkpoint)
        self.assertEqual(learner.context_tokens, 16)
        np.testing.assert_array_equal(learner.memory.counts, np.ones(len(learner.memory)))
        with (
            patch.dict(sys.modules, {"jayce_model": None}),
            redirect_stdout(io.StringIO()) as output,
        ):
            ChatSession(parent=False, run_dir=self.run_dir).answer("At Cedar Station")
        self.assertIn("the lantern glows amber.", output.getvalue())
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_fresh_process_trains_and_answers_with_parent_imports_blocked(self):
        script = """
import sys
for name in ('jayce_model', 'torch', 'transformers', 'llama_cpp', 'huggingface_hub'):
    sys.modules[name] = None
from jayce_train import run_training
from jayce import ChatSession
data_path, run_path = sys.argv[1:]
run_training(data_path, run_dir=run_path)
ChatSession(parent=False, run_dir=run_path).answer('At Cedar Station')
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.data), str(self.run_dir)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("the lantern glows amber.", result.stdout)

    def test_native_loader_rejects_parent_checkpoint_without_changing_it(self):
        learner = RawTextLearner(ByteContextEncoder(), context_tokens=32)
        state = new_state(ParquetCorpus(self.data), "parent-model", 32)
        learner.memory.save(str(self.checkpoint), learner.identity, training_state=state)
        before = self.checkpoint.read_bytes()
        with self.assertRaisesRegex(PrototypeError, "needs a parent"):
            load_native_memory(self.checkpoint)
        self.assertEqual(self.checkpoint.read_bytes(), before)


    def test_new_data_source_keeps_the_old_memory_and_learns_the_new_text(self):
        other = self.root / "other.parquet"
        pq.write_table(pa.table({"text": ["Beside Maple Bridge a violet kite floats."]}), other)
        for source in (self.data, other):
            with (
                patch.dict(sys.modules, {"jayce_model": None}),
                redirect_stdout(io.StringIO()),
            ):
                run_training(source, run_dir=self.run_dir)
        learner = load_native_memory(self.checkpoint)
        self.assertEqual(learner.continue_text("At Cedar Station")["text"], " the lantern glows amber.")
        self.assertEqual(learner.continue_text("Beside Maple Bridge")["text"], " a violet kite floats.")


if __name__ == "__main__":
    unittest.main()
