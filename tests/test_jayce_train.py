# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import io
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_jayce_tokens import FakeEncoder

from jayce_text import RawTextLearner, load_text_memory
from jayce_tokens import END, PrototypeError
from jayce_train import (
    ParquetCorpus,
    new_state,
    read_state,
    train,
    training_lock,
    validate_resume,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


class TokenBatchTests(unittest.TestCase):
    def test_batches_keep_original_context_and_only_the_real_document_end(self):
        encoder = FakeEncoder()
        learner = RawTextLearner(encoder, context_tokens=2)
        tokens = encoder.encode("ABCDE")
        learner.teach_tokens(tokens, 2)
        learner.teach_tokens(tokens, 2, offset=2)
        self.assertNotIn(END, learner.memory.labels)
        learner.teach_tokens(tokens, 2, offset=4)
        self.assertEqual(encoder.visits, [(65,), (65, 66), (66, 67), (67, 68), (68, 69)])
        self.assertEqual(learner.memory.labels.tolist(), [66, 67, 68, 69, END])
        self.assertEqual(learner.continue_text("ABC")["text"], "DE")


@unittest.skipIf(pa is None, "Install the corpus extra for Parquet training tests")
class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.checkpoint = self.root / "run/prototypes.npz"
        self.encoder = FakeEncoder()

    def corpus(self, *shards):
        for number, rows in enumerate(shards):
            pq.write_table(
                pa.table({"text": pa.array(rows, type=pa.string())}),
                self.data / f"{number:03}.parquet", row_group_size=2,
            )
        return ParquetCorpus(self.data)

    def learner(self, capacity=100):
        return load_text_memory(self.encoder, str(self.checkpoint), context_tokens=2, capacity=capacity)

    def run_loop(self, learner, corpus, state=None, max_batches=0, stop=None):
        with redirect_stdout(io.StringIO()):
            return train(
                learner, corpus, state or read_state(self.checkpoint) or new_state(corpus, "fake", 2),
                self.checkpoint, batch_tokens=2, sleep_seconds=0.01, max_batches=max_batches,
                stop=stop or threading.Event(),
            )

    def test_resume_mid_document_then_cross_row_groups_and_shards_without_reteaching(self):
        corpus = self.corpus(["ABCDE", None, "", "XY"], ["1234"])
        self.assertEqual(self.run_loop(self.learner(), corpus, max_batches=1), "paused")
        self.assertEqual(read_state(self.checkpoint)["cursor"]["offset"], 2)
        self.assertEqual(self.run_loop(self.learner(), corpus, max_batches=1), "paused")
        self.assertEqual(read_state(self.checkpoint)["cursor"]["offset"], 4)
        learner = self.learner()
        self.assertEqual(self.run_loop(learner, corpus), "complete")
        state = read_state(self.checkpoint)
        self.assertEqual((state["targets"], state["documents"], state["skipped_empty"]), (11, 3, 2))
        self.assertEqual(learner.memory.labels.tolist(), [66, 67, 68, 69, END, 89, END, 50, 51, 52, END])
        np.testing.assert_array_equal(learner.memory.counts, np.ones(11))
        self.assertEqual(learner.continue_text("ABC")["text"], "DE")
        before = self.checkpoint.read_bytes()
        self.assertEqual(self.run_loop(learner, corpus), "complete")
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_full_memory_stops_and_larger_capacity_resumes_at_the_next_token(self):
        corpus = self.corpus(["ABCDE"])
        self.assertEqual(self.run_loop(self.learner(capacity=3), corpus), "capacity")
        state = read_state(self.checkpoint)
        self.assertEqual((state["targets"], state["cursor"]["offset"]), (3, 3))
        learner = self.learner()
        learner.memory.max_prototypes = 6
        self.assertEqual(self.run_loop(learner, corpus), "complete")
        self.assertEqual(learner.memory.labels.tolist(), [66, 67, 68, 69, END])
        np.testing.assert_array_equal(learner.memory.counts, np.ones(5))

    def test_atomic_save_failure_preserves_both_memory_and_position(self):
        corpus = self.corpus(["ABCDE"])
        learner = self.learner()
        self.run_loop(learner, corpus, max_batches=1)
        before = self.checkpoint.read_bytes()
        learner.teach_tokens(self.encoder.encode("ABCDE"), 2, offset=2)
        state = read_state(self.checkpoint)
        state["cursor"]["offset"] = 4
        with patch("jayce_tokens.os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            learner.memory.save(str(self.checkpoint), learner.identity, training_state=state)
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertEqual(read_state(self.checkpoint)["cursor"]["offset"], 2)
        self.assertEqual(len(self.learner().memory), 2)
        self.run_loop(self.learner(), corpus)
        np.testing.assert_array_equal(self.learner().memory.counts, np.ones(5))

    def test_stop_during_batch_saves_it_and_skips_sleep(self):
        corpus = self.corpus(["ABCDE"])
        learner = self.learner()
        stop = threading.Event()
        teach = learner.teach_tokens

        def stop_after_teaching(*args, **kwargs):
            result = teach(*args, **kwargs)
            stop.set()
            return result

        with patch.object(learner, "teach_tokens", side_effect=stop_after_teaching), patch.object(stop, "wait") as wait:
            self.assertEqual(self.run_loop(learner, corpus, stop=stop), "paused")
        wait.assert_not_called()
        self.assertEqual(read_state(self.checkpoint)["targets"], 2)

    def test_sleep_occurs_between_saved_batches_and_can_stop_the_run(self):
        corpus = self.corpus(["ABCDE"])
        stop = threading.Event()

        def pause(seconds):
            self.assertEqual(seconds, 0.01)
            self.assertEqual(read_state(self.checkpoint)["targets"], 2)
            stop.set()

        with patch.object(stop, "wait", side_effect=pause):
            self.assertEqual(self.run_loop(self.learner(), corpus, stop=stop), "paused")
        self.assertEqual(read_state(self.checkpoint)["cursor"]["offset"], 2)

    def test_changed_input_and_incompatible_settings_are_rejected(self):
        corpus = self.corpus(["ABC"])
        state = new_state(corpus, "fake", 2)
        for model, context in (("other", 2), ("fake", 3)):
            with self.assertRaises(PrototypeError):
                validate_resume(state, corpus, model, context)
        pq.write_table(pa.table({"text": ["XYZ123"]}), self.data / "000.parquet")
        with self.assertRaises(PrototypeError):
            validate_resume(state, ParquetCorpus(self.data), "fake", 2)

    def test_missing_text_and_invalid_record_stop_with_saved_progress(self):
        corpus = self.corpus(["ABC"])
        with self.assertRaises(PrototypeError):
            list(corpus.documents({"file": 0, "group": 0, "row": 2, "offset": 1}))
        pq.write_table(pa.table({"wrong": ["ABC"]}), self.data / "001.parquet")
        with self.assertRaisesRegex(PrototypeError, "text column"):
            ParquetCorpus(self.data)

    def test_empty_shard_before_documents_does_not_block_the_corpus(self):
        with pq.ParquetWriter(self.data / "000.parquet", pa.schema([("text", pa.string())])):
            pass
        pq.write_table(pa.table({"text": ["ABC"]}), self.data / "001.parquet")
        corpus = ParquetCorpus(self.data)
        self.assertEqual(self.run_loop(self.learner(), corpus), "complete")
        self.assertEqual(read_state(self.checkpoint)["targets"], 3)

    @unittest.skipIf(os.name == "nt", "The shell trainer uses Unix advisory locks")
    def test_training_lock_blocks_duplicate_writers_and_releases(self):
        with (
            training_lock(self.root / "run"),
            self.assertRaisesRegex(PrototypeError, "already"),
            training_lock(self.root / "run"),
        ):
            self.fail("second writer acquired the lock")
        with training_lock(self.root / "run"):
            pass


if __name__ == "__main__":
    unittest.main()
