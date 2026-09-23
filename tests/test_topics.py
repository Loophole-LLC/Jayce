# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jayce_topics import CATALOG_PAGE, CATALOG_URL, download_topics, load_topics


class TopicCatalogTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.cache = Path(folder) / "topics.json"
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def test_download_discovers_subject_lists_and_keeps_only_article_topics(self):
        pages = {
            CATALOG_PAGE: [
                {"ns": 4, "title": CATALOG_PAGE + "/Mathematics"},
                {"ns": 4, "title": CATALOG_PAGE + "/Languages"},
                {"ns": 4, "title": "Wikipedia:Vital articles/Level 5"},
                {"ns": 0, "title": "An unrelated navigation link"},
            ],
            CATALOG_PAGE + "/Languages": [
                {"ns": 0, "title": "French language"}, {"ns": 0, "title": "Logic"},
                {"ns": 4, "title": "Wikipedia:About"},
            ],
            CATALOG_PAGE + "/Mathematics": [
                {"ns": 0, "title": "Calculus"}, {"ns": 0, "title": "Logic"},
                {"ns": 14, "title": "Category:Mathematics"},
            ],
        }
        with patch("jayce_topics._page_links", side_effect=pages.__getitem__) as fetch:
            self.assertEqual(download_topics(), ["Calculus", "French language", "Logic"])
        self.assertEqual(fetch.call_count, 3)

    def test_download_is_cached_and_subsequent_offline_run_needs_no_network(self):
        topics = ["Calculus", "French language"]
        with patch("jayce_topics.download_topics", return_value=topics) as download:
            self.assertEqual(load_topics(self.cache, ["starter"]), topics)
        download.assert_called_once()
        saved = self.cache.read_bytes()
        self.assertEqual(json.loads(saved), {"source": CATALOG_URL, "topics": topics})
        with patch("jayce_topics.urlopen", side_effect=AssertionError("no network")):
            self.assertEqual(load_topics(self.cache, ["starter"]), topics)
        self.assertEqual(self.cache.read_bytes(), saved)
        self.assertFalse(self.cache.with_suffix(".tmp").exists())

    def test_first_download_failure_clearly_falls_back_to_saved_topics(self):
        with patch("jayce_topics.download_topics", side_effect=OSError("offline")):
            self.assertEqual(load_topics(self.cache, ["Calculus", "French language"]),
                             ["Calculus", "French language"])
        self.assertFalse(self.cache.exists())
        self.assertIn("Topic download failed", self.output.getvalue())
        self.assertIn("Using 2 saved or starter topics", self.output.getvalue())

    def test_broken_cache_is_replaced_only_after_successful_download(self):
        for bad_cache in ("not json", '{"source":"wrong","topics":["Wrong"]}',
                          json.dumps({"source": CATALOG_URL, "topics": [None]})):
            with self.subTest(cache=bad_cache):
                self.cache.write_text(bad_cache)
                with patch("jayce_topics.download_topics", side_effect=OSError("offline")):
                    self.assertEqual(load_topics(self.cache, ["starter"]), ["starter"])
                self.assertEqual(self.cache.read_text(), bad_cache)
                with patch("jayce_topics.download_topics", return_value=["Calculus"]):
                    self.assertEqual(load_topics(self.cache, ["starter"]), ["Calculus"])
                self.assertEqual(json.loads(self.cache.read_text())["topics"], ["Calculus"])

    def test_partial_or_empty_download_never_becomes_a_cached_catalog(self):
        for second_response in ([], OSError("connection lost")):
            with self.subTest(response=second_response), patch("jayce_topics._page_links", side_effect=[
                [{"ns": 4, "title": CATALOG_PAGE + "/Math"}], second_response,
            ]):
                self.assertEqual(load_topics(self.cache, ["starter"]), ["starter"])
            self.assertFalse(self.cache.exists())

    def test_failed_cache_write_still_uses_downloaded_topics_and_removes_temporary(self):
        with (
            patch("jayce_topics.download_topics", return_value=["Calculus"]),
            patch("jayce_topics.Path.replace", side_effect=OSError("disk full")),
        ):
            self.assertEqual(load_topics(self.cache, ["starter"]), ["Calculus"])
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.cache.with_suffix(".tmp").exists())
        self.assertIn("Could not cache topics", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
