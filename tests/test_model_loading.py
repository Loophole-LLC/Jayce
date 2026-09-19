# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Model downloads must work with or without a saved Hugging Face login."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jayce_model import (
    ContextEncoder,
    ModelUnavailable,
    _available_memory,
    _total_memory,
    load_model,
)

GIB = 2**30


def denied(status):
    http_error = OSError("Access denied")
    http_error.response = SimpleNamespace(status_code=status)
    error = OSError("Wrapped Hub error")
    error.__cause__ = http_error
    return error


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = self.enterContext(
            patch("jayce_model.AutoTokenizer.from_pretrained")
        )
        self.model = self.enterContext(
            patch("jayce_model.AutoModelForCausalLM.from_pretrained")
        )
        self.saved_token = self.enterContext(
            patch("jayce_model.huggingface_hub.get_token", return_value="test-token")
        )
        self.enterContext(patch("jayce_model._announce_download"))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_public_model_ignores_saved_login_for_tokenizer_and_weights(self):
        tokenizer, model = load_model("example/parent")
        self.assertIs(tokenizer, self.tokenizer.return_value)
        self.assertIs(model, self.model.return_value)
        self.assertIs(self.tokenizer.call_args.kwargs["token"], False)
        self.assertIs(self.model.call_args.kwargs["token"], False)
        self.saved_token.assert_not_called()
        model.eval.assert_called_once_with()

    def test_private_tokenizer_retries_with_saved_login(self):
        self.tokenizer.side_effect = [denied(401), Mock()]
        load_model("owner/private-model")
        self.assertEqual(
            [call.kwargs["token"] for call in self.tokenizer.call_args_list],
            [False, None],
        )
        self.assertIsNone(self.model.call_args.kwargs["token"])

    def test_gated_weights_retry_with_saved_login(self):
        self.model.side_effect = [denied(403), Mock()]
        load_model("owner/gated-model")
        self.assertEqual(
            [call.kwargs["token"] for call in self.model.call_args_list],
            [False, None],
        )

    def test_denied_access_without_login_explains_how_to_sign_in(self):
        self.saved_token.return_value = None
        self.tokenizer.side_effect = denied(401)
        with self.assertRaisesRegex(ModelUnavailable, "hf auth login"):
            load_model("owner/private-model")
        self.assertEqual(self.tokenizer.call_count, 1)

    def test_rejected_saved_login_is_not_retried_again(self):
        self.tokenizer.side_effect = denied(401)
        with self.assertRaisesRegex(ModelUnavailable, "denied access"):
            load_model("owner/private-model")
        self.assertEqual(self.tokenizer.call_count, 2)

    def test_other_errors_do_not_trigger_login_retry(self):
        self.tokenizer.side_effect = OSError("Connection unavailable")
        with self.assertRaisesRegex(ModelUnavailable, "Connection unavailable"):
            load_model("example/parent")
        self.assertEqual(self.tokenizer.call_count, 1)
        self.saved_token.assert_not_called()


class GGUFLoadingTests(unittest.TestCase):
    source = "hf://example/parent/model-Q4_K_M.gguf"

    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(folder) / "model-Q4_K_M.gguf"
        self.path.write_bytes(b"test weights")
        self.llama = Mock()
        self.llama.return_value.model_path = str(self.path)
        self.llama.return_value.n_vocab.return_value = 256
        self.llama.return_value.n_ctx.return_value = 4096
        self.enterContext(patch.dict("sys.modules", {
            "llama_cpp": SimpleNamespace(Llama=self.llama, LLAMA_POOLING_TYPE_NONE=0)
        }))
        self.enterContext(patch("jayce_model._filter_expected_embedding_warning"))
        # Unknown memory never blocks a load; the memory tests set their own values.
        self.enterContext(patch("jayce_model._available_memory", return_value=None))
        self.enterContext(patch("jayce_model._total_memory", return_value=None))
        self.cached = self.enterContext(patch(
            "jayce_model.huggingface_hub.try_to_load_from_cache", return_value=None
        ))
        self.download = self.enterContext(patch(
            "jayce_model.huggingface_hub.hf_hub_download", return_value=str(self.path)
        ))
        self.saved_token = self.enterContext(patch(
            "jayce_model.huggingface_hub.get_token", return_value="test-token"
        ))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_remote_gguf_downloads_anonymously_and_loads_embeddings(self):
        tokenizer, model = load_model(self.source)
        self.assertIsNone(tokenizer)
        self.assertIs(model, self.llama.return_value)
        self.download.assert_called_once_with(
            "example/parent", "model-Q4_K_M.gguf", token=False
        )
        self.saved_token.assert_not_called()
        self.assertEqual(self.llama.call_args.kwargs["model_path"], str(self.path))
        self.assertTrue(self.llama.call_args.kwargs["embedding"])
        self.assertEqual(self.llama.call_args.kwargs["pooling_type"], 0)

    def test_cached_gguf_needs_no_network_or_login(self):
        self.cached.return_value = str(self.path)
        with patch("jayce_model.huggingface_hub.constants.HF_HUB_OFFLINE", True):
            load_model(self.source)
        self.download.assert_not_called()
        self.saved_token.assert_not_called()

    def test_local_gguf_needs_no_hub_lookup(self):
        load_model(str(self.path))
        self.cached.assert_not_called()
        self.download.assert_not_called()

    def test_private_gguf_retries_saved_login_only_after_access_denied(self):
        self.download.side_effect = [denied(401), str(self.path)]
        load_model(self.source)
        self.assertEqual([c.kwargs["token"] for c in self.download.call_args_list], [False, None])

    def test_gguf_download_failure_is_actionable(self):
        self.download.side_effect = OSError("Connection unavailable")
        with self.assertRaisesRegex(ModelUnavailable, "internet connection"):
            load_model(self.source)
        self.saved_token.assert_not_called()
        self.llama.assert_not_called()

    def test_invalid_remote_reference_fails_before_download(self):
        for source in ("hf://example/model.gguf", "hf://example/parent/config.json"):
            with self.subTest(source=source), self.assertRaises(ModelUnavailable):
                load_model(source)
        self.download.assert_not_called()

    def test_remote_memory_identity_includes_actual_weights(self):
        tokenizer, model = load_model(self.source)
        original = ContextEncoder(tokenizer, model, self.source).identity
        self.assertEqual(original, ContextEncoder(tokenizer, model, self.source).identity)
        self.path.write_bytes(b"different model weights")
        self.assertNotEqual(original, ContextEncoder(tokenizer, model, self.source).identity)

    def test_too_little_free_memory_stops_before_loading(self):
        with patch("jayce_model._available_memory", return_value=1 * GIB):
            with self.assertRaisesRegex(ModelUnavailable, "free memory.*only 1.0 GiB"):
                load_model(str(self.path))
        self.llama.assert_not_called()

    def test_enough_free_memory_loads(self):
        with patch("jayce_model._available_memory", return_value=64 * GIB):
            load_model(str(self.path))
        self.llama.assert_called_once()

    def test_memory_check_can_be_skipped(self):
        with (
            patch("jayce_model._available_memory", return_value=1 * GIB),
            patch.dict("os.environ", {"JAYCE_SKIP_MEMORY_CHECK": "1"}),
        ):
            load_model(str(self.path))
        self.llama.assert_called_once()

    def test_memory_needed_grows_with_the_model_file(self):
        self.path.write_bytes(b"x" * 1024)          # tiny file: only the fixed overhead
        with patch("jayce_model._available_memory", return_value=int(1.4 * GIB)):
            with self.assertRaisesRegex(ModelUnavailable, "about 1.5 GiB"):
                load_model(str(self.path))


    def test_computer_too_small_for_the_model_suggests_a_smaller_one(self):
        with patch("jayce_model._total_memory", return_value=1 * GIB):
            with self.assertRaisesRegex(
                ModelUnavailable, "computer has only 1.0 GiB.*smaller model"
            ):
                load_model(str(self.path))
        self.llama.assert_not_called()

    def test_the_system_needs_room_on_top_of_the_model(self):
        # The tiny test file needs 1.5 GiB, plus 1 GiB left for the system.
        with patch("jayce_model._total_memory", return_value=int(2.4 * GIB)):
            with self.assertRaisesRegex(
                ModelUnavailable, "2.5 GiB.*including room for the system"
            ):
                load_model(str(self.path))
        self.llama.assert_not_called()
        with patch("jayce_model._total_memory", return_value=int(2.6 * GIB)):
            load_model(str(self.path))
        self.llama.assert_called_once()

    def test_total_memory_protects_where_available_memory_is_unknown(self):
        # macOS and Windows report no reliable "available" figure; total still counts.
        with (
            patch("jayce_model._total_memory", return_value=1 * GIB),
            patch("jayce_model._available_memory", return_value=None),
        ):
            with self.assertRaises(ModelUnavailable):
                load_model(str(self.path))
        self.llama.assert_not_called()

    def test_enough_total_memory_loads_without_an_available_figure(self):
        with patch("jayce_model._total_memory", return_value=8 * GIB):
            load_model(str(self.path))
        self.llama.assert_called_once()

    def test_skipping_the_memory_check_also_skips_the_total_memory_check(self):
        with (
            patch("jayce_model._total_memory", return_value=1 * GIB),
            patch.dict("os.environ", {"JAYCE_SKIP_MEMORY_CHECK": "1"}),
        ):
            load_model(str(self.path))
        self.llama.assert_called_once()


class TotalMemoryTests(unittest.TestCase):
    def test_reports_the_installed_memory(self):
        self.assertGreater(_total_memory(), 0)

    def test_unknown_when_psutil_is_missing(self):
        with patch.dict("sys.modules", {"psutil": None}):
            self.assertIsNone(_total_memory())


class AvailableMemoryTests(unittest.TestCase):
    def read(self, text):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(folder) / "meminfo"
        path.write_text(text)
        return _available_memory(str(path))

    def test_reads_mem_available_in_bytes(self):
        text = (
            "MemTotal:       8000000 kB\n"
            "MemFree:        100 kB\n"
            "MemAvailable:   4096000 kB\n"
        )
        self.assertEqual(self.read(text), 4096000 * 1024)

    def test_missing_or_malformed_information_means_unknown(self):
        self.assertIsNone(_available_memory("/no/such/meminfo"))
        self.assertIsNone(self.read("MemTotal: 8000000 kB\n"))
        self.assertIsNone(self.read("MemAvailable: lots\n"))


if __name__ == "__main__":
    unittest.main()
