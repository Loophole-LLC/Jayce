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

from jayce import ChatSession, chat, handle_command
from jayce_lessons import SharedLessons
from jayce_tokens import PrototypeError, answer_context


class ChatCommandTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.checkpoint = self.root / "lessons.npz"
        self.session = ChatSession(parent=False, run_dir=self.root)

    def command(self, name, argument=""):
        with redirect_stdout(io.StringIO()) as output:
            handle_command(self.session, name, argument)
        return output.getvalue()

    def test_clear_preserves_memory_and_stats_do_not_track_lessons(self):
        self.command("/teach", "Favorite color? => Blue")
        self.command("/teach", "Favorite color? => Amber")
        self.command("/teach", "Triangle sides? => Three")
        before = self.checkpoint.read_bytes()
        self.session.history = [{"role": "user", "content": "Hello"}]
        self.command("/clear")
        self.assertEqual(self.session.history, [])
        self.assertIn("Memory used:", self.command("/stats"))
        self.assertNotIn("Saved lessons", self.command("/stats"))
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertEqual(SharedLessons(self.checkpoint).reply(answer_context("Favorite color?")).text, "Amber")

    def test_invalid_teach_and_removed_commands_do_not_write(self):
        self.assertIn("Usage: /teach", self.command("/teach", "missing an answer"))
        for command in ("/learn", "/learn-url", "/save", "/forget-chat", "/lessons", "/continue-text"):
            self.assertIn("Unknown command", self.command(command))
        self.assertFalse(self.checkpoint.exists())
        help_text = self.command("/help")
        for removed in ("advanced", "/learn ", "/save", "/forget-chat", "/lessons"):
            self.assertNotIn(removed, help_text)

    def test_native_chat_help_and_recall_work_without_parent_imports(self):
        self.command("/teach", "Favorite color? => Amber")
        before = self.checkpoint.read_bytes()
        script = """
import sys
for name in ('jayce_model', 'torch', 'transformers', 'llama_cpp', 'huggingface_hub'):
    sys.modules[name] = None
from jayce import ChatSession, chat
chat(ChatSession(parent=False, run_dir=sys.argv[1]))
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.root)],
                                input="/help\n/learning on\nFavorite color?\n/stats\n/quit\n",
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Jayce> Amber", result.stdout)
        self.assertIn("Memory used:", result.stdout)
        self.assertIn("No parent is loaded", result.stdout)
        self.assertNotIn("/lessons", result.stdout)
        self.assertNotIn("Unknown command", result.stdout)
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_learning_toggle_controls_generation_and_corrected_answers_persist(self):
        calls = []
        def generate(tokenizer, model, history, question, limit, *, on_token, on_finished):
            calls.append((question, len(history)))
            answer = ["A", "B", "C"][len(calls) - 1]
            on_token(answer)
            on_finished(True)
            return answer
        with (
            patch("jayce_model.load_model", return_value=(None, object())),
            patch("jayce_model.generate_chat", side_effect=generate),
            patch("builtins.input", side_effect=[
                "Question?", "/clear", "Question?", "/learning off", "/clear",
                "Question?", "Unknown?", "/learning on", "Question?", "/quit",
            ]),
            redirect_stdout(io.StringIO()) as output,
        ):
            chat(ChatSession(run_dir=self.root))
        self.assertEqual(calls, [("Question?", 0)] * 3)
        learned = SharedLessons(self.checkpoint)
        self.assertEqual(learned.memory.teacher_examples, 3)
        self.assertEqual(learned.reply(answer_context("Question?")).text, "C")
        text = output.getvalue()
        self.assertIn("Jayce> Jayce don't know\nParent> A", text)
        self.assertIn("Jayce (correction)> B", text)
        off = text.split("Parent answers and teaching: off.")[1].split("Parent answers and teaching: on.")[0]
        self.assertIn("Jayce> B", off)
        self.assertIn("Jayce> Jayce don't know", off)
        self.assertNotIn("Parent>", off)

    def test_failed_generation_does_not_learn_or_enter_history(self):
        with (
            patch("jayce_model.load_model", return_value=(None, object())),
            patch("jayce_model.generate_chat", side_effect=RuntimeError("generation failed")),
            redirect_stdout(io.StringIO()) as output,
        ):
            session = ChatSession(run_dir=self.root)
            session.answer("Question?")
        self.assertIn("Generation failed", output.getvalue())
        self.assertEqual(session.history, [])
        self.assertFalse(self.checkpoint.exists())

    def test_unavailable_memory_still_allows_parent_but_learning_off_does_not_generate(self):
        def generate(*args, on_token, on_finished):
            on_token("A")
            on_finished(True)
            return "A"
        with (
            patch("jayce_model.load_model", return_value=(None, object())),
            patch("jayce_model.generate_chat", side_effect=generate) as parent,
            patch("jayce.SharedLessons", side_effect=PrototypeError("unreadable memory")),
            redirect_stdout(io.StringIO()) as output,
        ):
            session = ChatSession(run_dir=self.root)
            session.answer("Q")
            handle_command(session, "/learning", "off")
            session.answer("Unknown?")
        parent.assert_called_once()
        self.assertIn("Parent> A", output.getvalue())
        self.assertIsNone(session.last_reply.text)
        self.assertFalse(self.checkpoint.exists())

    def test_failed_recall_is_not_echoed_or_saved_as_success(self):
        def generate(*args, on_token, on_finished):
            on_token("B")
            on_finished(True)
            return "B"
        self.command("/teach", "Letter? => A")
        before = self.checkpoint.read_bytes()
        with (
            patch("jayce_model.load_model", return_value=(None, object())),
            patch("jayce_model.generate_chat", side_effect=generate),
            patch("jayce_lessons.teach_lesson", side_effect=PrototypeError("could not recall")),
            redirect_stdout(io.StringIO()) as output,
        ):
            ChatSession(run_dir=self.root).answer("Letter?")
        self.assertIn("could not finish learning/saving", output.getvalue())
        self.assertNotIn("Jayce (correction)> B", output.getvalue())
        self.assertEqual(self.checkpoint.read_bytes(), before)
