# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from jayce_native import LessonEncoder, load_native_lessons
from jayce_parent_train import DEFAULT_TOPICS, generate_topic_lessons, run_training, new_state, read_state, teach_topics
from jayce import ChatSession
from jayce_tokens import PrototypeError, PrototypeResponder, Reply, answer_context


class NativeLessonTests(unittest.TestCase):
    def test_long_answers_retain_the_question_and_recall_without_a_parent(self):
        responder = PrototypeResponder(LessonEncoder())
        # The prefixes share far more than 128 bytes, but must end differently.
        common = "A shared description of the lantern and its location. " * 3
        for question, color in (("First lantern?", "amber"), ("Second lantern?", "blue")):
            responder.teach(answer_context(question), common + color)
        for question, color in (("First lantern?", "amber"), ("Second lantern?", "blue")):
            self.assertEqual(responder.reply(answer_context(question), max_tokens=256).text, common + color)
        self.assertIsNone(responder.reply(answer_context("Third lantern?")).text)


class ParentTrainingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.checkpoint = self.root / "lessons.npz"
        self.state = new_state("teacher", ["geometry"], 2)

    def run_loop(self, responder, generate, **kwargs):
        state = read_state(self.checkpoint) or self.state
        kwargs.setdefault("lesson_limit", 1)
        with redirect_stdout(io.StringIO()):
            return teach_topics(responder, state, self.checkpoint, generate, **kwargs)

    def test_known_and_unknown_answers_are_both_taught_and_verified(self):
        responder = load_native_lessons(self.checkpoint)
        responder.teach(answer_context("Known?"), "A")
        generate = Mock(return_value=[("Known?", "B"), ("New?", "C")])
        with patch.object(responder, "teach", wraps=responder.teach) as teach:
            self.assertEqual(self.run_loop(responder, generate, lesson_limit=2), "complete")
        self.assertEqual(teach.call_count, 2)
        teach.assert_any_call(answer_context("Known?"), "B", teacher=True, max_tokens=2048, correct=True)
        teach.assert_any_call(answer_context("New?"), "C", teacher=True, max_tokens=2048, correct=True)
        state = read_state(self.checkpoint)
        self.assertEqual((state["checked"], state["known"], state["taught"]), (2, 0, 2))
        restored = load_native_lessons(self.checkpoint)
        self.assertEqual(restored.reply(answer_context("Known?")).text, "B")
        self.assertEqual(restored.reply(answer_context("New?")).text, "C")
        self.assertEqual(len(state["lessons"]), 2)

    def test_capacity_resume_keeps_the_exact_plan_and_does_not_repeat_a_lesson(self):
        responder = load_native_lessons(self.checkpoint, capacity=4)
        generate = Mock(return_value=[("Q1", "A"), ("Q2", "BC")])
        self.assertEqual(self.run_loop(responder, generate, lesson_limit=2), "capacity")
        self.assertEqual(read_state(self.checkpoint)["pending"]["offset"], 1)
        restored = load_native_lessons(self.checkpoint)
        restored.memory.max_prototypes = 6
        self.assertEqual(self.run_loop(restored, generate), "complete")
        generate.assert_called_once_with("geometry", 2)
        np.testing.assert_array_equal(restored.memory.counts, np.ones(5))
        self.assertEqual(read_state(self.checkpoint)["taught"], 2)

    def test_pause_and_restart_never_need_an_interactive_prompt(self):
        generate = Mock(return_value=[("Q1", "A"), ("Q2", "B")])
        with patch("builtins.input", side_effect=AssertionError("no interactive prompts")):
            self.assertEqual(self.run_loop(load_native_lessons(self.checkpoint), generate, lesson_limit=1), "complete")
            self.assertEqual(self.run_loop(load_native_lessons(self.checkpoint), generate), "complete")
        generate.assert_called_once()
        self.assertEqual(read_state(self.checkpoint)["checked"], 2)

    def test_interrupt_during_generation_preserves_plan_for_resume(self):
        stop = threading.Event()
        def generate(*args):
            stop.set()
            return [("Q", "A")]
        self.assertEqual(self.run_loop(load_native_lessons(self.checkpoint), generate, stop=stop), "paused")
        self.assertEqual(read_state(self.checkpoint)["pending"]["offset"], 0)
        no_regeneration = Mock(side_effect=AssertionError("plan already saved"))
        self.assertEqual(self.run_loop(load_native_lessons(self.checkpoint), no_regeneration), "complete")

    def test_failed_recall_does_not_commit_an_unverified_lesson(self):
        responder = load_native_lessons(self.checkpoint)
        with patch.object(responder, "reply", return_value=Reply(None, "no match")):
            with self.assertRaisesRegex(PrototypeError, "could not recall"):
                self.run_loop(responder, lambda *args: [("Q", "A")])
        self.assertEqual(load_native_lessons(self.checkpoint).memory.examples, 0)
        state = read_state(self.checkpoint)
        self.assertEqual((state["pending"]["offset"], state["taught"]), (0, 0))

    def test_failed_save_retains_prior_memory_and_resume_position(self):
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("Q1", "A"), ("Q2", "B")], lesson_limit=1)
        before = self.checkpoint.read_bytes()
        with patch("jayce_tokens.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [])
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_plain_native_cli_can_recall_without_corpus_or_model(self):
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("How many sides does a triangle have?", "Three.")])
        before = self.checkpoint.read_bytes()
        script = """
import sys
for name in ('jayce_model', 'torch', 'transformers', 'llama_cpp', 'huggingface_hub'):
    sys.modules[name] = None
from jayce import ChatSession
ChatSession(parent=False, run_dir=sys.argv[1]).answer('How many sides does a triangle have?')
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.root)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Jayce> Three.", result.stdout)
        self.assertNotIn("/lessons", result.stdout)
        self.assertEqual(self.checkpoint.read_bytes(), before)


    def test_invalid_pending_position_is_rejected(self):
        self.state["pending"] = {"pairs": [["Q", "A"]], "offset": 1}
        responder = load_native_lessons(self.checkpoint)
        responder.memory.save(str(self.checkpoint), responder.encoder.identity, training_state=self.state)
        with self.assertRaisesRegex(PrototypeError, "invalid pending"):
            read_state(self.checkpoint)

    def test_bounded_loop_includes_repeats_and_can_change_limit_on_resume(self):
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("Q1", "A"), ("Q1", "A"), ("Q2", "B")])
        generate = Mock(return_value=[("Q3", "C"), ("Q4", "D")])
        self.run_loop(load_native_lessons(self.checkpoint), generate)
        state = read_state(self.checkpoint)
        self.assertEqual((state["checked"], state["known"], state["taught"]), (2, 0, 2))
        generate.assert_not_called()
        self.run_loop(load_native_lessons(self.checkpoint), generate, lesson_limit=2)
        self.assertEqual(read_state(self.checkpoint)["taught"], 4)

    def test_plain_cli_keeps_teaching_until_interrupted_without_topics_or_prompts(self):
        stop = threading.Event()
        calls = []
        random_topics = ["weather", "basic chemistry", "plants and animals", "world geography"]
        def generate(tokenizer, model, topic, count, **kwargs):
            calls.append(topic)
            if len(calls) == 4:
                stop.set()
            return [(f"Question {len(calls)}?", "A")]
        with (
            patch("jayce_model.load_model", return_value=(None, object())),
            patch("jayce_parent_train.load_topics", return_value=DEFAULT_TOPICS),
            patch("jayce_parent_train.generate_topic_lessons", side_effect=generate),
            patch("jayce_parent_train.threading.Event", return_value=stop),
            patch("jayce_parent_train.random.choice", side_effect=random_topics) as choose,
            patch("builtins.input", side_effect=AssertionError("no interactive prompts")),
            redirect_stdout(io.StringIO()),
        ):
            run_training(run_dir=self.root)
        state = read_state(self.checkpoint)
        self.assertEqual((state["checked"], state["taught"], state["status"]), (3, 3, "paused"))
        self.assertEqual(calls, random_topics)
        for call in choose.call_args_list:
            self.assertEqual(call.args[0], DEFAULT_TOPICS)

    def test_random_teaching_finishes_saved_plan_before_choosing_a_new_area(self):
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("Q1", "A"), ("Q2", "B")])
        with patch("jayce_parent_train.random.choice", return_value="weather"):
            self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("Q3", "C")],
                          lesson_limit=2, random_topics=True)
        state = read_state(self.checkpoint)
        self.assertEqual([(item["question"], item["topic"]) for item in state["lessons"]],
                         [("Q1", "geometry"), ("Q2", "geometry"), ("Q3", "weather")])
        self.assertEqual(state["topics"], DEFAULT_TOPICS)

    def test_downloaded_catalog_supplies_topics_after_old_pending_plan_finishes(self):
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("Q1", "A"), ("Q2", "B")])
        catalog = ["Calculus", "French language"]
        generate = Mock(return_value=[("Derivative of x?", "1")])
        with patch("jayce_parent_train.random.choice", return_value="Calculus") as choose:
            self.run_loop(load_native_lessons(self.checkpoint), generate, lesson_limit=2,
                          random_topics=True, topic_pool=catalog)
        generate.assert_called_once_with("Calculus", 1)
        for picked in choose.call_args_list:
            self.assertEqual(picked.args[0], catalog)
        state = read_state(self.checkpoint)
        self.assertEqual(state["topics"], catalog)
        self.assertEqual([(item["question"], item["topic"]) for item in state["lessons"]],
                         [("Q1", "geometry"), ("Q2", "geometry"), ("Derivative of x?", "Calculus")])

    def test_large_catalog_does_not_require_thousands_of_failures_before_stopping(self):
        catalog = [f"Topic {i}" for i in range(10000)]
        generate = Mock(return_value=[])
        result = self.run_loop(load_native_lessons(self.checkpoint), generate,
                               random_topics=True, topic_pool=catalog)
        self.assertEqual(result, "no_progress")
        self.assertEqual(generate.call_count, 10)

    def test_200_lessons_keeps_requesting_when_teacher_returns_only_one_per_batch(self):
        self.state["topics"] = ["geography"]
        calls = []
        def generate(topic, count):
            calls.append(count)
            return [(f"Fact {len(calls)}?", "A")]
        result = self.run_loop(load_native_lessons(self.checkpoint), generate, lesson_limit=200)
        self.assertEqual(result, "complete")
        self.assertEqual(read_state(self.checkpoint)["taught"], 200)
        self.assertEqual(len(calls), 200)
        self.assertEqual(load_native_lessons(self.checkpoint).reply(answer_context("Fact 200?")).text, "A")

    def test_repeats_keep_training_in_full_memory_without_growing_the_question_index(self):
        self.run_loop(load_native_lessons(self.checkpoint, capacity=2), lambda *args: [("Known?", "A")])
        generate = Mock(return_value=[("Known?", "A")])
        self.assertEqual(self.run_loop(load_native_lessons(self.checkpoint), generate, lesson_limit=200), "complete")
        self.assertEqual(generate.call_count, 200)
        state = read_state(self.checkpoint)
        self.assertEqual((state["taught"], state["known"], state["no_progress"]), (201, 0, 0))
        self.assertEqual(len(state["lessons"]), 1)
        restored = load_native_lessons(self.checkpoint)
        self.assertEqual(len(restored.memory), 2)
        self.assertEqual(restored.memory.teacher_examples, 201)
        np.testing.assert_array_equal(restored.memory.counts, [201, 201])
        self.assertEqual(restored.reply(answer_context("Known?")).text, "A")

    def test_changed_wording_is_learned_and_normalized_questions_update_the_same_memory(self):
        from jayce_lessons import SharedLessons

        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("What is a triangle?", "Three sides.")])
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [
            ("What's a triangle?", "A shape with three sides."),
            ("Describe a triangle.", "It has three sides."),
        ], lesson_limit=2)
        restored = SharedLessons(self.checkpoint)
        self.assertEqual(len(restored.lessons), 2)
        for question in ("What is a triangle?", "What's a triangle?"):
            self.assertEqual(restored.reply(answer_context(question)).text, "A shape with three sides.")
        self.assertEqual(restored.reply(answer_context("Describe a triangle.")).text, "It has three sides.")

    def test_question_wordings_resume_and_all_recall_without_parent(self):
        questions = ["How many sides does a triangle have?", "What is the number of sides in a triangle?",
                     "How many edges make up a triangle?"]
        answer = "A triangle has three sides."
        def generate(*args, on_finished, **kwargs):
            on_finished(True)
            return json.dumps({"questions": questions, "answer": answer})
        model = Mock()
        model.n_ctx.return_value = 4096
        with (
            patch("jayce_parent_train.random.randint", return_value=3),
            patch("jayce_model.chat_prompt_token_count", return_value=100),
            patch("jayce_model.generate_chat", side_effect=generate) as parent,
        ):
            self.run_loop(load_native_lessons(self.checkpoint),
                          lambda topic, count: generate_topic_lessons(None, model, topic, count))
            pending = read_state(self.checkpoint)["pending"]
            self.assertEqual(pending, {"pairs": [[q, answer] for q in questions], "offset": 1})
            self.assertIsNone(load_native_lessons(self.checkpoint).reply(answer_context(questions[1])).text)
            self.run_loop(load_native_lessons(self.checkpoint),
                          Mock(side_effect=AssertionError("resume must reuse saved versions")), lesson_limit=2)
        parent.assert_called_once()
        state = read_state(self.checkpoint)
        self.assertEqual(state["taught"], 3)
        self.assertEqual(len(state["lessons"]), 3)
        self.assertIsNone(state["pending"])
        self.assertEqual(load_native_lessons(self.checkpoint).memory.teacher_examples, 3)
        script = """
import json, sys
for name in ('jayce_model', 'torch', 'transformers', 'llama_cpp', 'huggingface_hub'):
    sys.modules[name] = None
from jayce import ChatSession
chat = ChatSession(parent=False, run_dir=sys.argv[1])
for question in json.loads(sys.argv[2]):
    chat.answer(question)
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.root), json.dumps(questions)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Jayce> " + answer), 3)

    def test_only_unusable_batches_trigger_the_no_progress_stop(self):
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("Known?", "A")])
        generate = Mock(return_value=[])
        self.assertEqual(self.run_loop(load_native_lessons(self.checkpoint), generate), "no_progress")
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(read_state(self.checkpoint)["taught"], 1)

    def test_automatic_capacity_growth_preserves_answers_and_has_a_bound(self):
        responder = load_native_lessons(self.checkpoint, capacity=2)
        self.assertEqual(self.run_loop(responder, lambda *args: [("Q", "AB")], auto_grow=True, capacity_limit=4), "complete")
        self.assertEqual(responder.memory.max_prototypes, 4)
        self.assertEqual(self.run_loop(responder, lambda *args: [("Q2", "CD")], auto_grow=True, capacity_limit=4), "capacity")
        self.assertEqual(load_native_lessons(self.checkpoint).reply(answer_context("Q")).text, "AB")

    def test_old_complete_checkpoint_resumes_without_losing_answers(self):
        responder = load_native_lessons(self.checkpoint)
        responder.teach(answer_context("Old?"), "A")
        self.state.update(format="jayce-parent-teaching-v1", topic_index=1, status="complete", checked=1, taught=1,
                          lessons=[{"topic":"geometry", "question":"Old?", "answer":"A"}])
        self.state.pop("rounds")
        self.state.pop("no_progress")
        responder.memory.save(str(self.checkpoint), responder.encoder.identity, training_state=self.state)
        before = self.checkpoint.read_bytes()
        self.assertEqual(read_state(self.checkpoint)["topic_index"], 0)
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertEqual(self.run_loop(responder, lambda *args: [("New?", "B")]), "complete")
        restored = load_native_lessons(self.checkpoint)
        self.assertEqual(restored.reply(answer_context("Old?")).text, "A")
        self.assertEqual(restored.reply(answer_context("New?")).text, "B")

    def test_default_cli_recalls_an_answer_longer_than_128_bytes(self):
        answer = "A useful answer can be longer than one hundred twenty-eight bytes. " * 3
        self.run_loop(load_native_lessons(self.checkpoint), lambda *args: [("Long answer?", answer.strip())])
        with (
            redirect_stdout(io.StringIO()) as output,
        ):
            ChatSession(parent=False, run_dir=self.root).answer("Long answer?")
        self.assertIn(answer.strip(), output.getvalue())


class LessonGenerationTests(unittest.TestCase):
    def test_generation_uses_topic_data_without_documents_and_rejects_truncation(self):
        for finished in (True, False):
            def generate(*args, on_finished, **kwargs):
                on_finished(finished)
                return '{"questions":["How many sides does a triangle have?"],"answer":"Three."}'
            with (
                self.subTest(finished=finished),
                patch("jayce_parent_train.random.randint", return_value=1),
                patch("jayce_model.chat_prompt_token_count", return_value=100),
                patch("jayce_model.generate_chat", side_effect=generate) as parent,
            ):
                model = Mock()
                model.n_ctx.return_value = 4096
                if finished:
                    pairs = generate_topic_lessons(None, model, "geometry", 1)
                    self.assertEqual(pairs, [("How many sides does a triangle have?", "Three.")])
                else:
                    with self.assertRaisesRegex(PrototypeError, "did not finish"):
                        generate_topic_lessons(None, model, "geometry", 1)
            data = json.loads(parent.call_args.args[3])
            self.assertEqual((data["topic"], data["number_of_lessons"]), ("geometry", 1))

    def test_parent_is_allowed_to_repeat_examples_without_a_known_question_list(self):
        def generate(*args, on_finished, **kwargs):
            on_finished(True)
            return '{"questions":["Q"],"answer":"A"}\n{"questions":["Q"],"answer":"A"}'
        model = Mock()
        model.n_ctx.return_value = 4096
        with (
            patch("jayce_parent_train.random.randint", return_value=1),
            patch("jayce_model.chat_prompt_token_count", return_value=100),
            patch("jayce_model.generate_chat", side_effect=generate) as parent,
        ):
            pairs = generate_topic_lessons(None, model, "geography", 2, batch=9)
        self.assertEqual(pairs, [("Q", "A"), ("Q", "A")])
        prompt = json.loads(parent.call_args.args[3])
        self.assertEqual(prompt["batch"], 9)
        self.assertEqual(set(prompt), {"task", "topic", "number_of_lessons", "batch", "repetitions"})
        self.assertIn("Repetition is welcome", parent.call_args.kwargs["system_message"])
        self.assertIn("Pair neighboring lessons as contrasts", parent.call_args.kwargs["system_message"])

    def test_random_question_counts_are_generated_in_one_parent_call(self):
        lessons = [
            {"questions": ["How many sides does a triangle have?"], "answer": "Three."},
            {"questions": [
                "What is the capital of France?", "Which city is France's capital?",
                "Name the capital city of France.", "France has which city as its capital?",
                "Tell me the name of the French capital.",
            ], "answer": "Paris."},
        ]
        def generate(*args, on_finished, **kwargs):
            on_finished(True)
            return json.dumps(lessons)
        model = Mock()
        model.n_ctx.return_value = 4096
        with (
            patch("jayce_parent_train.random.randint", side_effect=[1, 5]) as choose,
            patch("jayce_model.chat_prompt_token_count", return_value=100),
            patch("jayce_model.generate_chat", side_effect=generate) as parent,
        ):
            pairs = generate_topic_lessons(None, model, "general knowledge", 2)
        self.assertEqual(pairs, [(q, item["answer"]) for item in lessons for q in item["questions"]])
        self.assertEqual(choose.call_count, 2)
        for call in choose.call_args_list:
            self.assertEqual(call.args, (1, 5))
        parent.assert_called_once()
        self.assertEqual(json.loads(parent.call_args.args[3])["repetitions"], [1, 5])

    def test_duplicate_wordings_within_a_lesson_are_collapsed(self):
        def generate(*args, on_finished, **kwargs):
            on_finished(True)
            return json.dumps({"questions": ["What is a triangle?", "What's a triangle?",
                                              "Describe a triangle."], "answer": "A three-sided shape."})
        model = Mock()
        model.n_ctx.return_value = 4096
        with (
            patch("jayce_parent_train.random.randint", return_value=3),
            patch("jayce_model.chat_prompt_token_count", return_value=100),
            patch("jayce_model.generate_chat", side_effect=generate),
        ):
            pairs = generate_topic_lessons(None, model, "geometry", 1)
        self.assertEqual(pairs, [("What is a triangle?", "A three-sided shape."),
                                 ("Describe a triangle.", "A three-sided shape.")])

    def test_malformed_or_oversized_wordings_are_not_taught(self):
        for questions, answer in (
            ([], "A"), (["Q", ""], "A"),
            (["Q", None], "A"), (["Q", "x" * 1025], "A"), ("Q1", "A"),
            (["Q1", "Q2"], None), (["Q1", "Q2"], ""), (["Q1", "Q2"], "x" * 257),
        ):
            def generate(*args, on_finished, **kwargs):
                on_finished(True)
                return json.dumps({"questions": questions, "answer": answer})
            model = Mock()
            model.n_ctx.return_value = 4096
            with (
                self.subTest(questions=questions, answer=answer),
                patch("jayce_parent_train.random.randint", return_value=2),
                patch("jayce_model.chat_prompt_token_count", return_value=100),
                patch("jayce_model.generate_chat", side_effect=generate),
            ):
                with self.assertRaisesRegex(PrototypeError, "requested question wordings"):
                    generate_topic_lessons(None, model, "geography", 1)

    def test_usable_short_or_extra_question_lists_are_kept_within_budget(self):
        from jayce_parent_train import _parse_qa_pairs

        for questions in (["Q1"], ["Q1", "Q2", "Q3", "Q4"]):
            with self.subTest(questions=questions):
                pairs = _parse_qa_pairs(json.dumps({"questions": questions, "answer": "A"}), [2])
                self.assertEqual(pairs, [(q, "A") for q in questions[:2]])

    def test_teacher_context_budget_is_checked_before_generation(self):
        model = Mock()
        model.n_ctx.return_value = 512
        with patch("jayce_model.chat_prompt_token_count", return_value=200), patch("jayce_model.generate_chat") as parent:
            with self.assertRaisesRegex(PrototypeError, "context is too small"):
                generate_topic_lessons(None, model, "geography", 1)
        parent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
