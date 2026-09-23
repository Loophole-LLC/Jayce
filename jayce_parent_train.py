# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Practice randomly chosen facts from the parent, including repeated examples."""

from __future__ import annotations

import json
import random
import re
import threading
from pathlib import Path

from jayce_config import AUTO_CAPACITY_LIMIT, BATCH_SIZE, DEFAULT_NATIVE_RUN, DEFAULT_TOPICS, PARENT_MODEL
from jayce_native import load_native_lessons
from jayce_lessons import new_state, question_key, read_state, save_lessons, teach_lesson
from jayce_tokens import DEFAULT_CAPACITY, MemoryFull, PrototypeError
from jayce_topics import load_topics
from jayce_checkpoint import set_capacity, training_lock, training_stop

LESSON_INSTRUCTIONS = (
    "You are teaching a small learner from your own knowledge. The user supplies a topic as JSON data. "
    "Generate the requested number of lessons about that topic. "
    "For each lesson, write different ways to ask about ONE fact, with ONE short answer "
    "that correctly answers every question. Use the question counts in repetitions, in order. "
    "Vary the questions' wording and sentence structure, not just capitalization or punctuation. "
    "Keep every question self-contained and asking for the same information. "
    "Pair neighboring lessons as contrasts: use similar question wording about different "
    "entities or quantities that need DIFFERENT answers. For example, France's capital is "
    "Paris, while Germany's capital is Berlin. Change both the fact and its answer; "
    "never make up a false fact just to create a contrast. An unpaired final lesson is fine. "
    "Repetition is welcome: familiar facts may be practiced again in the same or different wording. "
    "Use basic, stable facts you are confident about. Each answer must be at most 12 words. "
    "Do not ask the user anything or require a source document. "
    'Output only one JSON object per lesson with keys "questions" (a list of strings) and "answer", '
    'one object per line. Example: {"questions":["How many sides does a triangle have?",'
    '"What is the number of sides in a triangle?"],"answer":"A triangle has three sides."}'
)


def generate_topic_lessons(tokenizer, model, topic: str, count: int, *, batch=0):
    from jayce_model import chat_prompt_token_count, generate_chat

    repetitions = [random.randint(1, 5) for _ in range(count)]
    output_tokens = max(384, count * 100 + sum(repetitions) * 60)
    context_limit = model.n_ctx() if tokenizer is None else model.config.max_position_embeddings
    prompt = json.dumps({
        "task": f"Generate {count} question-and-answer lessons, one JSON object per line.",
        "topic": topic, "number_of_lessons": count, "batch": batch,
        "repetitions": repetitions,
    }, ensure_ascii=False)
    if chat_prompt_token_count(tokenizer, model, prompt, LESSON_INSTRUCTIONS) + output_tokens > context_limit:
        raise PrototypeError("The teacher's context is too small to generate a lesson batch.")
    completed = []
    result = generate_chat(
        tokenizer, model, [], prompt, output_tokens,
        on_finished=completed.append, system_message=LESSON_INSTRUCTIONS,
    )
    if completed != [True]:
        raise PrototypeError("The parent did not finish its lesson plan; that batch was not taught.")
    return _parse_qa_pairs(result, repetitions)


def _parse_qa_pairs(reply: str, repetitions: list[int]) -> list[tuple[str, str]]:
    reply = reply.strip()
    items: list[object]
    try:
        parsed = json.loads(reply)
        items = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        # The model likely wrote one JSON object per line instead of a single
        # document; pull out each {...} span rather than requiring valid JSON overall.
        items = []
        for match in re.finditer(r"\{[^{}]*\}", reply):
            try:
                items.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    pairs: list[tuple[str, str]] = []
    for item, repeats in zip(items, repetitions):
        if not isinstance(item, dict):
            continue
        questions, answer = item.get("questions"), item.get("answer")
        if (
            not isinstance(answer, str) or not answer.strip()
            or len(answer.encode("utf-8")) > 256
            or not isinstance(questions, list) or not questions
            or any(not isinstance(q, str) or not q.strip() or len(q.encode("utf-8")) > 1024 for q in questions)
        ):
            continue
        # Keep useful wordings within this lesson. Facts can still be practiced
        # again in later lessons; there is no global skip list.
        seen = set()
        # Parents sometimes supply fewer or extra variants. Keep usable work
        # within the requested budget instead of discarding an entire batch.
        for question in questions[:repeats]:
            key = question_key(question)
            if key not in seen:
                pairs.append((question.strip(), answer.strip()))
                seen.add(key)
    if not pairs:
        raise PrototypeError(
            "The parent did not return usable lessons with the requested question wordings."
        )
    return pairs


def teach_topics(responder, state, checkpoint, generate, *, lesson_limit=0, stop=None,
                 auto_grow=False, capacity_limit=AUTO_CAPACITY_LIMIT, random_topics=False,
                 topic_pool=None):
    """Teach every example until interrupted or full; repetitions are training too."""
    stop = stop or threading.Event()
    taught_at_start = state["taught"]
    state["no_progress"] = 0
    topic_pool = topic_pool or DEFAULT_TOPICS
    stalled_limit = max(3, min(10, len(topic_pool) if random_topics else len(state["topics"])))

    def choose_random_topic():
        state["topics"] = list(topic_pool)
        topic = random.choice(topic_pool)
        state["topic_index"] = state["topics"].index(topic)

    # Finish any saved plan first, even if an older run used a specific topic.
    if random_topics and state["pending"] is None:
        choose_random_topic()
    if state["pending"] is not None:
        print(f"Resuming saved examples about {state['topics'][state['topic_index']]}...", flush=True)

    def save(status):
        state["status"] = status
        save_lessons(responder, checkpoint, state)

    def next_batch():
        state["rounds"] += 1
        if random_topics:
            choose_random_topic()
        else:
            state["topic_index"] = (state["topic_index"] + 1) % len(state["topics"])
        state["pending"] = None

    save("running")
    while True:
        if stop.is_set():
            save("paused")
            return "paused"
        taught = state["taught"] - taught_at_start
        if lesson_limit and taught >= lesson_limit:
            save("complete")
            return "complete"
        if state["no_progress"] >= stalled_limit:
            save("no_progress")
            return "no_progress"
        topic = state["topics"][state["topic_index"]]
        if state["pending"] is None:
            count = min(state["per_topic"], lesson_limit - taught) if lesson_limit else state["per_topic"]
            print(f"Learning about {topic}...", flush=True)
            try:
                pairs = generate(topic, count)
                if not pairs:
                    raise PrototypeError("The parent returned no usable lessons.")
            except PrototypeError as error:
                state["no_progress"] += 1
                next_batch()
                save("running")
                print(f"{error} Trying another batch.", flush=True)
                continue
            state["pending"] = {"pairs": pairs, "offset": 0}
            save("running")  # Retain the exact plan even if teaching stops midway.
            if stop.is_set():
                save("paused")
                return "paused"
        pending = state["pending"]
        question, answer = pending["pairs"][pending["offset"]]
        while True:
            try:
                teach_lesson(responder, state, question, answer, topic=topic)
                break
            except MemoryFull:
                if not auto_grow or responder.memory.max_prototypes >= capacity_limit:
                    save("capacity")
                    return "capacity"
                responder.memory.max_prototypes = min(capacity_limit, responder.memory.max_prototypes * 2)
                save("running")
                print(f"Growing memory to {responder.memory.max_prototypes:,} slots.", flush=True)
        state["taught"] += 1
        state["checked"] += 1
        state["no_progress"] = 0
        pending["offset"] += 1
        if pending["offset"] == len(pending["pairs"]):
            next_batch()
        save("running")
        print(f"Learned: {question} → {answer.strip()}", flush=True)


def run_training(capacity=None, *, run_dir=DEFAULT_NATIVE_RUN):
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint = run_dir / "lessons.npz"
    with training_lock(run_dir), training_stop() as stop:
        state = read_state(checkpoint) or new_state(PARENT_MODEL, DEFAULT_TOPICS)
        state.update(model=PARENT_MODEL, per_topic=BATCH_SIZE)
        responder = load_native_lessons(checkpoint, capacity or DEFAULT_CAPACITY)
        set_capacity(responder.memory, capacity)
        from jayce_model import load_model

        print("Learning random facts with 1–5 question wordings each. Ctrl+C pauses. Saving automatically.")
        topics = load_topics(run_dir / "topics.json", fallback=state["topics"])
        if stop.is_set():
            print("Paused before loading the parent.")
            return
        print("Loading the parent...", flush=True)
        tokenizer, model = load_model(PARENT_MODEL)

        def generate(topic, count):
            return generate_topic_lessons(tokenizer, model, topic, count, batch=state["rounds"])

        reason = teach_topics(responder, state, checkpoint, generate, stop=stop,
                              auto_grow=capacity is None, random_topics=True, topic_pool=topics)
        print("Paused. Memory saved. Use ./jayce --no-parent to chat without the parent.")
        if reason == "capacity":
            print("Memory is full. Increase --capacity to continue.")
        elif reason == "no_progress":
            print("The parent repeatedly returned unusable examples. Rerun to continue.")
