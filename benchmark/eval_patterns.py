# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Offline comparison of exact recall and adaptive APM; never opens user memory.

Run `.venv/bin/python benchmark/eval_patterns.py [dataset.json]`.
Use a separate, unseen dataset for independent evaluation; this fixture is also
used during development and does not establish broad generalization.
"""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jayce_native import LegacyLessonEncoder, LessonEncoder
from jayce_tokens import PrototypeResponder, answer_context


def evaluate(dataset, encoder):
    trained = {question for question, _ in dataset["train"]}
    if any(case["question"] in trained for case in dataset["test"]):
        raise ValueError("Test questions must be held out of training.")
    responder = PrototypeResponder(encoder)
    start = time.perf_counter()
    for question, answer in dataset["train"]:
        responder.teach(answer_context(question), answer, correct=True, max_tokens=2048)
    elapsed = time.perf_counter() - start
    cases = [{"group": "replay", "question": q, "answer": a} for q, a in dict(dataset["train"]).items()]
    cases += dataset["test"]
    groups, outcomes = {}, []
    for case in cases:
        actual = responder.reply(answer_context(case["question"]), max_tokens=2048).text
        result = ("correct" if actual is not None and actual == case["answer"] else
                  "rejected" if actual is None and case["answer"] is None else
                  "abstained" if actual is None else "wrong")
        group = groups.setdefault(case["group"], dict(correct=0, wrong=0, abstained=0, rejected=0))
        group[result] += 1
        outcomes.append({**case, "actual": actual, "result": result})
    return {
        "prototypes": len(responder.memory),
        "array_bytes": sum(array.nbytes for array in responder.memory.checkpoint_arrays().values()),
        "training_seconds": elapsed, "groups": groups, "outcomes": outcomes,
    }


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("patterns.json")
    dataset = json.loads(source.read_text())
    for name, encoder in (("Exact replay baseline", LegacyLessonEncoder()), ("Adaptive APM", LessonEncoder())):
        result = evaluate(dataset, encoder)
        print(f"\n{name}: {result['prototypes']} prototypes, {result['array_bytes']:,} array bytes, "
              f"{result['training_seconds']:.3f}s teaching")
        print(f"{'group':14} {'correct':>8} {'wrong':>8} {'abstain':>8} {'reject':>8}")
        for group, counts in result["groups"].items():
            print(f"{group:14} {counts['correct']:8} {counts['wrong']:8} {counts['abstained']:8} {counts['rejected']:8}")
        for case in result["outcomes"]:
            if case["result"] == "wrong":
                print(f"Wrong: {case['question']} -> {case['actual']!r}; expected {case['answer']!r}")
    print("\n'Abstain' means an expected answer was withheld; 'reject' means an unfamiliar question was withheld as intended.")
    print("This small development fixture tests wording transfer, not general language or mathematical reasoning.")


if __name__ == "__main__":
    main()
