# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Chat with shared native memory, optionally learning from a parent model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from jayce_config import DEFAULT_NATIVE_RUN, PARENT_MODEL
from jayce_lessons import MAX_ANSWER_BYTES, SharedLessons
from jayce_native import load_native_memory
from jayce_tokens import PrototypeError, PrototypeResponder, Reply, answer_context

LICENSE_NOTICE = "Copyright (C) 2026 Loophole, LLC. AGPL-3.0-only; no warranty. See LICENSE.md."


def _style(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None:
        return text
    return f"\033[{code}m{text}\033[0m"


def try_prototype_reply(
    responder: PrototypeResponder | None, context: str, limit: int
) -> Reply:
    if responder is None:
        return Reply(None, "prototype memory unavailable")
    try:
        return responder.reply(context, max_tokens=limit)
    except (ValueError, RuntimeError) as error:
        return Reply(None, f"could not read context vectors: {error}")


def show_prototype_reply(reply: Reply, stage: str = "") -> None:
    label = f"Jayce ({stage})" if stage else "Jayce"
    text = reply.text if reply.text is not None else "Jayce don't know"
    print(_style(f"{label}> ", "1;36") + text, flush=True)


class ChatSession:
    """Both chat modes use the same recall, commands, and saved answers."""

    def __init__(self, *, parent=True, run_dir=DEFAULT_NATIVE_RUN,
                 model_name=PARENT_MODEL, max_new_tokens=128, history_turns=8):
        self.model_name = model_name if parent else None
        self.learning_enabled = parent
        self.max_new_tokens = max_new_tokens
        self.history_turns = history_turns
        self.history = []
        self.last_reply = None
        run_dir = Path(run_dir)
        checkpoint = run_dir / "prototypes.npz"
        self.native_text_learner = load_native_memory(checkpoint) if checkpoint.exists() else None
        try:
            self.responder = SharedLessons(run_dir / "lessons.npz")
        except (PrototypeError, OSError) as error:
            if not parent:
                raise
            self.responder = None
            print(f"Jayce's answer memory is unavailable: {error}. The parent can still answer.")
        self.tokenizer = self.model = None
        if parent:
            from jayce_model import load_model

            print("Loading the parent...", flush=True)
            self.tokenizer, self.model = load_model(model_name)

    def answer(self, question):
        context = answer_context(question)
        reply = try_prototype_reply(self.responder, context, MAX_ANSWER_BYTES)
        if reply.text is None and self.native_text_learner is not None:
            continuation = self.native_text_learner.continue_text(question, 256)
            if continuation["text"]:
                reply = Reply(continuation["text"], continuation["stop"], continuation["tokens"])
        self.last_reply = reply
        show_prototype_reply(reply)
        if not self.learning_enabled:
            return
        from jayce_model import generate_chat

        print(_style("Parent> ", "1;32"), end="", flush=True)
        completed = []
        try:
            answer = generate_chat(
                self.tokenizer, self.model, self.history, question, self.max_new_tokens,
                on_token=lambda token: print(token, end="", flush=True),
                on_finished=completed.append,
            )
        except Exception as error:  # A failed parent turn must not close the chat.
            print(f"\nGeneration failed: {error}")
            return
        print()
        if completed != [True]:
            print("  Jayce did not learn this answer: the parent stopped before a complete ending.")
        elif answer.strip() and self.responder is not None:
            agreement = reply.text is not None and " ".join(reply.text.split()) == " ".join(answer.split())
            label = "after practice" if agreement else "correction" if reply.text is not None else "after learning"
            try:
                self.responder.teach(context, answer, teacher=True, correct=True)
                if agreement:
                    print("  Parent confirms Jayce's answer → Jayce practices it again.")
                else:
                    print("  Parent teaches Jayce → saved. Jayce tries again.")
                self.last_reply = self.responder.reply(context)
                show_prototype_reply(self.last_reply, label)
            except (ValueError, OSError, RuntimeError) as error:
                print(f"  Jayce could not finish learning/saving: {error}")
        self.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        self.history = self.history[-2 * self.history_turns:]


def handle_command(session, command, argument=""):
    command, argument = command.lower(), argument.strip()
    if command == "/help":
        print("Commands:")
        if session.model_name is not None:
            print("  /learning on|off         Turn parent answers and teaching on or off")
        print("  /teach question => answer  Teach or correct an answer; saves automatically")
        print("  /clear                   Clear conversation history; keep memory")
        print("  /stats                   Show memory usage and parent status")
        print("  /quit                    Exit")
    elif command == "/learning":
        if session.model_name is None:
            print("No parent is loaded. Restart with ./jayce to enable parent teaching.")
        elif argument.lower() not in {"on", "off"}:
            print("Use /learning on or /learning off.")
        else:
            session.learning_enabled = argument.lower() == "on"
            print(f"Parent answers and teaching: {argument.lower()}.")
    elif command == "/teach":
        question, separator, answer = argument.partition("=>")
        if not separator or not question.strip() or not answer.strip():
            print("Usage: /teach question => answer")
        elif session.responder is None:
            print("Jayce's answer memory is unavailable.")
        else:
            session.responder.teach(answer_context(question.strip()), answer.strip(), correct=True)
            print("Learned and saved. Ask the question to try it.")
    elif command == "/clear":
        session.history.clear()
        print("New conversation started. Learned answers are unchanged.")
    elif command == "/stats":
        print(f"Parent: {session.model_name or 'none'}")
        print(f"Parent answers and teaching: {'on' if session.learning_enabled else 'off'}")
        if session.responder is not None:
            session.responder.refresh()
            memory = session.responder.memory
            print(f"Memory used: {len(memory)}/{memory.max_prototypes} slots")
        if session.native_text_learner is not None:
            memory = session.native_text_learner.memory
            print(f"Text memory used: {len(memory)}/{memory.max_prototypes} slots")
    else:
        print("Unknown command. Type /help.")


def chat(session):
    print(_style(LICENSE_NOTICE, "2"))
    print("Jayce — with parent." if session.model_name else "Jayce — no parent model.")
    if session.learning_enabled:
        print("Jayce tries first, then learns and saves every complete parent answer, including repeats.")
    else:
        print("Ask a taught question or enter a learned text prefix. Unfamiliar wording may not match.")
    print("Type /help for commands; /quit or Ctrl+D exits.\n")
    while True:
        try:
            value = input(_style("› ", "1;36")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return
        if not value:
            continue
        command, *rest = value.split(maxsplit=1)
        if command.lower() in {"/quit", "/exit", "/q"}:
            print("Session ended.")
            return
        try:
            if command.startswith("/"):
                handle_command(session, command, rest[0] if rest else "")
            else:
                session.answer(value)
        except (ValueError, OSError, RuntimeError) as error:
            print(f"Jayce: {error}")


def main(argv=None):
    from jayce_cli import parse_args

    args = parse_args(argv)
    try:
        if args.command == "train":
            if args.data is None:
                from jayce_parent_train import run_training
                run_training(args.capacity)
            else:
                from jayce_train import run_training
                run_training(args.data, args.capacity)
        else:
            chat(ChatSession(parent=not args.no_parent))
    except KeyboardInterrupt:
        print("\nInterrupted. The last saved memory is safe.")
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        raise SystemExit(f"Jayce: {error}")


if __name__ == "__main__":
    main()
