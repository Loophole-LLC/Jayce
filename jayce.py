# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Command-line chat: Jayce tries, the parent answers, then Jayce learns."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from jayce_config import PARENT_MODEL
from jayce_model import (
    ContextEncoder,
    ModelUnavailable,
    generate_chat,
    load_model,
    model_label,
)
from jayce_tokens import (
    PrototypeError,
    PrototypeResponder,
    Reply,
    TokenMemory,
    answer_context,
    read_examples,
)

LICENSE_NOTICE = (
    "Copyright (C) 2026 Loophole, LLC. AGPL-3.0-only; no warranty. See LICENSE.md."
)


def default_prototype_file(model_name: str) -> str:
    """Keep each parent's vectors separate; leave legacy memory files untouched."""
    source = str(Path(model_name).resolve()) if Path(model_name).exists() else model_name
    precision = (
        "gguf" if model_name.lower().endswith(".gguf")
        else "float16" if os.environ.get("JAYCE_DTYPE") == "float16" else "float32"
    )
    key = hashlib.sha256(f"{source}\n{precision}".encode()).hexdigest()[:16]
    return str(Path(__file__).with_name(f"jayce-prototypes-{key}.npz"))


def learn_examples(
    responder: PrototypeResponder,
    pairs: list[tuple[str, str]],
    prototype_file: str | None,
    *,
    correct: bool = False,
) -> int:
    """Use the same teaching and saving path for startup imports and /learn or /teach."""
    before = responder.memory.examples
    try:
        for question, answer in pairs:
            responder.teach(answer_context(question), answer, correct=correct)
    finally:
        # If a later example fails, retain and save the earlier complete examples.
        if responder.memory.examples > before:
            responder.save(prototype_file)
    return responder.memory.examples - before


def _style(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None:
        return text
    return f"\033[{code}m{text}\033[0m"


def prepare_responder(
    tokenizer,
    model,
    model_name: str,
    prototype_file: str | None,
    training_files: list[str],
) -> PrototypeResponder | None:
    try:
        encoder = ContextEncoder(tokenizer, model, model_name)
        memory = (
            TokenMemory.load(prototype_file, encoder.identity, encoder.vocab_size)
            if prototype_file and Path(prototype_file).exists()
            else TokenMemory()
        )
        responder = PrototypeResponder(encoder, memory)
        for path in training_files:
            learned = learn_examples(responder, read_examples(path), prototype_file)
            print(f"Jayce learned {learned} examples from {path}.", flush=True)
        return responder
    except (PrototypeError, OSError, RuntimeError) as error:
        print(
            f"Jayce is unavailable: {error} The language model can still answer.",
            flush=True,
        )
        return None


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


def teach_completed_answer(
    responder: PrototypeResponder | None,
    context: str,
    answer: str,
    completed: bool,
    enabled: bool,
    prototype_file: str | None,
    previous_reply: Reply | None = None,
    limit: int = 128,
) -> Reply | None:
    if not answer.strip():
        return
    # Agreement is textual, not a claim that the teacher knows the truth.
    if (
        previous_reply is not None
        and previous_reply.text is not None
        and " ".join(previous_reply.text.split()) == " ".join(answer.split())
    ):
        if completed:
            print(
                _style("  Parent confirms Jayce's answer. No correction needed.", "2"),
                flush=True,
            )
        return
    if not completed:
        print(
            _style(
                "  Jayce did not learn this answer: the model stopped before a complete ending.",
                "2",
            )
        )
        return
    correction = previous_reply is not None and previous_reply.text is not None
    if not enabled or responder is None:
        if correction:
            print(
                _style(
                    "  The parent gave a different answer. Jayce's learning is off or unavailable.",
                    "2",
                )
            )
        return
    if correction:
        print(
            _style("  Parent correction → Jayce learns and tries again.", "2"),
            flush=True,
        )
    else:
        print(_style("  Parent teaches Jayce → Jayce tries again.", "2"), flush=True)
    try:
        responder.teach(context, answer, teacher=True, correct=True)
        if prototype_file:
            try:
                responder.save(prototype_file)
            except OSError as error:
                print(
                    f"  Learned for this session, but could not save prototypes: {error}"
                )
        retry = try_prototype_reply(responder, context, limit)
        show_prototype_reply(retry, "correction" if correction else "after learning")
        return retry
    except (ValueError, OSError, RuntimeError) as error:
        print(f"  Jayce could not finish learning/saving: {error}", flush=True)


class ChatSession:
    """One parent model and its prototype memory, shared by both chat modes."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        history_turns: int = 8,
        prototype_file: str | None = None,
        learning_enabled: bool = True,
        training_files: list[str] | None = None,
    ) -> None:
        if max_new_tokens < 1 or history_turns < 1:
            raise ValueError("Reply length and history turns must be positive")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.history_turns = history_turns
        self.prototype_file = prototype_file
        self.learning_enabled = learning_enabled
        self.history: list[dict[str, str]] = []
        self.last_reply: Reply | None = None

        started = time.perf_counter()
        self.tokenizer, self.model = load_model(model_name)
        self.responder = prepare_responder(
            self.tokenizer, self.model, model_name, prototype_file, training_files or []
        )
        self.load_seconds = time.perf_counter() - started

    def answer(self, question: str) -> None:
        """Try APM first, get the parent's answer, then learn and retry if needed."""
        context = answer_context(question)
        # Make this attempt before the teacher answer exists; do not feed it to the parent.
        prototype_reply = try_prototype_reply(
            self.responder, context, self.max_new_tokens
        )
        self.last_reply = prototype_reply
        show_prototype_reply(prototype_reply)
        print(
            _style(f"Parent ({model_label(self.model_name)})> ", "1;32"),
            end="",
            flush=True,
        )
        completion: list[bool] = []
        try:
            answer = generate_chat(
                self.tokenizer,
                self.model,
                self.history,
                question,
                self.max_new_tokens,
                self.model_name,
                on_token=lambda token: print(token, end="", flush=True),
                on_finished=completion.append,
            )
        except Exception as error:  # noqa: BLE001 - a failed model turn must not close the chat.
            print(f"\nGeneration failed: {error}")
            return
        print()
        retry = teach_completed_answer(
            self.responder,
            context,
            answer,
            bool(completion and completion[0]),
            self.learning_enabled,
            self.prototype_file,
            prototype_reply,
            self.max_new_tokens,
        )
        if retry is not None:
            self.last_reply = retry
        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        self.history = self.history[-2 * self.history_turns :]


def handle_command(session: ChatSession, command: str, argument: str) -> None:
    """Handle one slash command without running a conversation turn."""
    if command == "/help":
        print(
            "Commands:\n"
            "  /teach question => answer  Teach Jayce a standalone example\n"
            "  /learn <file.jsonl>        Learn question/answer examples from a file\n"
            "  /learning on|off          Learn from completed model answers\n"
            "  /reset-jayce              Clear learned token prototypes\n"
            "  /clear                    Clear the current chat history\n"
            "  /stats                    Show prototype counts and the last match\n"
            "  /model                    Show the loaded model\n"
            "  /save                     Save token prototypes\n"
            "  /quit                     Exit"
        )
        return
    if command == "/learning":
        mode = argument.strip().lower()
        if mode not in {"on", "off"}:
            print(
                f"Learning is {'on' if session.learning_enabled else 'off'}. Use /learning on or /learning off."
            )
        else:
            session.learning_enabled = mode == "on"
            print(f"Learning from model answers {mode}.")
        return
    if command in {"/teach", "/learn", "/reset-jayce"}:
        if session.responder is None:
            print(
                "Jayce's prototype memory is unavailable. Restart with a separate --prototype-file."
            )
            return
        if command == "/reset-jayce":
            session.responder.memory = TokenMemory()
            session.responder.save(session.prototype_file)
            print("Jayce's token prototypes cleared.")
            return
        before = session.responder.memory.examples
        try:
            if command == "/teach":
                question, separator, answer = argument.partition("=>")
                if not separator or not question.strip() or not answer.strip():
                    print("Usage: /teach question => answer")
                    return
                pairs = [(question.strip(), answer.strip())]
            else:
                pairs = read_examples(argument.strip())
            learned = learn_examples(
                session.responder,
                pairs,
                session.prototype_file,
                correct=command == "/teach",
            )
            print(f"Jayce learned {learned} example(s). Ask a question to try it.")
        except (ValueError, OSError, RuntimeError) as error:
            print(
                f"Teaching stopped after {session.responder.memory.examples - before} example(s): {error}"
            )
        return
    if command == "/clear":
        session.history.clear()
        print("Conversation context cleared.")
        return
    if command == "/stats":
        print(
            f"Model: {session.model_name}\n"
            f"Context: {len(session.history) // 2} conversation turns "
            f"(limit {session.history_turns})"
        )
        if session.responder is not None:
            learned = session.responder.memory
            print(
                f"Jayce: {len(learned)} token prototypes from {learned.examples} answers "
                f"({learned.teacher_examples} model answers)\n"
                f"Learning from model answers: {'on' if session.learning_enabled else 'off'}"
            )
        if session.last_reply is not None:
            print(f"Last Jayce result: {session.last_reply.reason}")
            if session.last_reply.similarity is not None:
                print(
                    f"Weakest token similarity: {session.last_reply.similarity:.3f} (not a correctness probability)"
                )
        return
    if command == "/save":
        if session.responder is not None and session.prototype_file:
            try:
                session.responder.save(session.prototype_file)
                print("Token prototypes saved.")
            except OSError as error:
                print(f"Could not save token prototypes: {error}")
        else:
            print("No prototype memory is available to save.")
        return
    if command == "/model":
        print(session.model_name)
        return
    print("Unknown command. Type /help to see available commands.")


def chat(session: ChatSession) -> None:
    print(_style(LICENSE_NOTICE, "2"))
    print(f"Ready in {session.load_seconds:.1f}s")
    print(f"Model: {session.model_name}")
    if session.prototype_file:
        print(f"Prototypes: {session.prototype_file}")
    print(
        f"Jayce learns from model answers: {'on' if session.learning_enabled else 'off'}"
    )
    print(
        "Jayce tries first. The parent answers every turn. With learning on, differences teach Jayce."
    )
    print("Type /help for commands. Ctrl-D or /quit exits.\n")

    while True:
        try:
            user_text = input(_style("› ", "1;36")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return
        if not user_text:
            continue

        command, _, argument = user_text.partition(" ")
        command = command.lower()
        if command in {"/quit", "/exit", "/q"}:
            print("Session ended.")
            return
        if command.startswith("/"):
            try:
                handle_command(session, command, argument)
            except (OSError, RuntimeError, ValueError) as error:
                print(f"Command failed: {error}")
        else:
            session.answer(user_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jayce",
        description="Get a model answer and, when learned prototypes match, a separate Jayce answer.",
        epilog=LICENSE_NOTICE,
    )
    parser.add_argument(
        "--model", help="Parent model ID, local GGUF path, or hf://owner/repo/file.gguf"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum reply length (default: 128)",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=8,
        help="Number of earlier chat turns to keep (default: 8)",
    )
    parser.add_argument(
        "--prompt",
        help="Ask one question, then exit; omit to start an interactive chat",
    )
    parser.add_argument(
        "--prototype-file",
        help="File for learned next-token prototypes (default: separate file per model/precision)",
    )
    parser.add_argument(
        "--no-learning", action="store_true", help="Do not learn from model answers"
    )
    parser.add_argument(
        "--learn",
        action="append",
        default=[],
        metavar="FILE.jsonl",
        help="Learn explicit question/answer pairs before chatting; may be repeated",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.history_turns < 1:
        parser.error("--history-turns must be positive")
    model_name = args.model or PARENT_MODEL
    interactive = not args.prompt

    try:
        if interactive:
            print(
                _style("Jayce", "1;36")
                + _style("  try → model answer → learn → try again", "1")
            )
            print(f"Loading {Path(model_name).name} once...", flush=True)
        session = ChatSession(
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            history_turns=args.history_turns,
            prototype_file=args.prototype_file or default_prototype_file(model_name),
            learning_enabled=not args.no_learning,
            training_files=args.learn,
        )
        if interactive:
            chat(session)
        else:
            session.answer(args.prompt)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except ModelUnavailable as error:
        raise SystemExit(f"Jayce: {error}")


if __name__ == "__main__":
    main()
