# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Stream Parquet text into bounded APM memory, checkpoint, pause, and resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jayce_config import DEFAULT_NATIVE_RUN
from jayce_checkpoint import set_capacity, training_lock, training_stop
from jayce_text import MAX_DOCUMENT_CHARS, STATE_FORMAT, load_text_memory, read_state
from jayce_tokens import DEFAULT_CAPACITY, PrototypeError


def position(file=0, group=0, row=0, offset=0):
    return {"file": file, "group": group, "row": row, "offset": offset}


class ParquetCorpus:
    def __init__(self, path: Path):
        import pyarrow.parquet as pq

        self.files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        if not self.files or any(not p.is_file() or p.suffix != ".parquet" for p in self.files):
            raise PrototypeError(f"No Parquet files found in {path}.")
        self.groups = []
        inventory = []
        for file in self.files:
            stat = file.stat()
            inventory.append((str(file.resolve()), stat.st_size, stat.st_mtime_ns))
            with pq.ParquetFile(file) as parquet:
                if "text" not in parquet.schema_arrow.names:
                    raise PrototypeError(f"{file}: expected a text column.")
                self.groups.append([
                    parquet.metadata.row_group(i).num_rows
                    for i in range(parquet.num_row_groups)
                ])
        self.identity = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
        self.rows = sum(sum(groups) for groups in self.groups)

    def documents(self, cursor):
        """Read only text, starting at the saved row group; no whole-shard load."""
        import pyarrow.parquet as pq

        if (
            set(cursor) != {"file", "group", "row", "offset"}
            or any(type(value) is not int or value < 0 for value in cursor.values())
            or cursor["file"] > len(self.files)
        ):
            raise PrototypeError("Invalid saved corpus position.")
        if cursor["file"] == len(self.files):
            if cursor != position(file=len(self.files)):
                raise PrototypeError("Invalid end-of-corpus position.")
            return
        groups = self.groups[cursor["file"]]
        if groups and (
            cursor["group"] >= len(groups)
            or cursor["row"] > groups[cursor["group"]]
            or (cursor["offset"] and cursor["row"] == groups[cursor["group"]])
        ):
            raise PrototypeError("Saved position is outside this corpus.")
        if not groups and any(cursor[key] for key in ("group", "row", "offset")):
            raise PrototypeError("Saved position is outside this empty shard.")
        for file_index in range(cursor["file"], len(self.files)):
            with pq.ParquetFile(self.files[file_index]) as parquet:
                first_group = cursor["group"] if file_index == cursor["file"] else 0
                for group in range(first_group, parquet.num_row_groups):
                    first_row = cursor["row"] if (file_index, group) == (
                        cursor["file"], cursor["group"]
                    ) else 0
                    if first_row == self.groups[file_index][group]:
                        continue
                    row = 0
                    for batch in parquet.iter_batches(
                        batch_size=16, row_groups=[group], columns=["text"], use_threads=False,
                    ):
                        for value in batch.column(0):
                            if row >= first_row:
                                yield value.as_py(), position(file_index, group, row)
                            row += 1


def new_state(corpus, model, context):
    return {
        "format": STATE_FORMAT, "corpus": corpus.identity,
        "model": model, "context": context, "cursor": position(),
        "batches": 0, "targets": 0, "documents": 0, "skipped_empty": 0,
        "status": "ready",
    }


def validate_resume(state, corpus, model, context):
    if (state["corpus"], state["model"], state["context"]) != (corpus.identity, model, context):
        raise PrototypeError(
            "The corpus, model, or context changed. Restore the original settings "
            "before resuming this memory."
        )


def train(learner, corpus, state, checkpoint, *, batch_tokens, sleep_seconds, max_batches, stop):
    batches_this_run = 0

    def save(status):
        state["status"] = status
        # Memory and cursor share one atomic NPZ replacement. A crash cannot
        # advance the cursor without committing the corresponding prototypes.
        learner.memory.save(str(checkpoint), learner.identity, training_state=state)

    def halt_reason():
        if stop.is_set() or (max_batches and batches_this_run >= max_batches):
            return "paused"
        if len(learner.memory) >= learner.memory.max_prototypes:
            return "capacity"
        return None

    if state["status"] == "complete":
        return "complete"
    save("running")
    cursor = state["cursor"].copy()
    for text, here in corpus.documents(cursor):
        if reason := halt_reason():
            save(reason)
            return reason
        source = f"{corpus.files[here['file']].name}, group {here['group']}, row {here['row']}"
        if text is not None and not isinstance(text, str):
            raise PrototypeError(f"{source}: text must be a string or null.")
        next_row = {**here, "row": here["row"] + 1}
        offset = cursor["offset"] if all(here[k] == cursor[k] for k in ("file", "group", "row")) else 0
        if text is None or not text.strip():
            if offset:
                raise PrototypeError(f"{source}: cannot resume inside an empty document.")
            state["cursor"] = next_row
            state["skipped_empty"] += 1
            continue
        if len(text) > MAX_DOCUMENT_CHARS:
            raise PrototypeError(f"{source}: document exceeds {MAX_DOCUMENT_CHARS} characters.")
        tokens = learner.encoder.encode(text, prefix=True)
        if not tokens or offset >= len(tokens):
            raise PrototypeError(f"{source}: invalid tokenized document or saved offset.")
        while offset < len(tokens):
            if reason := halt_reason():
                save(reason)
                return reason
            free = learner.memory.max_prototypes - len(learner.memory)
            result = learner.teach_tokens(tokens, min(batch_tokens, free), offset=offset)
            offset += result["targets"]
            state["targets"] += result["targets"]
            state["batches"] += 1
            batches_this_run += 1
            if result["complete_document"]:
                state["documents"] += 1
                state["cursor"] = next_row
            else:
                state["cursor"] = {**here, "offset": offset}
            reason = halt_reason()
            save(reason or "running")
            print(
                f"Batch {state['batches']}: +{result['targets']} targets "
                f"({state['targets']} total), {state['documents']} complete documents; "
                f"memory {len(learner.memory)}/{learner.memory.max_prototypes}; "
                f"{source}, target {offset}/{len(tokens)}. Saved.", flush=True,
            )
            if reason:
                return reason
            if sleep_seconds:
                print(f"Sleeping {sleep_seconds:g}s...", flush=True)
                stop.wait(sleep_seconds)
    state["cursor"] = position(file=len(corpus.files))
    save("complete")
    return "complete"


def run_training(data, capacity=None, *, run_dir=DEFAULT_NATIVE_RUN):
    from jayce_native import ByteContextEncoder, DEFAULT_CONTEXT, NATIVE_MODEL

    data = Path(data).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint = run_dir / "prototypes.npz"
    with training_lock(run_dir), training_stop() as stop:
        corpus = ParquetCorpus(data)
        state = read_state(checkpoint) or new_state(corpus, NATIVE_MODEL, DEFAULT_CONTEXT)
        context = state["context"]
        if state["corpus"] != corpus.identity and state["model"] == NATIVE_MODEL:
            print("Learning a new data source. Keeping previously learned memory.", flush=True)
            state = new_state(corpus, NATIVE_MODEL, context)
        validate_resume(state, corpus, NATIVE_MODEL, context)
        learner = load_text_memory(ByteContextEncoder(), str(checkpoint), context_tokens=context,
                                   capacity=capacity or DEFAULT_CAPACITY)
        set_capacity(learner.memory, capacity)
        if state["status"] == "complete":
            print("This data source is already learned. Pass another file or directory to learn more.")
            return
        print(f"Data: {data} ({len(corpus.files)} files, {corpus.rows:,} rows)")
        print("Training Jayce directly from text. No parent model is loaded.", flush=True)
        reason = train(learner, corpus, state, checkpoint, batch_tokens=256,
                       sleep_seconds=1, max_batches=0, stop=stop)
        print(f"Stopped: {reason}. Memory saved.")
        if reason == "capacity":
            print("Memory is full. Increase --capacity to continue.")
