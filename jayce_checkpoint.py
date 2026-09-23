# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Small shared helpers for safe checkpoint writers and interruptible training."""

import signal
import threading
from contextlib import contextmanager

from jayce_tokens import PrototypeError


@contextmanager
def training_lock(run_dir):
    import fcntl

    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep the inode: deleting it can let two processes hold different locks.
    with (run_dir / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PrototypeError("Another trainer is already using this memory. Stop it before teaching here.") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def training_stop():
    stop = threading.Event()

    def request_stop(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        print("\nStopping after the current work is saved. Press Ctrl+C again to abort it.", flush=True)

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield stop
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def set_capacity(memory, capacity):
    if capacity is None:
        return
    if capacity < max(len(memory), memory.max_prototypes):
        raise PrototypeError("--capacity can only increase the saved capacity.")
    memory.max_prototypes = capacity
