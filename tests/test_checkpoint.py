# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import io
import signal
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from jayce_checkpoint import set_capacity, training_stop
from jayce_tokens import PrototypeError, TokenMemory


class CheckpointTests(unittest.TestCase):
    def test_capacity_can_grow_but_never_silently_shrink_saved_memory(self):
        memory = TokenMemory(max_prototypes=8)
        set_capacity(memory, None)
        self.assertEqual(memory.max_prototypes, 8)
        set_capacity(memory, 16)
        self.assertEqual(memory.max_prototypes, 16)
        with self.assertRaisesRegex(PrototypeError, 'can only increase'):
            set_capacity(memory, 4)
        self.assertEqual(memory.max_prototypes, 16)

    def test_first_signal_requests_a_save_second_aborts_and_handlers_are_restored(self):
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with patch('jayce_checkpoint.signal.signal', side_effect=lambda sig, handler: old_handlers[sig]) as install:
            with self.assertRaises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
                with training_stop() as stop:
                    handler = install.call_args_list[0].args[1]
                    self.assertFalse(stop.is_set())
                    handler(signal.SIGINT, None)
                    self.assertTrue(stop.is_set())
                    handler(signal.SIGINT, None)
            self.assertEqual([call.args for call in install.call_args_list[-2:]], list(old_handlers.items()))
