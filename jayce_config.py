# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Shared defaults for native training and optional parent teaching."""

from pathlib import Path

# Hugging Face model ID, hf://owner/repo/file.gguf, or local GGUF path.
# Remote GGUF files download once and are cached.
# The 4-bit 4B Instruct parent is a ~2.5 GB download, suitable for an 8 GB Mac.
PARENT_MODEL = (
    "hf://bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/"
    "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)

DEFAULT_NATIVE_RUN = Path(__file__).resolve().parent / "data/fineweb-edu/native"

BATCH_SIZE = 5
AUTO_CAPACITY_LIMIT = 32768
# Offline fallback before the first Wikipedia topic download.
DEFAULT_TOPICS = ["basic arithmetic", "basic geometry", "the solar system", "plants and animals", "weather",
                  "world geography", "everyday physics", "basic chemistry", "human history", "everyday technology"]
