# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Choose the parent model. Restart Jayce after changing it."""

# Hugging Face model ID, hf://owner/repo/file.gguf, or local GGUF path.
# --model overrides this for one run. Remote GGUF files download once and are cached.
# The 4-bit 4B Instruct parent is a ~2.5 GB download, suitable for an 8 GB Mac.
PARENT_MODEL = (
    "hf://bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/"
    "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)
