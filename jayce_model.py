# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Local model backends: loading, parent answers, and frozen context vectors.

All Transformers and llama.cpp details live here. The APM learning rules are in
jayce_tokens.py and depend only on NumPy.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import huggingface_hub
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jayce_tokens import CONTEXT_FORMAT, PrototypeError, unit

_LLAMA_LOG_FILTER = None
_EMBEDDING_WARNING = b"embeddings required but some input tokens were not marked as outputs -> overriding"


def _filter_expected_embedding_warning() -> None:
    """Hide a known harmless warning from the local model library."""
    global _LLAMA_LOG_FILTER
    if _LLAMA_LOG_FILTER is not None:
        return

    import ctypes

    import llama_cpp
    from llama_cpp._logger import llama_log_callback as default_callback

    @llama_cpp.llama_log_callback
    def callback(level, message, user_data):
        if message and _EMBEDDING_WARNING in message:
            return
        default_callback(level, message, user_data)

    _LLAMA_LOG_FILTER = callback
    llama_cpp.llama_log_set(callback, ctypes.c_void_p(0))


class ModelUnavailable(RuntimeError):
    """The parent model could not be loaded. The message says what to do about it."""


def _announce_download(model_id: str) -> None:
    """Say so before a long wait: models download once, then load from disk."""
    if Path(model_id).exists() or huggingface_hub.constants.HF_HUB_OFFLINE:
        return
    if isinstance(huggingface_hub.try_to_load_from_cache(model_id, "config.json"), str):
        return
    print(
        f"First run: downloading {model_id} from Hugging Face. "
        "This happens once; later runs load it from disk.",
        flush=True,
    )


def _access_denied(error: BaseException) -> bool:
    """Transformers wraps Hub errors; inspect their HTTP status, not message text."""
    while error is not None:
        response = getattr(error, "response", None)
        if response is not None and response.status_code in (401, 403):
            return True
        error = error.__cause__ or error.__context__
    return False


def _load_transformers(path: str, dtype, *, token: bool | None = False):
    """Public models need no login; only send saved credentials when required."""
    tokenizer = AutoTokenizer.from_pretrained(path, token=token)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, token=token)
    return tokenizer, model


def _download_gguf(source: str) -> str:
    """Resolve hf://owner/repo/file.gguf, reusing the local cache offline."""
    parts = source.removeprefix("hf://").split("/", 2)
    if len(parts) != 3 or not all(parts) or not parts[2].lower().endswith(".gguf"):
        raise ModelUnavailable("Use hf://owner/repo/file.gguf for a remote GGUF model.")
    owner, repo, filename = parts
    repo_id = f"{owner}/{repo}"
    cached = huggingface_hub.try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str):
        return cached
    if not huggingface_hub.constants.HF_HUB_OFFLINE:
        print(
            f"First run: downloading {filename} from {repo_id}. "
            "This happens once; later runs load it from disk.",
            flush=True,
        )
    try:
        try:
            return huggingface_hub.hf_hub_download(repo_id, filename, token=False)
        except OSError as error:
            if not _access_denied(error) or not huggingface_hub.get_token():
                raise
            return huggingface_hub.hf_hub_download(repo_id, filename, token=None)
    except (OSError, ValueError) as error:
        if _access_denied(error):
            raise ModelUnavailable(
                f"Hugging Face denied access to '{source}'. Check the repository and filename. "
                "For a private or gated model, make sure your account has access "
                "and sign in with `uv run hf auth login`."
            ) from error
        raise ModelUnavailable(
            f"Could not download '{source}'. Check the repository, filename and internet "
            "connection. After the first download, the cached model works offline."
        ) from error


_GIB = 2**30
# Beyond the weights, a 4,096-token context needs about 1 GiB for the attention cache
# and llama.cpp's working buffers, plus Python and the imported libraries. Measured
# with Qwen3-4B on the CPU backend; the extra half GiB is margin.
_GGUF_EXTRA_MEMORY = 1.5 * _GIB
# The operating system and desktop need memory too while the model runs. Without this
# margin a computer with about 4 GiB in total passes on paper, then thrashes.
_SYSTEM_MEMORY = 1 * _GIB


def _available_memory(meminfo: str = "/proc/meminfo") -> int | None:
    """Bytes the system can give a new program without swapping; None if unknown.

    Only Linux reports this reliably. macOS and Windows keep memory in caches and
    compressed pages that they reclaim on demand, so their "available" figure can sit
    far below what a program can actually get (an 8 GiB Mac often shows about 1 GiB
    while loading this model fine). The check is skipped there rather than risk
    refusing a model that would run.
    """
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _total_memory() -> int | None:
    """Physical memory installed, in bytes; None if it cannot be read.

    Unlike "available" memory, this means the same thing on every system, so it can
    safely refuse a model that could never fit. psutil is installed with accelerate.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, OSError, RuntimeError):
        return None


def _require_free_memory(path: str) -> None:
    """Stop before loading a model that cannot fit, or cannot fit right now."""
    if os.environ.get("JAYCE_SKIP_MEMORY_CHECK") == "1":
        return
    needed = Path(path).stat().st_size + _GGUF_EXTRA_MEMORY
    total = _total_memory()
    if total is not None and total < needed + _SYSTEM_MEMORY:
        raise ModelUnavailable(
            f"This model needs about {(needed + _SYSTEM_MEMORY) / _GIB:.1f} GiB of "
            f"memory, including room for the system, but this computer has only "
            f"{total / _GIB:.1f} GiB. Running out of memory can freeze a computer, so "
            "Jayce stopped before loading it. Choose a smaller model with --model, or "
            "set JAYCE_SKIP_MEMORY_CHECK=1 to try anyway."
        )
    available = _available_memory()
    if available is not None and available < needed:
        raise ModelUnavailable(
            f"This model needs about {needed / _GIB:.1f} GiB of free memory, but only "
            f"{available / _GIB:.1f} GiB is available. Running out of memory can "
            "freeze a computer, so Jayce stopped before loading it. Close other "
            "programs and try again. To load the model anyway, set "
            "JAYCE_SKIP_MEMORY_CHECK=1."
        )


def load_model(path: str):
    """Return a tokenizer/model pair; a GGUF model handles its own tokenization."""
    if path.startswith("hf://"):
        path = _download_gguf(path)
    if path.lower().endswith(".gguf"):
        if not Path(path).is_file():
            raise ModelUnavailable(f"GGUF file not found: {path}")
        try:
            from llama_cpp import LLAMA_POOLING_TYPE_NONE, Llama
        except ImportError as error:
            raise ModelUnavailable(
                "GGUF models need llama-cpp-python, which is not installed. "
                "Run `uv sync --locked` to install the project dependencies. "
                "Building llama-cpp-python requires C++ build tools."
            ) from error

        _require_free_memory(path)
        _filter_expected_embedding_warning()
        model = Llama(
            model_path=path,
            n_ctx=4096,
            n_threads=max(4, os.cpu_count() or 4),
            n_gpu_layers=-1,
            embedding=True,
            pooling_type=LLAMA_POOLING_TYPE_NONE,
            # Use the model's own chat template rather than forcing a different format.
            chat_format=None,
            verbose=False,
        )
        return None, model
    _announce_download(path)
    dtype = (
        torch.float16 if os.environ.get("JAYCE_DTYPE") == "float16" else torch.float32
    )
    try:
        try:
            tokenizer, model = _load_transformers(path, dtype)
        except OSError as error:
            if not _access_denied(error) or not huggingface_hub.get_token():
                raise
            print(
                "Public model access was denied; trying your saved Hugging Face login.",
                flush=True,
            )
            tokenizer, model = _load_transformers(path, dtype, token=None)
    except OSError as error:
        if _access_denied(error):
            raise ModelUnavailable(
                f"Hugging Face denied access to '{path}'. Check the model ID. "
                "For a private or gated model, make sure your account has access "
                "and sign in with `uv run hf auth login`."
            ) from error
        lines = str(error).strip().splitlines()
        raise ModelUnavailable(
            f"Could not load model '{path}'. {lines[0] if lines else ''}\n"
            "Check the model name and your internet connection. A model is downloaded "
            "once, then works offline."
        ) from error
    model.eval()
    return tokenizer, model


def model_label(model_name: str) -> str:
    """Show a GGUF filename without its extension; keep model IDs intact."""
    path = Path(model_name)
    return path.stem if path.suffix.lower() == ".gguf" else model_name


def _system_message() -> str:
    return (
        "Answer the user accurately and briefly. If you are uncertain, say so. "
        "For identification questions, reply with only the identified item. "
        "If asked to identify a quoted letter, return only that letter. "
        "JAYCE stands for Jayce Associates Your Categorized Exemplars. "
        "APM stands for Adaptive Prototype Memory."
    )


def generate_chat(
    tokenizer,
    model,
    history: list[dict[str, str]],
    question: str,
    max_new_tokens: int,
    model_name: str,
    on_token: Callable[[str], None] | None = None,
    on_finished: Callable[[bool], None] | None = None,
) -> str:
    system_message = _system_message()
    messages = [
        {"role": "system", "content": system_message},
        *history,
        {"role": "user", "content": question},
    ]
    if tokenizer is None:
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stream=on_token is not None,
        )
        if on_token is None:
            if on_finished:
                on_finished(result["choices"][0].get("finish_reason") == "stop")
            return result["choices"][0]["message"]["content"].strip()

        pieces: list[str] = []
        finished = False
        for event in result:
            choices = event.get("choices") or []
            if not choices:
                continue
            finished = finished or choices[0].get("finish_reason") == "stop"
            token = choices[0].get("delta", {}).get("content", "")
            if token:
                pieces.append(token)
                on_token(token)
        if on_finished:
            on_finished(finished)
        return "".join(pieces).strip()

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = input_ids.to(next(model.parameters()).device)
    generation_options = {
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "pad_token_id": tokenizer.eos_token_id,
    }

    def ended(output) -> bool:
        eos = getattr(model.generation_config, "eos_token_id", None)
        eos_ids = eos if isinstance(eos, list) else [eos]
        return output.shape[-1] > input_ids.shape[-1] and int(output[0, -1]) in eos_ids

    if on_token is None:
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                **generation_options,
            )
        new_tokens = output[0, input_ids.shape[-1] :]
        if on_finished:
            on_finished(ended(output))
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_errors: list[BaseException] = []
    completed: list[bool] = []

    def run_generation() -> None:
        try:
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    streamer=streamer,
                    **generation_options,
                )
                completed.append(ended(output))
        except BaseException as error:  # noqa: BLE001 - re-raised on the caller thread below.
            # End the iterator on every worker failure so the caller cannot hang.
            generation_errors.append(error)
            streamer.end()

    worker = threading.Thread(target=run_generation, daemon=True)
    worker.start()
    pieces: list[str] = []
    for piece in streamer:
        if piece:
            pieces.append(piece)
            on_token(piece)
    worker.join()
    if generation_errors:
        raise generation_errors[0]
    if on_finished:
        on_finished(bool(completed and completed[0]))
    return "".join(pieces).strip()


class ContextEncoder:
    """Read final-layer context vectors from Transformers or llama.cpp.

    Both paths use causal attention, retaining a cache while APM selects tokens.
    No sampling, generation call, or output-logit selection occurs here.
    """

    def __init__(self, tokenizer, model, model_name: str) -> None:
        self.tokenizer, self.model = tokenizer, model
        self.past = None
        self.used = 0
        self.backend = "llama.cpp" if tokenizer is None else "transformers"
        if tokenizer is None:
            self.vocab_size = model.n_vocab()
            self.context_limit = model.n_ctx()
        else:
            self.vocab_size = len(tokenizer)
            self.context_limit = int(
                getattr(model.config, "max_position_embeddings", 4096)
            )
        signature = {
            "format": CONTEXT_FORMAT,
            "backend": self.backend,
            "model": model_name,
            "vocabulary": self.vocab_size,
        }
        if tokenizer is not None:
            signature["precision"] = str(model.dtype)
            signature["tokenizer"] = hashlib.sha256(
                json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
            ).hexdigest()
        # A remote GGUF's identity must include the downloaded weights too.
        path = Path(model.model_path if model_name.startswith("hf://") else model_name)
        if path.is_file():
            signature.update(
                path=str(path.resolve()),
                bytes=path.stat().st_size,
                modified=path.stat().st_mtime_ns,
            )
        else:
            signature["revision"] = getattr(
                getattr(model, "config", None), "_commit_hash", None
            )
        self.identity = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()

    def encode(self, text: str, *, prefix: bool = False) -> list[int]:
        if self.tokenizer is None:
            return self.model.tokenize(
                text.encode("utf-8"), add_bos=prefix, special=False
            )
        return self.tokenizer.encode(text, add_special_tokens=prefix)

    def decode(self, tokens: list[int]) -> str:
        if self.tokenizer is None:
            return self.model.detokenize(tokens, special=False).decode(
                "utf-8", errors="replace"
            )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def start(self, tokens: list[int]) -> np.ndarray:
        self.close()
        return self._evaluate(tokens)

    def advance(self, token: int) -> np.ndarray:
        return self._evaluate([token])

    def _evaluate(self, tokens: list[int]) -> np.ndarray:
        if not tokens or self.used + len(tokens) > self.context_limit:
            raise PrototypeError(
                "Question and answer exceed the model's context window."
            )
        if self.tokenizer is None:
            import llama_cpp

            if self.model.pooling_type() != llama_cpp.LLAMA_POOLING_TYPE_NONE:
                raise PrototypeError(
                    "Jayce needs token-level embeddings (pooling NONE)."
                )
            self.model.eval(tokens)
            pointer = self.model._ctx.get_embeddings_ith(-1)
            if not pointer:
                raise PrototypeError(
                    "This model did not expose a final-token context vector."
                )
            vector = np.ctypeslib.as_array(pointer, shape=(self.model.n_embd(),)).copy()
        else:
            device = next(self.model.parameters()).device
            with torch.inference_mode():
                output = self.model.base_model(
                    input_ids=torch.tensor([tokens], dtype=torch.long, device=device),
                    past_key_values=self.past,
                    use_cache=True,
                    return_dict=True,
                )
            self.past = output.past_key_values
            vector = output.last_hidden_state[0, -1].float().cpu().numpy().copy()
        self.used += len(tokens)
        return unit(vector)

    def close(self) -> None:
        self.past = None
        self.used = 0
        if self.tokenizer is None:
            self.model.reset()
