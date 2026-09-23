# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Local model backends: loading and parent answers.

All Transformers and llama.cpp details live here. The APM learning rules are in
jayce_tokens.py and depend only on NumPy.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

import huggingface_hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jayce_tokens import PrototypeError


_SYSTEM_MESSAGE = (
    "Answer the user accurately and briefly. If you are uncertain, say so. "
    "For identification questions, reply with only the identified item. "
    "If asked to identify a quoted letter, return only that letter. "
    "JAYCE stands for Jayce Associates Your Categorized Exemplars. "
    "APM stands for Adaptive Prototype Memory."
)


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
            "Jayce stopped before loading it. Choose a smaller model in jayce_config.py, or "
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
            from llama_cpp import Llama
        except ImportError as error:
            raise ModelUnavailable(
                "GGUF models need llama-cpp-python, which is not installed. "
                "Run `uv sync --locked --extra parent` to install the project dependencies. "
                "Building llama-cpp-python requires C++ build tools."
            ) from error

        _require_free_memory(path)
        cpu_only = os.environ.get("JAYCE_CPU") == "1"
        if cpu_only:
            print("Running the GGUF model on CPU (JAYCE_CPU=1).", flush=True)
        model = Llama(
            model_path=path,
            n_ctx=4096,
            n_threads=max(4, os.cpu_count() or 4),
            # Bound output scratch buffers on machines with 8 GiB RAM.
            n_batch=128,
            n_ubatch=128,
            n_gpu_layers=0 if cpu_only else -1,
            offload_kqv=not cpu_only,
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


def chat_prompt_token_count(tokenizer, model, question: str, system_message: str) -> int:
    """Count the actual chat template without evaluating the model."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]
    if tokenizer is not None:
        return len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        ))

    from llama_cpp.llama_chat_format import get_chat_completion_handler

    # Use exactly the handler selected by Llama.create_chat_completion, including
    # its BOS/special-token rules. Stop at the completion boundary, before inference.
    handler = (
        model.chat_handler or model._chat_handlers.get(model.chat_format)
        or get_chat_completion_handler(model.chat_format)
    )

    class PromptCount(Exception):
        pass

    class PromptProbe:
        verbose = False

        def tokenize(self, *args, **kwargs):
            return model.tokenize(*args, **kwargs)

        def create_completion(self, prompt, **kwargs):
            tokens = (
                model.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
                if isinstance(prompt, str) else prompt
            )
            raise PromptCount(len(tokens))

    try:
        handler(llama=PromptProbe(), messages=messages, stream=False, max_tokens=1)
    except PromptCount as result:
        return result.args[0]
    except AttributeError as error:
        raise PrototypeError("Cannot measure the configured GGUF chat template.") from error
    raise PrototypeError("The GGUF chat handler did not expose a text prompt.")


def generate_chat(
    tokenizer,
    model,
    history: list[dict[str, str]],
    question: str,
    max_new_tokens: int,
    on_token: Callable[[str], None] | None = None,
    on_finished: Callable[[bool], None] | None = None,
    system_message: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_message or _SYSTEM_MESSAGE},
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
