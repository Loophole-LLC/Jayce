<p align="center">
  <img src="https://loophole.company/assets/jayce-logo.svg" alt="JAYCE — Adaptive Prototype Memory" width="460">
</p>

**JAYCE** stands for **Jayce Associates Your Categorized Exemplars**, inspired by watching a toddler
learn toy names through examples and corrections. This project is for reading the code and learning
how **Adaptive Prototype Memory (APM)** works.

APM keeps a limited set of labeled prototypes. Each starts from an **exemplar**—a specific example
saved for comparison—and can change as more examples arrive.

<p align="center">
  <img src="https://loophole.company/assets/jayce-demo.gif?v=20260921-1" alt="Terminal recording of Jayce. Jayce says it doesn't know, the parent model answers, and Jayce learns and repeats the answer. Later, Jayce is taught a wrong answer and the parent corrects it." width="788">
</p>
<p align="center"><sub>A real session with Qwen3-4B Instruct (Q4_0) as the parent. Model loading is skipped and long waits are shortened.</sub></p>

- **[Watch Jayce learn](#watch-jayce-learn):** a real chat showing learning, correction, and recall.
- **[APM vs. backpropagation](#benchmark-results):** accuracy, training time, and memory comparisons.
- **[Read the code](#read-the-code):** where to start and what each file does.

## Four commands

| Command | Behavior |
|---|---|
| `./jayce` | Chat with the parent; learn and save automatically |
| `./jayce --no-parent` | Chat using Jayce's saved memory alone |
| `./jayce train` | Learn random lessons from the parent |
| `./jayce train DATA` | Learn from a Parquet file or directory without a parent |

All four use the same memory in `data/fineweb-edu/native/`: `lessons.npz` for answers
and `prototypes.npz` for text. Changing modes does not change the learner.
Chat learning and saving are on by default. Training saves automatically; **Ctrl+C**
pauses it, and running the same command continues from saved progress.

Type `/stats` for memory usage and `/quit` to exit. `/learning off` pauses automatic
teaching in a parent chat.
`./jayce --help` shows the four commands.

### Capacity

Capacity is the one optional training setting:

```sh
./jayce train --capacity 32768
./jayce train data/fineweb-edu --capacity 32768
```

It limits **prototype slots, not lessons**, for that training run. An answer can use
many slots. Adaptive Q&A memory also keeps at most that many question feature vectors,
shared across answer positions. Existing memory is preserved; increase capacity to continue when full.
Without the option, parent teaching can grow memory as needed up to 32,768 slots.
New data-training memories start with 4,096 slots and retain their saved capacity.

### Learning from the parent

`./jayce train` downloads the topic names from Wikipedia's
[Vital Articles Level 4](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level_4)
catalog, covering roughly 10,000 subjects, including languages and calculus. It caches
them in `data/fineweb-edu/native/topics.json` and reuses them offline. If the first
download fails, it says so and uses the saved or starter topics. Remove only
`topics.json` to download an updated catalog on the next parent training run.

Jayce picks random topics from this catalog and asks Qwen for short factual lessons;
it downloads topic names, not article text. For each fact, it asks Qwen for
**1–5 different ways to ask the question**, with
one consistent answer. Qwen is also asked to pair neighboring facts with similar questions
and different answers, such as the capitals of France and Germany. These contrasts help
APM keep similar-looking questions with different answers apart.
Jayce learns each wording, so both “How many sides does a triangle
have?” and “What is the number of sides in a triangle?” can recall the same answer.
Repeated question wordings within a lesson are collapsed; familiar facts can still be
practiced again in later lessons. Each learned wording is checked for recall and saved
automatically. Related wordings can share prototypes; distinct contexts need new slots.
No topic, lesson list, or manual teaching prompt is needed.

It keeps going until you stop it, memory fills, or repeated attempts produce no usable
examples. Repetition does not stop training. It preserves unfinished examples on
restart. Qwen can make mistakes;
verified recall means Jayce can replay the supplied answer, not that it is correct.

### Learning from data

```sh
./jayce train data/fineweb-edu
./jayce train /path/to/shard.parquet
./jayce --no-parent
```

Directories are searched recursively for Parquet files with a `text` column. Quote paths
containing spaces. Training learns text continuations directly, without loading Qwen.
Enter an exact prefix from learned text to try recall; raw text does not automatically
become question-and-answer knowledge. Resume with the same data source, or give another
file to learn more while keeping the existing memory.

Jayce is an experimental text-pattern learner. Q&A features share words, short ordered
phrases, and spelling patterns across questions. This supports some unfamiliar wordings,
but does not supply general semantic understanding or mathematical rules. Unfamiliar
synonyms and questions may still get “Jayce don't know.”

The launcher installs dependencies through uv and downloads Qwen on the first parent run.
Later runs reuse the cached model. Independent chat and data training do not install
parent libraries.

## Parent model setup

**Requirements:** about 4 GB of free memory and 5 GB of disk, which in practice means a computer
with 6 GB of memory or more. Jayce stops before loading the model if the computer does not have
enough memory in total, and on Linux also if too little is free right now, because running out of
memory can freeze the whole computer. Choose a smaller model in [jayce_config.py](jayce_config.py), close other programs,
or set `JAYCE_SKIP_MEMORY_CHECK=1` to load it anyway.

If a GGUF model runs out of GPU memory while answering, run `JAYCE_CPU=1 ./jayce`
to keep its weights and attention computation on the CPU. This is slower and still
needs enough system memory. The same setting works with `./jayce train`;
training from a data path does not use a GPU.

### macOS (Apple silicon), Linux, or WSL

Install C++ build tools first: Xcode Command Line Tools on macOS (`xcode-select --install`)
or `build-essential` on Ubuntu/WSL.

```sh
git clone https://github.com/Loophole-LLC/Jayce.git
cd Jayce
./jayce
```

The launcher installs [uv](https://docs.astral.sh/uv/) if needed, sets up Python 3.12 and the
dependencies, and downloads **Qwen3-4B-Instruct-2507** in 4-bit GGUF format
([Q4_K_M, about 2.5 GB](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF)).
Later runs reuse the cached model, including offline. No Hugging Face login is needed.

### Windows (64-bit Intel/AMD)

Install Git, [uv](https://docs.astral.sh/uv/getting-started/installation/#winget), and the
Visual Studio C++ build tools from PowerShell:

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
```

The last command installs [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
with the **Desktop development with C++** workload, including the **MSVC x64/x86 C++ build
tools** and a **Windows SDK**. Jayce needs these to build the llama.cpp backend during setup.
The download is large, and Windows may ask you to approve the installer. To install the tools
yourself, run the Build Tools installer and select that workload.

Open **x64 Native Tools Command Prompt for VS** from the Start menu, then run:

```bat
git clone https://github.com/Loophole-LLC/Jayce.git
cd Jayce
uv run --locked --extra parent jayce.py
```

uv sets up Python and the dependencies; Jayce downloads the same parent model shown above.
After setup, run `uv run --locked --extra parent jayce.py` from the Jayce folder in PowerShell to chat again.
Use that command wherever the parent-mode examples below say `./jayce`, and keep commands on one line.

<details>
<summary>Windows setup errors</summary>

**Certificate verification errors:** Tell uv to use the certificates trusted by Windows:

```bat
uv --system-certs run --locked --extra parent jayce.py
```

This can help on networks with a company proxy or custom certificates.
See [uv's certificate documentation](https://docs.astral.sh/uv/concepts/authentication/certificates/).

**`CMAKE_C_COMPILER` or `CMAKE_CXX_COMPILER` not set:** CMake could not configure the C/C++
compiler. Usually the build tools are missing or unavailable in the current terminal.

1. Open **Visual Studio Installer → Modify** and check that the C++ workload and components
   listed above are installed.
2. Open a new **x64 Native Tools Command Prompt for VS** from the Start menu.
3. Run `where.exe cl`. It should print a path to Microsoft's `cl.exe` compiler. If it finds
   nothing, recheck the installation and the terminal you opened.
4. In that same window, change to the Jayce folder and retry the command above.

The developer prompt sets the compiler paths and SDK environment for you.
See [Microsoft's build-tools guide](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line).

</details>

The default backend uses Metal on supported Macs and CPU on Windows/Linux.

## Watch Jayce learn

**Jayce tries → the parent answers → APM learns → Jayce tries again.**

In this recorded chat, **Qwen3-4B Instruct (Q4_0)** is the parent model. Jayce starts with empty memory:

```text
› How many wheels does a bicycle have?
Jayce> Jayce don't know
Parent (Qwen3-4B-Instruct-2507-Q4_0)> A bicycle has 2 wheels.
  Parent teaches Jayce → Jayce tries again.
Jayce (after learning)> A bicycle has 2 wheels.

› How many wheels does a skateboard have?
Jayce> Jayce don't know
Parent (Qwen3-4B-Instruct-2507-Q4_0)> A skateboard has 4 wheels.
  Parent teaches Jayce → Jayce tries again.
Jayce (after learning)> A skateboard has 4 wheels.
```

Ask the bicycle question again, and Jayce remembers:

```text
› How many wheels does a bicycle have?
Jayce> A bicycle has 2 wheels.
Parent (Qwen3-4B-Instruct-2507-Q4_0)> A bicycle has 2 wheels.
  Parent confirms Jayce's answer. No correction needed.
```

Current chat also practices and saves confirmed answers again.

Teach an incorrect answer, then let the parent correct it:

```text
› /teach How many sides does a triangle have? => 4
Jayce learned 1 example(s). Ask a question to try it.

› How many sides does a triangle have?
Jayce> 4
Parent (Qwen3-4B-Instruct-2507-Q4_0)> A triangle has 3 sides.
  Parent correction → Jayce learns and tries again.
Jayce (correction)> A triangle has 3 sides.
```

The correction remains after restarting with the same prototype file:

```text
› How many sides does a triangle have?
Jayce> A triangle has 3 sides.
Parent (Qwen3-4B-Instruct-2507-Q4_0)> A triangle has three sides.
  Parent correction → Jayce learns and tries again.
Jayce (correction)> A triangle has three sides.
```

Jayce recalled the corrected answer from the saved file. This time the parent wrote “three” instead
of “3”, and answers are compared as text, so Jayce learned that wording too.

With learning on, Qwen answers every turn using fixed pretrained weights. Jayce encodes
the question and answer prefix using its own native features and stores prototypes labeled
with the next byte or the end of the answer. Its reply and retry come from these prototype
matches. Qwen is no longer needed to recall the learned answer in a later session.

<details>
<summary>Try the conversation yourself</summary>

Run `./jayce` and enter the questions and teaching command above. Use `/quit`, then
`./jayce --no-parent` to check recall without loading Qwen. Existing memory is reused.
The recording used Q4_0; the default is Q4_K_M. Parent answers can differ.

</details>

**Jayce can also learn the parent's mistakes.** `/learning off` switches to answers from
Jayce's memory alone: no parent reply, correction, or automatic learning. Start in this mode
with `./jayce --no-parent`. In a parent chat, use `/learning on` to resume teaching.
When Jayce has no strong complete match, it says **Jayce don't know** without asking the parent.
`./jayce --no-parent` uses the same memories without loading Qwen at all.

## Benchmark results

This Java benchmark compares APM with a neural network that has one hidden layer and learns
through backpropagation using Adam. Backpropagation calculates how each weight affects the error;
Adam uses those values to update the weights. Neither learner replays old training examples.

The tests use rotated digits, rotated clothing images, and two sets of generated shapes. Each
stream passes through twelve phases with changing conditions. Settings are chosen on two random
seeds and tested on ten separate seeds. Seeds make the random choices repeatable.

**Latest run: September 21, 2026.** The full run took **6 minutes 49 seconds** on an Apple A18 Pro
with 8 GiB RAM, macOS 27.0, and OpenJDK 26.0.1.

### Accuracy

Each score measures final accuracy across all four conditions, averaged over ten test seeds.
The comparisons use:

- **Same examples:** the same number of training examples.
- **Same counted work:** similar estimated computation, rather than elapsed time.
- **Same memory ceiling:** the same examples, with APM's storage capped at the estimated space used
  by the network's weights and Adam's saved values.

| Test stream | Comparison | APM accuracy | Backprop accuracy |
|---|---|---:|---:|
| Rotated digits (MNIST) | Same examples | 88.63% | 82.59% |
| Rotated digits (MNIST) | Same counted work | 87.37% | 77.36% |
| Rotated digits (MNIST) | Same memory ceiling | 87.88% | 82.59% |
| Rotated clothing (Fashion-MNIST) | Same examples | 69.28% | 71.75% |
| Rotated clothing (Fashion-MNIST) | Same counted work | 68.33% | 66.23% |
| Rotated clothing (Fashion-MNIST) | Same memory ceiling | 68.05% | 71.75% |
| Shapes, mixed | Same examples | 89.33% | 97.36% |
| Shapes, mixed | Same counted work | 86.56% | 94.05% |
| Shapes, mixed | Same memory ceiling | 86.16% | 97.36% |
| Shapes, shuffled pixels | Same examples | 92.13% | 94.65% |
| Shapes, shuffled pixels | Same counted work | 89.83% | 91.50% |
| Shapes, shuffled pixels | Same memory ceiling | 86.26% | 94.65% |

### Training time

Average training time for the **same examples** comparison: 12,288 examples per test seed.
Times are in milliseconds (ms).

| Test stream | APM training time (ms) | Backprop training time (ms) |
|---|---:|---:|
| Rotated digits (MNIST) | 72.56 | 279.05 |
| Rotated clothing (Fashion-MNIST) | 77.73 | 308.14 |
| Shapes, mixed | 17.05 | 26.59 |
| Shapes, shuffled pixels | 19.07 | 31.25 |

These times cover learner updates only. They exclude data preparation, predictions, and accuracy
checks. Background desktop activity was not controlled, so timings may vary between runs.

### What we learned

- **APM had higher accuracy in all three digit comparisons**, including with a memory cap.
- **Backprop had higher accuracy in 8 of 12 comparisons.** It won every shape comparison and the
  clothing tests with equal examples or memory. APM won the clothing test with equal counted work.
- **APM's training calls were about 1.6–4.0 times faster** in the four equal-example tests.
- **APM used more model memory with equal examples:** about 735 versus 466 KiB for the image tests,
  and 96 versus 52 KiB for the shape tests. These estimates exclude Java overhead and temporary
  memory. A KiB is 1,024 bytes.

This comparison covers one network and four test streams. It does not establish an overall
advantage for APM or measure chat speed. Other prototype methods, nearest-neighbor lookup,
and neural networks with replay remain untested.

See the [results guide](benchmark/results/README.md) for detailed scores and uncertainty estimates.
Run `./benchmark/run-benchmark` with JDK 17 or newer to repeat the comparison.
The [benchmark guide](benchmark/README.md) explains the settings and output files.

## How APM works

In the Java classifier, an input is a list of numbers, such as image pixels. Each label has a few
prototype slots:

1. Receive an input and its correct label.
2. If that label has an empty slot, save a copy of the input.
3. Otherwise, move its closest prototype toward the input.
4. Predict a label by finding the closest prototype across all labels.

The update is `prototype += rate * (input - prototype)`. With a rate of 0.1, each value moves
one tenth of the way toward the new example.

The chat uses byte labels and a shared pool of slots. A prototype represents a family
of question features at one specific answer prefix. New examples can join a nearby
prototype with the same output if the resulting cluster stays close to its examples
and separated from competing outputs. Its center is the normalized, practice-weighted
mean of its question exemplars. Repetition changes those weights.

Question exemplars are stored once and referenced from each answer position. Exact
taught contexts follow their prototype membership, preserving recall as centers move.
Unseen wordings use prototype similarity and the margin over a different output.
Corrections detach the corrected question from its old clusters, recenter the remaining
members, and teach the replacement. Each emitted byte still comes from a prototype label.

[![JAYCE toy-label lesson](https://loophole.company/assets/jayce-toy-learning.png?v=20260921-2)](https://loophole.company/jayce-toy-learning.html)

[Step through the toy lesson](https://loophole.company/jayce-toy-learning.html) to see examples,
corrections, and prototype updates. Its map places toys using illustrative body-length and bulk
scores; similar scores put toys close together, even when their labels differ.

Related reading: [Exemplar theory](https://en.wikipedia.org/wiki/Exemplar_theory).

### Native text memory

Jayce converts text into UTF-8 bytes. During teaching, its lesson encoder represents
the question with shared word, phrase, and character features. A separate fingerprint
identifies the exact answer prefix plus surface guards for numbers, operators, negation,
and quoted text. APM associates those features with the next byte or the answer's ending.
These guards are conservative checks, not a complete parser of meaning.
Recall follows the associations until the learned ending.
Weak or ambiguous matches make Jayce withhold the answer. Corrections update the
associations without changing Qwen's weights.

Data training uses a sliding byte context to learn text continuations. Neither form of
recall needs Qwen. The Java classifier benchmarks above do not measure this chat
pipeline's speed or memory use.

## Read the code

Start with the Java learner, then follow the Python chat:

1. **[JayceMemory.java](benchmark/reference/java/company/loophole/jayce/apm/JayceMemory.java)** — the
   standalone Java APM learner. Read `train`, `updateOne`, and `predictClass` for the learning rule.
2. **[jayce_tokens.py](jayce_tokens.py)** — the Python token-prototype memory, using only NumPy.
   `TokenMemory.learn` saves and updates prototypes; `match` picks a token label.
   `PrototypeResponder.teach` and `reply` connect those steps to whole answers.
3. **[jayce.py](jayce.py)** — start at `ChatSession.answer` to follow
   **try → parent answer → learn → retry**. The remaining code handles commands and saved files.

Supporting files:

| File | Role |
|---|---|
| [jayce_cli.py](jayce_cli.py) | Select parent chat, independent recall, or training from the command and data path |
| [jayce_native.py](jayce_native.py) | Parent-free byte encoders and compatible memory loading |
| [jayce_checkpoint.py](jayce_checkpoint.py) | Shared writer lock, capacity checks, and graceful interruption |
| [jayce_lessons.py](jayce_lessons.py) | Shared lesson lookup, verified teaching, and locked atomic saves |
| [jayce_parent_train.py](jayce_parent_train.py) | Generate lessons from Qwen's knowledge and teach native memory automatically |
| [jayce_train.py](jayce_train.py) | Stream Parquet text, save progress, and resume training |
| [jayce_model.py](jayce_model.py) | Load a parent model and generate answers |
| [jayce_text.py](jayce_text.py) | Learn document tokens directly and measure raw-text continuation |
| [jayce_config.py](jayce_config.py) | Shared native and parent defaults |
| [SoftmaxBackpropNetwork.java](benchmark/reference/java/company/loophole/jayce/backprop/SoftmaxBackpropNetwork.java) | The neural network used for comparison |

## Chat and teach

Run `./jayce` and type a question to chat.
Normal chat learns from the parent automatically and saves each completed lesson.
Type `/help` to see the chat commands:

| Command | Purpose |
|---|---|
| `/teach question => answer` | Teach or correct one answer and save it |
| `/learning on` or `/learning off` | Turn parent answers and automatic teaching on or off |
| `/clear` | Clear conversation history while keeping memory |
| `/stats` | Show memory usage and parent status |
| `/quit` | Exit |

For example, `/teach What is my favorite color? => Blue` saves an answer you can recall
from either chat mode. Explicit teaching works while automatic learning is off.
`/learning on` requires a parent chat; start one with `./jayce`.

### What gets saved

`data/fineweb-edu/native/lessons.npz` stores native context vectors, next-byte labels,
an ending label, and a catalog of questions and answers. Both chat modes and parent teaching
use this file regardless of the teacher model or its precision. The question records
support lookup and memory restoration. Recall still traverses the prototypes; it does not
return an answer directly from the catalog.

Each chat lesson is checked for complete recall before being saved. Writers reload the
latest memory under a lock and preserve the parent's training progress. Each verified
lesson is saved automatically. Open sessions refresh Q&A memory when it changes.
Corpus text memory is saved alongside the lessons and is read by both run modes.
Jayce returns an answer only when every token and its ending have a strong match. Otherwise,
it says **Jayce don't know**.

Jayce uses only the current question, so ask self-contained questions. Rephrased questions may
not match saved prototypes. The parent also sees the conversation history. Answers are compared
as text, ignoring whitespace, to label a reply as confirmation or correction. Both
go through learning, including exact repeats. Interrupted or length-limited parent
answers are not learned.

New lesson memories share **4,096 slots across all byte labels** (about 8 MiB of
512-dimensional float32 prototype vectors). They additionally store up to 4,096 shared
384-dimensional question vectors (about 6 MiB), membership indices and practice counts.
Atomic updates and Python indices use additional working memory. Existing memories retain
their saved capacity. Compatible contexts share slots; unrelated contexts get new ones.
Parent feedback and `/teach` replace conflicting labels for the corrected question.

Longer answers need more slots. Shared chat lessons accept up to 2,048 UTF-8 bytes, within
the 4,096-byte question-and-answer context window. Qwen's generation budget remains
128 model tokens by default; it is separate from Jayce's byte recall budget. If a whole
answer cannot fit in memory, that lesson is rejected and the previous memory stays intact.

### Upgrading beta memories

Older `jayce-byte-lessons-v1` files are rebuilt in memory from their saved question/answer
records. Every record must replay correctly in both the old and new learner. Reading
does not change the file. The next teaching save preserves the original as
`lessons.v1-backup.npz` before atomically saving the new version, including pending
training progress. Rebuilding a large old memory can make its first load slower.
The rebuild uses one exemplar per latest saved question/answer; historical per-token
practice weights restart, while overall teaching counters and the old file are retained.
Raw-text checkpoints keep their existing format.

### Checking pattern transfer

```sh
.venv/bin/python benchmark/eval_patterns.py
```

This offline developer check trains separate temporary in-memory learners using the old
fingerprints and the adaptive features. It reports exact replay, untaught rewordings,
contrasting facts, unfamiliar questions, synonyms, and new math problems. It never opens
your saved memory or loads Qwen. It also reports prototype counts and NumPy array bytes;
those bytes exclude Python overhead and are not peak RAM measurements.

The small fixture is used during development, so its results do not establish broad
generalization. Pass a separate JSON file with the same `train`/`test` structure as
`benchmark/patterns.json` for an independent evaluation. New math rules and distant
synonyms deliberately expose the current learner's limits.
Increase capacity when training to make room for more lessons. A strong match can still be wrong.

## Choose the parent model

Edit [jayce_config.py](jayce_config.py) and restart:

```python
PARENT_MODEL = (
    "hf://bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/"
    "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)
```

Remote GGUF references use `hf://owner/repo/file.gguf`; Jayce downloads just that file and
caches it. Hugging Face Transformers model IDs and local GGUF paths also work.

Both parent backends are installed by `./jayce` using the optional `parent` extra. The model must be supported by Transformers
or llama.cpp. For Transformers only, `JAYCE_DTYPE=float16` selects
half precision; otherwise it uses float32. GGUF precision is determined by the downloaded file.
Shared native lessons work across teacher models and precision settings.

## Run the checks

```sh
# Native tests require only NumPy and PyArrow:
uv run --extra corpus python -m unittest discover -s tests -p 'test_jayce_native.py' -v
# The complete suite covers both optional dependencies:
uv run --extra parent --extra corpus python -m unittest discover -s tests -v
./benchmark/benchmark-java smoke
```

The Python checks cover learning, corrections, saved memories, and the chat loop. Set
`JAYCE_TEST_MODEL=default` to include real-model checks with the configured parent, including
skateboard/bicycle answers, learning, and recall after saving. You can also set it to another
model ID or GGUF reference for backend checks. The Java checks verify
backpropagation gradients and basic learner behavior; they need JDK 17 or newer.

Dependencies live in [pyproject.toml](pyproject.toml); [uv.lock](uv.lock) fixes their versions.

## License

Copyright (C) 2026 Loophole, LLC. Licensed under [AGPL-3.0-only](LICENSE.md), without warranty.
See [NOTICE.md](NOTICE.md) for coverage. Downloaded models, datasets, and dependencies keep their
own licenses.
