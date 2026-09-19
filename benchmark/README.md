# Java benchmark guide

This benchmark compares APM with a neural network trained by backpropagation.

## Run it

Requires Java 17 or newer. From the project folder:

```sh
./benchmark/run-benchmark
```

The command downloads missing datasets, checks the learners, chooses settings, and runs the tests.
Use a new output folder for each run. To name it yourself:

```sh
./benchmark/run-benchmark target/my-run
```

Relative output paths are resolved inside `benchmark/`. The example above writes to
`benchmark/target/my-run/`.

The [saved run](results/README.md) took **6 minutes 49 seconds**. On macOS, prevent idle sleep
while on AC power with:

```sh
caffeinate -is ./benchmark/run-benchmark
```

To run the steps separately, start from the project folder:

```sh
cd benchmark
./scripts/download_benchmark_data.sh
./benchmark-java smoke
./benchmark-java tune target/manual-run
./benchmark-java test target/manual-run
```

The `smoke` step checks the backpropagation calculations and basic learner behavior. The `tune`
step chooses settings; `test` evaluates them on separate seeds. Run one benchmark at a time.

## How the comparison works

Four streams cover rotated MNIST digits, rotated Fashion-MNIST clothing, and two sets of generated
shapes. Each stream has twelve phases with changing conditions. Both learners receive labeled
examples without replaying old training data.

Settings are chosen on two validation seeds and evaluated on ten separate test seeds. Seeds make
the random choices repeatable. Each learner, stream, and comparison gets its own settings:

- **Same examples:** equal numbers of training examples.
- **Same counted work:** similar estimated computation, rather than measured runtime.
- **Same memory ceiling:** equal examples, with APM's storage limited to the network's estimated
  training-state size, including its weights and Adam's saved values.

Memory estimates exclude Java object overhead and temporary working memory.
See the [main README](../README.md#benchmark-results) for results and conclusions.

## Output files

| File | Contents |
|---|---|
| `validation.csv` | Settings tried and their validation scores |
| `selected.csv` | Settings chosen for each comparison |
| `freeze.sha256` | Checksums tying the chosen settings to the Java source and datasets |
| `source.zip` | Exact Java source used for the run |
| `checkpoints.csv` | Accuracy measured during each stream |
| `final.csv` | Accuracy, model state, counted work, and training time for each test trial |
| `summary.csv` | Averages and paired comparisons across test seeds |

Before testing, the benchmark verifies that the source, archive, settings, and datasets match the
frozen checksums. The [saved results](results/README.md) also include a run log and machine details.

## Read the code

- [JayceMemory.java](reference/java/company/loophole/jayce/apm/JayceMemory.java): APM's prototypes and updates.
- [SoftmaxBackpropNetwork.java](reference/java/company/loophole/jayce/backprop/SoftmaxBackpropNetwork.java):
  the network, backpropagation, and Adam updates.
- [BackpropGradientCheck.java](reference/java/company/loophole/jayce/backprop/BackpropGradientCheck.java):
  a numerical check of the backpropagation calculations.
- [HardBenchmark.java](reference/java/company/loophole/jayce/intervention/HardBenchmark.java):
  tuning, testing, and result summaries.

---

Documentation copyright (C) 2026 Loophole, LLC. Licensed under
[AGPL-3.0-only](../LICENSE.md). Downloaded datasets retain their own licenses.
