# APM vs. backpropagation: saved results

These results compare APM with a neural network trained by backpropagation.
See the [main README](../../README.md#benchmark-results) for the tables and conclusions.

## Latest run

The full benchmark completed on **September 21, 2026** in **6 minutes 49 seconds** on an Apple A18 Pro
with 8 GiB RAM, macOS 27.0, and OpenJDK 26.0.1. It covered four streams, three comparisons, two
validation seeds, and ten separate test seeds.

All 720 validation trials and 240 test trials completed. The gradient check, dataset checksums,
frozen settings, and result checks passed.

- [summary.csv](jayce-vs-backprop/summary.csv): scores, memory estimates, and paired comparisons.
- [final.csv](jayce-vs-backprop/final.csv): all 240 individual test trials, including training time.
- [Run metadata](jayce-vs-backprop/run-metadata.json): machine details, command, checksums, and verification.
- [Execution log](jayce-vs-backprop/run.log): completed benchmark steps.

## Reading the numbers

**Accuracy** is the percentage of correct predictions, averaged across ten test seeds. The
standard deviation shows how much those scores vary between seeds.

**Memory** counts stored model values in KiB (1,024 bytes). APM stores prototypes; the network stores
weights and Adam's two saved values per weight. The estimates exclude Java object overhead and
temporary working memory. State is the amount used; capacity is the allowed maximum. With the
memory ceiling, APM's capacity is limited to the network's estimated training-state size.

**Counted work** estimates training operations. Equal counted work does not guarantee equal runtime.

**Training time** is the `seconds` column in `final.csv`. It sums the time spent updating the learner
across all twelve phases. It excludes preparing data, making predictions, and evaluating accuracy.

## Comparing the learners

The paired columns compare APM and backprop on matching test seeds:

- **Jayce minus backprop:** the accuracy gap in percentage points. Positive favors APM;
  negative favors backprop.
- **95% confidence interval:** the estimated uncertainty around the average paired gap.
- **Jayce ahead seeds:** how many of the ten seeds gave APM higher accuracy.

These columns appear on both learner rows and always describe APM minus backprop. The comparison
covers one network design and four streams. It does not include nearest-class-mean, kNN, LVQ,
or a network trained with replay.

## Reproduce the run

From the project folder, run `./benchmark/run-benchmark` to create a new results folder.

[validation.csv](jayce-vs-backprop/validation.csv) and [selected.csv](jayce-vs-backprop/selected.csv)
show how settings were chosen. [checkpoints.csv](jayce-vs-backprop/checkpoints.csv) records progress
during training. [freeze.sha256](jayce-vs-backprop/freeze.sha256) ties the chosen settings to the
source and dataset files. [source.zip](jayce-vs-backprop/source.zip) contains the Java source used
for this run under `reference/java/`. Its checksum is recorded in `freeze.sha256`.

See the [benchmark guide](../README.md) for commands and file descriptions.

---

Documentation copyright (C) 2026 Loophole, LLC. Licensed under
[AGPL-3.0-only](../../LICENSE.md). Downloaded datasets retain their own licenses.
