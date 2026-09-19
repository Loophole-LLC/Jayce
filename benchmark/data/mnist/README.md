# MNIST data

MNIST is a set of handwritten digit pictures, labeled 0 through 9. The downloaded files contain
60,000 training images and 10,000 test images. Each image is 28×28 gray pixels; the Java benchmark
shrinks each one to 14×14 pixels before learning.

Source: [TensorFlow-hosted MNIST archive](https://storage.googleapis.com/cvdf-datasets/mnist/)

SHA-256 checksums:

```text
440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  train-images-idx3-ubyte.gz
3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c  train-labels-idx1-ubyte.gz
8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6  t10k-images-idx3-ubyte.gz
f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6  t10k-labels-idx1-ubyte.gz
```

The downloaded files are ignored by Git. From the project folder, run
`./benchmark/scripts/download_benchmark_data.sh` to download and check both datasets.
`./benchmark/run-benchmark` does this automatically. To check these MNIST files by themselves, run
`shasum -a 256 raw/*.gz` in this folder and compare the results above.

---

Documentation copyright (C) 2026 Loophole, LLC. Licensed under
[AGPL-3.0-only](../../../LICENSE.md). Downloaded datasets retain their own licenses.
