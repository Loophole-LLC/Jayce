# Fashion-MNIST data

Fashion-MNIST is a set of clothing pictures, each labeled with one of ten clothing types. The
downloaded files contain 60,000 training images and 10,000 test images. Each image is 28×28 gray
pixels; the Java benchmark shrinks each one to 14×14 pixels before learning.

Source: [Fashion-MNIST S3 archive](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)

SHA-256 checksums:

```text
3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84  train-images-idx3-ubyte.gz
a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845  train-labels-idx1-ubyte.gz
346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073  t10k-images-idx3-ubyte.gz
67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5  t10k-labels-idx1-ubyte.gz
```

The downloaded files are ignored by Git. From the project folder, run
`./benchmark/scripts/download_benchmark_data.sh` to download and check both datasets.
`./benchmark/run-benchmark` does this automatically. To check these Fashion-MNIST files by
themselves, run `shasum -a 256 raw/*.gz` in this folder and compare the results above.

---

Documentation copyright (C) 2026 Loophole, LLC. Licensed under
[AGPL-3.0-only](../../../LICENSE.md). Downloaded datasets retain their own licenses.
