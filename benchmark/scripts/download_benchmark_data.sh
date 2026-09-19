#!/usr/bin/env bash
# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"

checksum() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

download() {
  local url="$1" destination="$2" expected="$3" temporary="${2}.download"
  mkdir -p "$(dirname -- "$destination")"
  if [[ ! -f "$destination" ]]; then
    printf '  downloading %s\n' "$url"
    curl --fail --location --retry 3 --silent --show-error --output "$temporary" "$url"
    mv "$temporary" "$destination"
  fi
  local actual
  actual="$(checksum "$destination")"
  if [[ "$actual" != "$expected" ]]; then
    rm -f "$destination"
    printf 'Checksum mismatch for %s\n' "$destination" >&2
    exit 1
  fi
}

mnist="$ROOT/data/mnist/raw"
fashion="$ROOT/data/fashion-mnist/raw"
base_mnist='https://storage.googleapis.com/cvdf-datasets/mnist'
base_fashion='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com'

download "$base_mnist/train-images-idx3-ubyte.gz" "$mnist/train-images-idx3-ubyte.gz" '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'
download "$base_mnist/train-labels-idx1-ubyte.gz" "$mnist/train-labels-idx1-ubyte.gz" '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'
download "$base_mnist/t10k-images-idx3-ubyte.gz" "$mnist/t10k-images-idx3-ubyte.gz" '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6'
download "$base_mnist/t10k-labels-idx1-ubyte.gz" "$mnist/t10k-labels-idx1-ubyte.gz" 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'
download "$base_fashion/train-images-idx3-ubyte.gz" "$fashion/train-images-idx3-ubyte.gz" '3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84'
download "$base_fashion/train-labels-idx1-ubyte.gz" "$fashion/train-labels-idx1-ubyte.gz" 'a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845'
download "$base_fashion/t10k-images-idx3-ubyte.gz" "$fashion/t10k-images-idx3-ubyte.gz" '346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073'
download "$base_fashion/t10k-labels-idx1-ubyte.gz" "$fashion/t10k-labels-idx1-ubyte.gz" '67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5'

printf 'All MNIST and Fashion-MNIST archives are present and checksum-verified.\n'
