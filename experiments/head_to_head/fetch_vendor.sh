#!/usr/bin/env bash
# Fetch the evaluation code and the data the head-to-head is scored with.
#
# Nothing here is redistributed by this repository. This script clones
# the DTM replication code base at the exact commit that was used,
# downloads the canonical Fashion-MNIST idx files, and verifies every
# file against the sha256 checksums below, which are the authority;
# README.md lists the same values so they can be read without running
# anything. If the two ever disagree, this script is right. The
# vendored tree is
# read-only from then on: no file in it is edited, and the harness
# imports it as a package.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENDOR="$HERE/vendor"
DATA="$ROOT/data"
COMMIT=7c22d19c218ab353a770f7c2b1504570b7cbe3ba
REPO=https://github.com/pschilliOrange/dtm-replication.git
MNIST=https://fashion-mnist.s3.eu-central-1.amazonaws.com

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

check() {   # check <file> <expected sha256>
  local got; got="$(sha "$1")"
  if [ "$got" != "$2" ]; then
    echo "CHECKSUM MISMATCH for $1" >&2
    echo "  expected $2" >&2
    echo "  got      $got" >&2
    exit 1
  fi
  echo "  ok  $(basename "$1")"
}

mkdir -p "$VENDOR" "$DATA"

if [ ! -d "$VENDOR/dtm-replication" ]; then
  echo "cloning the evaluation code at $COMMIT"
  git clone --quiet "$REPO" "$VENDOR/dtm-replication"
  git -C "$VENDOR/dtm-replication" checkout --quiet "$COMMIT"
fi
git -C "$VENDOR/dtm-replication" rev-parse HEAD > "$VENDOR/COMMIT"
echo "vendored at $(cat "$VENDOR/COMMIT")"

echo "verifying their shipped reference statistics"
check "$VENDOR/dtm-replication/thrmlDenoising/fid/precomputed_stats/bw_fashion_mnist_train.npz" \
      66003004dc99115b20c146bd3c2a7d9d85fb85a3c0c9e991f11951933f97c5d8

echo "fetching canonical Fashion-MNIST"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  [ -f "$DATA/$f" ] || curl -fsSL "$MNIST/$f" -o "$DATA/$f"
done
check "$DATA/train-images-idx3-ubyte.gz" \
      3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84
check "$DATA/train-labels-idx1-ubyte.gz" \
      a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845
check "$DATA/t10k-images-idx3-ubyte.gz" \
      346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073
check "$DATA/t10k-labels-idx1-ubyte.gz" \
      67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5

echo
echo "done. The Inception weights are downloaded by their own code on"
echo "first use, into \$TMPDIR/jax_fid/; expected sha256"
echo "  4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77"
