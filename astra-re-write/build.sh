#!/bin/sh
# Build only the paper; this script runs no experiments.
set -eu
cd "$(dirname "$0")"
mkdir -p build
tectonic --only-cached --keep-logs --keep-intermediates --outdir build main-astra-rewrite.tex
cp build/main-astra-rewrite.pdf main-astra-rewrite.pdf
