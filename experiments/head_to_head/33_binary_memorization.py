"""Memorization battery on the binary renders the paper scores.

E3b (`21_reporting.py`) ran distinct-code counts and
nearest-train-neighbour distances on the grayscale system. The binary
arms never got that battery, and it matters more there: a hard
threshold collapses distinct codes onto identical renders, so
code-distinctness is not evidence, and binary images can collide or
copy in ways grayscale ones cannot.

For each binary arm the paper reports (`comparator-two-level-P6`, Table
VII's binary-trained two-level row; `comparator-binary-one-level-{128,64}`):

* distinct renders among the 5,120, and the size of the largest
  duplicate group;
* exact copies of 0.1-binarized train images;
* the nearest-train-neighbour Hamming-distance distribution, against
  the same distribution for real images excluded from fitting (the 0.1-binarized
  2,000 images excluded from fitting; these were used for model selection) as the calibration.

Pure analysis on saved renders; no draw, no lanes.

Cost: a few minutes.

    .venv-dtm/bin/python experiments/head_to_head/33_binary_memorization.py
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
THRESHOLD = 0.1
N_TRAIN = 68000
SPLIT_SEED = 0


def load_split(root):
    """Reconstruct load_fashion(seed=0)'s 68k/2k split from canonical IDX."""
    images = []
    for name in ("train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"):
        with gzip.open(root / "data" / name, "rb") as fh:
            raw = fh.read()
        header = np.frombuffer(raw[:16], dtype=">u4")
        assert header[0] == 2051 and tuple(header[2:]) == (28, 28)
        images.append(np.frombuffer(raw[16:], np.uint8).reshape(int(header[1]), 784))
    x = np.concatenate(images)
    assert len(x) == 70000
    order = np.random.default_rng(SPLIT_SEED).permutation(len(x))
    binary = x.astype(np.float32) / 255.0 > THRESHOLD
    return binary[order[:N_TRAIN]], binary[order[N_TRAIN:]]


ARMS = ["comparator-two-level-P6", "comparator-binary-one-level-128",
        "comparator-binary-one-level-64"]
QS = [0, 1, 5, 25, 50, 75, 100]


def nn_hamming(q, t, chunk=1024):
    """Min Hamming distance from each row of q to any row of t."""
    t = t.astype(np.float32)
    t_sum = t.sum(1)
    out = np.empty(len(q), np.float32)
    for lo in range(0, len(q), chunk):
        blk = q[lo:lo + chunk].astype(np.float32)
        ham = blk.sum(1)[:, None] + t_sum[None, :] - 2.0 * blk @ t.T
        out[lo:lo + chunk] = ham.min(1)
    return out


def battery(b, train, log):
    keys = {}
    for i, row in enumerate(np.packbits(b.astype(np.uint8), axis=1)):
        keys.setdefault(row.tobytes(), []).append(i)
    sizes = sorted((len(v) for v in keys.values()), reverse=True)
    t0 = time.time()
    d = nn_hamming(b, train)
    q = np.percentile(d, QS)
    log(f"  distinct renders {len(keys)}/{len(b)}  largest duplicate "
        f"group {sizes[0]}")
    log(f"  exact train copies {(d == 0).sum()}  ({time.time() - t0:.0f}s)")
    log("  NN-to-train Hamming: "
        + "  ".join(f"q{p:02d} {v:.0f}" for p, v in zip(QS, q)))
    return dict(distinct=len(keys), n=len(b), largest_dup=int(sizes[0]),
                exact_copies=int((d == 0).sum()),
                nn_quantiles={f"q{p:02d}": float(v) for p, v in zip(QS, q)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--output", type=pathlib.Path,
                    default=ROOT / "experiments/head_to_head/audit-memorization.json")
    args = ap.parse_args()
    log = lambda message: print(message, flush=True)
    train, test = load_split(args.root)
    log(f"training reference n={len(train)}; excluded-from-fit reference n={len(test)}")
    log("The 2,000 reference images were used for model selection, not weight fitting.")
    res = {"split": {"seed": SPLIT_SEED, "n_train": len(train),
                      "n_reference": len(test), "threshold": THRESHOLD,
                      "reference_used_for_selection": True},
           "held-out-real": battery(test, train, log)}
    gen = args.root / "experiments/head_to_head/gen"

    for name in ARMS:
        p = gen / f"{name}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Required audit render missing: {p}")
        b = (np.load(p) > THRESHOLD).astype(np.float32)
        b = b.reshape(len(b), -1)
        log(f"\n{name} (n = {len(b)}):")
        res[name] = battery(b, train, log)
        res[name]["render_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()

    with open(args.output, "w") as fh2:
        json.dump(res, fh2, indent=1)
    log("\nsaved: audit-memorization.json")



if __name__ == "__main__":
    main()
