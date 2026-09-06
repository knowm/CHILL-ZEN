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
  the same distribution for held-out real images (the 0.1-binarized
  canonical test split) as the calibration.

Pure analysis on saved renders; no draw, no lanes.

Cost: a few minutes.

    .venv-dtm/bin/python experiments/head_to_head/33_binary_memorization.py
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

GEN = ev.HERE / "gen"
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
    log, fh = ev.make_log("33_binary_memorization")
    log(f"\n=== binary memorization battery "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")
    x_tr, _ = ev.fashion("train")
    train = ev.binarize(x_tr, ev.THEIR_THRESH).reshape(len(x_tr), -1)
    x_te, _ = ev.fashion("test")
    test = ev.binarize(x_te, ev.THEIR_THRESH).reshape(len(x_te), -1)

    log(f"\nheld-out real calibration (test split, n = {len(test)}):")
    res = {"held-out-real": battery(test, train, log)}

    for name in ARMS:
        p = GEN / f"{name}.npy"
        if not p.exists():
            log(f"\n{name}: MISSING render, skipped")
            continue
        b = (np.load(p) > ev.THEIR_THRESH).astype(np.float32)
        b = b.reshape(len(b), -1)
        log(f"\n{name} (n = {len(b)}):")
        res[name] = battery(b, train, log)

    with open(ev.HERE / "audit-memorization.json", "w") as fh2:
        json.dump(res, fh2, indent=1)
    log("\nsaved: audit-memorization.json")
    fh.close()


if __name__ == "__main__":
    main()
