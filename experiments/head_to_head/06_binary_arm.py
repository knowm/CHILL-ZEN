"""Score the binary-trained arms with their FID.

Their code, their shipped 60k reference, their n = 5120, threshold 0.1
-- which is the identity on these renders: the binarization already
happened at fit time, at 0.1, and the decode rounds at the midpoint,
so every saved image is exactly 0 or 1. Reads what
`experiments/19_binary_verdict.py` rendered into `gen/binary-*.npy`.

One check: the binary codec ceiling must land at or below the
grayscale codec ceiling's 14.21 (Table VII) -- the codec fits
0.1-binarized data better than midpoint data, so a ceiling above
14.21 means the render or threshold path is wrong, not the model.

Expected, against their reference at n = 5120 (real floor 1.91,
grayscale two-level 18.93, their 8-step DTM 24.90):

    arm                          FID
    binary-codec-ceiling-128    9.57
    binary-stack-ceiling        2.83
    binary-two-level-T0.3       9.82   (T0.5 9.93, T0.2 10.88, T0.1 16.81)
    binary-one-level-128-T0.2  12.47
    binary-one-level-64-T0.2   13.62
    binary-real-train           1.91

Cost: about an hour on first run; features are cached, so re-scoring
is free.

    .venv-dtm/bin/python experiments/head_to_head/06_binary_arm.py
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

GEN = ev.HERE / "gen"
CEILING_CHECK = 14.21
GRAY_TWO_LEVEL = 18.93
DTM_BAR = 24.90


def diversity(b, per=512):
    """Within-class mean pairwise pixel disagreement, 512 pairs a class."""
    rng = np.random.default_rng(0)
    out = []
    for c in range(len(b) // per):
        blk = b[c * per:(c + 1) * per].reshape(per, -1)
        i, j = rng.integers(0, per, 512), rng.integers(0, per, 512)
        m = i != j
        out.append(np.abs(blk[i[m]] - blk[j[m]]).mean())
    return float(np.mean(out))


def main():
    log, fh = ev.make_log("06_binary_arm")
    log(f"\n=== scoring the binary arms "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")
    mu_r, sig_r = ev.ref_stats()
    arms = sorted(p.stem for p in GEN.glob("binary-*.npy")
                  if not p.stem.endswith("-lab"))
    log(f"arms: {arms}")
    log(f"{'arm':<28}{'FID':>9}{'on-frac':>9}{'div':>8}")
    res = {}
    for name in arms:
        imgs = np.load(GEN / f"{name}.npy")
        b = (imgs > ev.THEIR_THRESH).astype(np.float32)
        act = ev.their_features(b, tag=name)
        mu, sig = ev.stats_of(act)
        fid, _, _ = ev.frechet(mu, sig, mu_r, sig_r)
        res[name] = dict(fid=fid, on=float(b.mean()), div=diversity(b))
        log(f"{name:<28}{fid:>9.2f}{res[name]['on']:>9.4f}"
            f"{res[name]['div']:>8.4f}")

    with open(ev.HERE / "binary-score-results.json", "w") as fh2:
        json.dump(res, fh2, indent=1)
    log("\nsaved: binary-score-results.json")

    ceil = res.get("binary-codec-ceiling-128", {}).get("fid")
    if ceil is None:
        log("no ceiling arm found -- render with 19_binary_verdict.py first")
        sys.exit(1)
    ok = ceil <= CEILING_CHECK
    log(f"binary codec ceiling {ceil:.2f} <= {CEILING_CHECK}: "
        f"{'OK' if ok else 'MISMATCH -- render/threshold path suspect'}")
    two = {k: v["fid"] for k, v in res.items()
           if k.startswith("binary-two-level")}
    if two:
        best = min(two, key=two.get)
        log(f"best two-level arm: {best} at {two[best]:.2f} "
            f"(grayscale two-level {GRAY_TWO_LEVEL}, DTM bar {DTM_BAR})")
    if not ok:
        sys.exit(1)
    fh.close()


if __name__ == "__main__":
    main()
