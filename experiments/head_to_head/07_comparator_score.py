"""Score every physical-operating-point arm with their FID.

One scorer for all three ports/experiments that render into
`gen/comparator-*.npy`, since the feature cache keys by tag and old
arms never recompute:

  * the binary comparator-noise sweep
    (`24_binary_comparator_sweep.py`): `comparator-two-level-P0` ..
    `-P8`, `-deployed`. Must reproduce the binary column of the
    Sec. IV C table in NUMBERS.md row for row.
  * the grayscale Table VII rows at the physical point
    (`25_comparator_headtohead.py`): `comparator-two-level`,
    `comparator-one-level-{16,32,64,128}`.
  * the binary one-level Table VII rows at the physical point
    (`27_binary_comparator_headtohead.py`): `comparator-binary-
    one-level-{128,64}`.

Their code, their shipped 60k reference, threshold 0.1, n = 5120, in
every case.

Expected for the deployed-T port (deployed T=0.3 9.82/9.88, grayscale
two-level 18.93 at the deployed point, DTM bar 24.90):

    point            FID
    deployed         9.88
    P0               22.65
    P1               21.67
    P2               16.39
    P3               12.99
    P4               11.23
    P5               12.39
    P6               11.10
    P7               11.57
    P8               12.27

Cost: about an hour on first run; features are cached, so re-scoring
is free.

    .venv-dtm/bin/python experiments/head_to_head/07_comparator_score.py
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

GEN = ev.HERE / "gen"


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
    log, fh = ev.make_log("07_comparator_score")
    log(f"\n=== scoring the comparator sweep "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")
    mu_r, sig_r = ev.ref_stats()
    arms = sorted(p.stem for p in GEN.glob("comparator-*.npy")
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
        log(f"{name:<28}{fid:>9.2f}{res[name]['on']:>9.4f}{res[name]['div']:>8.4f}")

    with open(ev.HERE / "comparator-score-results.json", "w") as fh2:
        json.dump(res, fh2, indent=1)
    log("\nsaved: comparator-score-results.json")

    phys = {k: v["fid"] for k, v in res.items()
            if not k.endswith("deployed")}
    if phys:
        best = min(phys, key=phys.get)
        log(f"best physical point: {best} at {phys[best]:.2f} "
            f"(DTM bar 24.90; grayscale two-level 18.73 at the physical "
            f"point, 18.93 at the deployed point)")
    fh.close()


if __name__ == "__main__":
    main()
