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
    `comparator-one-level-{16,32,64,128}`;
  * the same arms at every level of the register grid
    (`34_grayscale_grid.py`): `comparator-gray-{arm}-{P0,P1,P2,P3,P4,P6}`,
    summarized at the end as each arm's best level.
  * the binary one-level Table VII rows at the physical point
    (`27_binary_comparator_headtohead.py`): `comparator-binary-
    one-level-{128,64}`.

Their code, their shipped 60k reference, threshold 0.1, n = 5120, in
every case.

Expected on the banks taught on the physical fill read (NUMBERS.md
Sec. IV C; deployed fixed-gain control 10.63 on this seed block,
grayscale two-level 21.95 at P4 and 19.43 at the deployed point, DTM
bar 24.90):

    point            FID
    deployed        10.63
    P0              22.14
    P1              20.53
    P2              14.86
    P3              10.99
    P4              10.04
    P5              10.73
    P6               9.69
    P7              11.57

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
        gray = res.get("comparator-two-level", {}).get("fid")
        spath = ev.HERE / "score-results.json"
        dep = (json.loads(spath.read_text()).get("two-level@0.1", {})
               .get("theirs") if spath.exists() else None)
        extra = ""
        if gray is not None:
            extra += f"; grayscale two-level {gray:.2f} at the physical point"
        if dep is not None:
            extra += f", {dep:.2f} at the deployed point"
        log(f"best physical point: {best} at {phys[best]:.2f} "
            f"(DTM bar 24.90{extra})")
    grid = grayscale_best(res)
    if grid:
        log("\ngrayscale register grid (34_grayscale_grid.py), best level per arm:")
        for arm, (P, fid, curve) in grid.items():
            log(f"  {arm:<14} {P:>3} {fid:>7.2f}   "
                + "  ".join(f"{q} {v:.2f}" for q, v in curve))
    fh.close()


GRAY_ARMS = ("one-level-16", "one-level-32", "one-level-64", "one-level-128",
             "two-level")


def grayscale_best(res):
    """{arm: (best point, best FID, [(point, FID), ...])} over the
    `comparator-gray-{arm}-{P}` renders present in `res`; empty if none."""
    out = {}
    for arm in GRAY_ARMS:
        pre = f"comparator-gray-{arm}-"
        curve = sorted((k[len(pre):], v["fid"]) for k, v in res.items()
                       if k.startswith(pre))
        if curve:
            P, fid = min(curve, key=lambda t: t[1])
            out[arm] = (P, fid, curve)
    return out


if __name__ == "__main__":
    main()
