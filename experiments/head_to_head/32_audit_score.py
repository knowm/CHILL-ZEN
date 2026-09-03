"""Score the audit arms with their FID.

Globs `gen/audit-*.npy` — the Bernoulli calibration baselines
(`30_bernoulli_baseline.py`) and the free-label draws
(`31_free_draw.py`). Their code, their shipped 60k reference,
threshold 0.1, n = 5120, exactly as `07_comparator_score.py`; a
separate glob and a separate results file so no audit row can be
mistaken for a system arm by the figure or table scripts.

    .venv-dtm/bin/python experiments/head_to_head/32_audit_score.py
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

GEN = ev.HERE / "gen"


def main():
    log, fh = ev.make_log("32_audit_score")
    log(f"\n=== scoring the audit arms "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")
    mu_r, sig_r = ev.ref_stats()
    arms = sorted(p.stem for p in GEN.glob("audit-*.npy")
                  if not p.stem.endswith("-lab"))
    log(f"arms: {arms}")
    log(f"{'arm':<32}{'FID':>9}{'on-frac':>9}")
    res = {}
    for name in arms:
        imgs = np.load(GEN / f"{name}.npy")
        b = (imgs > ev.THEIR_THRESH).astype(np.float32)
        act = ev.their_features(b, tag=name)
        mu, sig = ev.stats_of(act)
        fid, _, _ = ev.frechet(mu, sig, mu_r, sig_r)
        res[name] = dict(fid=fid, on=float(b.mean()))
        log(f"{name:<32}{fid:>9.2f}{res[name]['on']:>9.4f}")

    with open(ev.HERE / "audit-score-results.json", "w") as fh2:
        json.dump(res, fh2, indent=1)
    log("\nsaved: audit-score-results.json")
    fh.close()


if __name__ == "__main__":
    main()
