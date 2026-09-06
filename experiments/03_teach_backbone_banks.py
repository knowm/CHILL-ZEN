"""Teach the backbone banks: G1 fills, R1 repairs.

Paper Sec. III B and Sec. IV A. Both banks read the tuple
[label x6 | M backbone addresses] and write back symbols at all 1 + M
target addresses. G1 is soft and teaches on Bernoulli-masked codes with
the label always held; R1 is sharp and teaches on the corruption G1
itself produces. Selection is the best epoch's probe, not the last.

Reported per backbone code (books x symbols @ keep-p): G1's fill accuracy on held-out holes against the
majority prior it must beat, and R1's repair and preserve rates.
Expected at the deployed 128-book code: fill 0.3065 against a 0.1358
prior (lift +0.1708), pool wrong fraction 0.542, repair 0.2648.

Cost: about 9 minutes per bank at 128 books. Requires 01.

    python experiments/03_teach_backbone_banks.py                # deployed
    python experiments/03_teach_backbone_banks.py --all          # 12 codes
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config, teach                     # noqa: E402
from chill_zen.data import N_TRAIN, load_fashion                   # noqa: E402
from chill_zen.levels import BackboneLevel, backbone_cfg           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--points", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("03_teach_backbone_banks")

    points = args.points or (config.BACKBONE_POINTS if args.all
                             else [config.BACKBONE])

    _, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN], y[N_TRAIN:]
    codes_store = torch.load(artifacts.path("backbone_codes"))

    bpath = artifacts.path("backbone_banks")
    store = torch.load(bpath) if bpath.exists() else {}
    results = {}
    for tag in points:
        if tag not in codes_store:
            log(f"[teach] {tag} has no codes yet — run 01 first; skipped")
            continue
        cs = codes_store[tag]
        level = BackboneLevel(tag, cs["tr"], cs["ev"], y_tr, y_ev)
        cfg = backbone_cfg(tag, level.M, level.NCH)
        if tag in store and store[tag]["cfg"] == cfg:
            results[tag] = store[tag]["res"]
            log(f"[teach] {tag} cached  G fill "
                f"{results[tag]['g']['acc']:.4f}  R repair "
                f"{results[tag]['r']['rep']:.4f}")
            continue
        log(f"[teach] {tag}: {level.LANES} lanes x {level.KSP} spaces x "
            f"{level.NCH} ch")
        g_sd, r_sd, res = teach.teach_backbone(level, y_tr, y_ev, log)
        store[tag] = dict(cfg=cfg, g=g_sd, r=r_sd, res=res)
        torch.save(store, bpath)
        results[tag] = res

    old = artifacts.path("backbone_teach_results")
    merged = torch.load(old)["results"] if old.exists() else {}
    merged.update(results)
    torch.save(dict(points=list(merged), results=merged), old)

    log(f"\n{'point':>13}{'fill':>8}{'lift':>8}{'repair':>8}{'pres-min':>9}")
    for tag, r in merged.items():
        log(f"{tag:>13}{r['g']['acc']:>8.4f}"
            f"{r['g']['acc'] - r['g']['prior']:>+8.4f}"
            f"{r['r']['rep']:>8.4f}{r['r']['pres'][2]:>9.4f}")
    fh.close()


if __name__ == "__main__":
    main()
