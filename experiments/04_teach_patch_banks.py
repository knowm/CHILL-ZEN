"""Teach the patch banks: G2 fills the residual level, R2 repairs it.

Paper Sec. III B and Sec. IV A. The tuple is
[label x6 | 128 backbone addresses | 98 patch addresses]. The backbone
addresses are held conditioning: never dropped, never targets. Only the
98 patch addresses are targets, so the bank is 1568 lanes over 232
spaces.

G2's probe carries a second reading, the *gap probe*: the same fill
with the backbone conditioning itself corrupted at rate 0.25 from the
class marginals. It measures how much this level depends on getting a
good backbone -- which is exactly what an imperfect draw fails to
supply.

Expected: G2 fill 0.3959 against a 0.3121 prior (lift +0.0838), gap
probe 0.3562; pool wrong fraction 0.467; R2 repair 0.3469, preserve
0.5829.

Cost: about 20 minutes for both banks. Requires 01-03.

    python experiments/04_teach_patch_banks.py
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config, teach                     # noqa: E402
from chill_zen.data import SEED                                    # noqa: E402
from chill_zen.levels import load_patch_level                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("04_teach_patch_banks")

    pl = load_patch_level(args.backbone, args.patch)
    cfg = pl.cfg()
    bpath = artifacts.path("patch_banks")
    store = torch.load(bpath) if bpath.exists() else {}
    if store and store.get("cfg") != cfg:
        raise RuntimeError(f"{bpath.name} was taught under a different "
                           "configuration — delete it to reteach")

    if "g" in store:
        gbank = pl.make_bank(SEED + 100)
        gbank.load_state_dict(store["g"])
        gcurve = store.get("gcurve", [])
        g_res = store.get("g_res")
        if g_res is None:            # a downloaded file carries no probe
            acc, gap, rows, _ = teach.probe_patch_fill(pl, gbank)
            g_res = dict(acc=acc, gap=gap, rows=rows, prior=None)
        log(f"[patch] G2 cached  fill {g_res['acc']:.4f} "
            f"(gap {g_res['gap']:.4f})")
    else:
        g_sd, g_res, gcurve = teach.teach_patch_fill(pl, log)
        store.update(cfg=cfg, g=g_sd, g_res=g_res, gcurve=gcurve)
        torch.save(store, bpath)
        gbank = pl.make_bank(SEED + 100)
        gbank.load_state_dict(g_sd)

    if "r" in store:
        rcurve = store.get("rcurve", [])
        r_res = store.get("r_res")
        if r_res is None:
            rbank = pl.make_bank(SEED + 200)
            rbank.load_state_dict(store["r"])
            rep, pres, wrong = teach.probe_patch_repair(pl, gbank, rbank)
            r_res = dict(rep=rep, pres=pres, wrong=wrong)
        log(f"[patch] R2 cached  repair {r_res['rep']:.4f}")
    else:
        r_sd, r_res, rcurve = teach.teach_patch_repair(pl, gbank, log)
        store.update(r=r_sd, r_res=r_res, rcurve=rcurve)
        torch.save(store, bpath)

    torch.save(dict(cfg=cfg, g=g_res, r=r_res, gcurve=gcurve, rcurve=rcurve),
               artifacts.path("patch_teach_results"))
    prior = f"{g_res['prior']:.4f}" if g_res.get("prior") else "n/a"
    log(f"[patch] G2 fill {g_res['acc']:.4f} (prior {prior}, gap "
        f"{g_res['gap']:.4f})   R2 repair {r_res['rep']:.4f}")
    fh.close()


if __name__ == "__main__":
    main()
