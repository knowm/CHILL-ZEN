"""Teach the joint repair bank: one sweep over both levels at once.

Paper Sec. III B (the joint bank; fig-teach.png is written but no longer in the paper). Two findings from the two-level runs
motivate it. A patch repair bank taught on clean real conditioning
slightly hurts generated backbones, and nothing in the stack reads the
bottom-up evidence -- the patch level knows things about the backbone
that no backbone-only bank can use. One bank whose targets are all 226
addresses lets the levels repair each other in a single cold parallel
sweep.

The pool is built to look like failure at run time: both levels
corrupted, and the patch level corrupted *under* the already-corrupted
backbone rather than under the truth.

Expected: eval pool wrong 0.528 (backbone 0.547 / patch 0.504); repair 0.2985, split 0.3139 on backbone
addresses against 0.2768 on patch addresses. The backbone number is the
bottom-up signal -- at a matched 64-book configuration the joint bank
repairs backbone addresses at 0.372 against 0.287 for the backbone-only
repair bank it replaces.

Cost: about 25 minutes (3616 lanes over 232 spaces). Requires 01-04.

    python experiments/05_teach_joint_bank.py
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
from chill_zen.levels import (BackboneLevel, JointLevel,           # noqa: E402
                         backbone_cfg, load_patch_level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("05_teach_joint_bank")

    pl = load_patch_level(args.backbone, args.patch)
    jl = JointLevel(pl)
    cfg = jl.cfg()

    # the two fill banks that build the pool
    gcodes = torch.load(artifacts.path("backbone_codes"))[args.backbone]
    bl = BackboneLevel(args.backbone, gcodes["tr"], gcodes["ev"],
                       pl.y_tr, pl.y_ev)
    bstore = torch.load(artifacts.path("backbone_banks"))
    assert bstore[args.backbone]["cfg"] == backbone_cfg(args.backbone, bl.M,
                                                        bl.NCH)
    g1 = bl.make_bank(SEED + 100)
    g1.load_state_dict(bstore[args.backbone]["g"])
    pstore = torch.load(artifacts.path("patch_banks"))
    assert pstore["cfg"] == pl.cfg()
    g2 = pl.make_bank(SEED + 100)
    g2.load_state_dict(pstore["g"])

    truth_tr = torch.cat([pl.g_tr, pl.p_tr], 1)
    truth_ev = torch.cat([pl.g_ev, pl.p_ev], 1)
    maj_g = torch.stack([torch.bincount(pl.g_tr[:, u], minlength=16).argmax()
                         for u in range(pl.NG)])
    maj = torch.cat([maj_g, pl.maj])

    ppath = artifacts.path("joint_pool")
    if ppath.exists() and torch.load(ppath)["cfg"] == cfg:
        pd = torch.load(ppath)
        pool_tr = (pd["g_tr"], pd["p_tr"])
        pool_ev = (pd["g_ev"], pd["p_ev"])
        log("[joint] pool cached")
    else:
        log("[joint] building the pool (fills at T = 0.3, both levels "
            "corrupted, patches under the corrupted backbone)")
        pool_tr = teach.joint_pool(
            jl, bl, g1, g2, pl.y_tr, pl.g_tr, pl.p_tr,
            torch.Generator().manual_seed(SEED + 500), log, "train")
        pool_ev = teach.joint_pool(
            jl, bl, g1, g2, pl.y_ev, pl.g_ev, pl.p_ev,
            torch.Generator().manual_seed(SEED + 550), log, "eval")
        torch.save(dict(cfg=cfg, g_tr=pool_tr[0], p_tr=pool_tr[1],
                        g_ev=pool_ev[0], p_ev=pool_ev[1]), ppath)

    bpath = artifacts.path("joint_bank")
    store = torch.load(bpath) if bpath.exists() else {}
    if store and store.get("cfg") != cfg:
        raise RuntimeError(f"{bpath.name} was taught under a different "
                           "configuration — delete it to reteach")
    if "bank" in store:
        log("[joint] bank cached")
        curve = store.get("curve", [])
        res = store.get("res")
        if res is None:          # a downloaded file carries no probe
            bank = jl.make_bank(SEED + 600)
            bank.load_state_dict(store["bank"])
            probe = teach.probe_joint(jl, pl.y_ev, truth_ev, pool_ev, maj)
            rep, reps, pres, _ = probe(bank)
            res = dict(rep=rep, reps=reps, pres=pres, wrong=probe.wrong)
    else:
        sd, res, curve = teach.teach_joint(
            jl, pl.y_tr, pl.y_ev, truth_tr, truth_ev, pool_tr, pool_ev,
            maj, log)
        torch.save(dict(cfg=cfg, bank=sd, res=res, curve=curve), bpath)
    torch.save(dict(cfg=cfg, **res, curve=curve),
               artifacts.path("joint_teach_results"))
    log(f"[joint] repair {res['rep']:.4f} (backbone {res['reps'][0]:.4f} / "
        f"patch {res['reps'][1]:.4f})")
    fh.close()


if __name__ == "__main__":
    main()
