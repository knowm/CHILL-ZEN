"""Teach the binary banks with the unchanged lane recipe.

The binary-trained arms of Sec. V C. `chill_zen.teach` unchanged, on
the binary codes from experiment 17: backbone G1/R1 at 128x16@p0.5 and
64x16@p0.5, then the patch level (G2/R2) and the joint bank on the 128
stack.

Two checks run before anything is taught, because a predecessor run
was invalidated by a soft-teach that silently applied hard feedback:

* `feedback_semantics_check` -- the fast LaneBank must equal the
  oracle Classifier for BOTH feedback kinds, and soft must not equal
  hard. A failure refuses to teach.
* the literal feedback kind at every `teach.py` adapt call site is
  asserted from the source and logged (soft at G sites, hard at
  R/joint sites).

G fill lift is a proceed/stop check only, never a quality measure:
that invalidated run scored a HIGHER lift while being worse at
everything that mattered.

Expected (config-keyed and cached):

    128x16@p0.5   G lift +0.2213   R repair 0.2926
    64x16@p0.5    G lift +0.2560   R repair 0.3344
    patch         G2 lift +0.1712  R2 repair 0.4274
    joint         repair 0.3499

Cost: hours on first teach; seconds when the banks exist.

    python experiments/18_teach_binary_banks.py
"""
import inspect
import os
import pathlib
import re
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ktram_neural_core.torch import Classifier                     # noqa: E402
from chill_zen import artifacts, teach                             # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.lanes import LaneBank                               # noqa: E402
from chill_zen.levels import (BackboneLevel, JointLevel,           # noqa: E402
                              PatchLevel, backbone_cfg)

BACKBONES = ["128x16@p0.5", "64x16@p0.5"]
STACK_BACKBONE = "128x16@p0.5"
PATCH_TAG = "2x16@p0.5"


def feedback_semantics_check():
    """LaneBank == oracle for both rules; soft != hard. Raises on fail."""
    K, L, Sv = 20, 16, 4
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-1, Sv, (64, K), generator=g)
    tgt = torch.randint(0, L, (64, 1), generator=g)
    ga = {}
    for fb in ("hard", "soft"):
        c = Classifier(L, K, Sv, init="medium", seed=5)
        f = LaneBank(L, K, Sv, init="medium", seed=5)
        for _ in range(3):
            c.adapt(a, tgt, feedback=fb)
            f.adapt(a, tgt, feedback=fb)
        if not (torch.equal(c.ga, f.ga) and torch.equal(c.gb, f.gb)):
            raise RuntimeError(f"LaneBank '{fb}' != oracle Classifier")
        ga[fb] = f.ga.clone()
    if torch.equal(ga["hard"], ga["soft"]):
        raise RuntimeError("LaneBank soft == hard -- the soft-feedback bug")
    return True


def assert_call_sites(log):
    """The literal feedback kind at every adapt call in teach.py."""
    src = inspect.getsource(teach)
    sites = [(m.start(), m.group(1))
             for m in re.finditer(r'feedback="(soft|hard)"', src)]
    kinds = [k for _, k in sites]
    assert kinds == ["soft", "hard", "soft", "hard", "hard"], \
        f"teach.py call sites changed: {kinds}"
    for i, (pos, k) in enumerate(sites):
        line = src[:pos].count("\n") + 1
        log(f"  [check] teach.py adapt site {i + 1} at source line {line}: "
            f'feedback="{k}"')


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("18_teach_binary_banks")
    log(f"binary bank teaching ({time.strftime('%Y-%m-%d %H:%M')})")
    feedback_semantics_check()
    log("[check] feedback_semantics_check PASS: LaneBank == oracle for both "
        "kinds, soft != hard")
    assert_call_sites(log)

    _, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    codes_store = torch.load(artifacts.path("binary_backbone_codes"))

    bpath = artifacts.path("binary_backbone_banks")
    store = torch.load(bpath) if bpath.exists() else {}
    stop = False
    for tag in BACKBONES:
        cs = codes_store[tag]
        level = BackboneLevel(tag, cs["tr"].long(), cs["ev"].long(),
                              y_tr, y_ev)
        cfg = dict(backbone_cfg(tag, level.M, level.NCH),
                   binarize=cs["cfg"]["binarize"])
        if tag in store and store[tag]["cfg"] == cfg:
            r = store[tag]["res"]
            log(f"[teach] {tag} cached  G fill {r['g']['acc']:.4f}  "
                f"R repair {r['r']['rep']:.4f}")
        else:
            log(f"[teach] {tag}: {level.LANES} lanes x {level.KSP} spaces")
            g_sd, r_sd, res = teach.teach_backbone(level, y_tr, y_ev, log)
            store[tag] = dict(cfg=cfg, g=g_sd, r=r_sd, res=res)
            torch.save(store, bpath)
            r = res
        lift = r["g"]["acc"] - r["g"]["prior"]
        ok = lift > 0 and r["r"]["rep"] > 0
        stop |= not ok
        log(f"[check] {tag}: G lift {lift:+.4f} (>0), R repair "
            f"{r['r']['rep']:.4f} (>0)  {'OK' if ok else 'STOP'}  "
            f"(lift is proceed/stop only, not quality)")
    if stop:
        log("backbone teaching failed its proceed checks -- stopping.")
        fh.close()
        sys.exit(1)

    # ---- the two-level stack on the 128 backbone ----
    cs = codes_store[STACK_BACKBONE]
    kd = torch.load(artifacts.path("binary_patch_codes"))[PATCH_TAG]
    pl = PatchLevel(STACK_BACKBONE, PATCH_TAG, y_tr, y_ev,
                    cs["tr"].long(), cs["ev"].long(),
                    kd["tr"].long().reshape(len(y_tr), -1),
                    kd["ev"].long().reshape(len(y_ev), -1))
    ppath = artifacts.path("binary_patch_banks")
    pcfg = dict(pl.cfg(), binarize=cs["cfg"]["binarize"])
    pstore = torch.load(ppath) if ppath.exists() else None
    if pstore is not None and pstore["cfg"] == pcfg:
        log("[teach] patch banks cached")
        g2_sd, r2_sd = pstore["g"], pstore["r"]
        g_res, r_res = pstore["g_res"], pstore["r_res"]
    else:
        g2_sd, g_res, _ = teach.teach_patch_fill(pl, log)
        g2 = pl.make_bank(SEED + 100)
        g2.load_state_dict(g2_sd)
        r2_sd, r_res, _ = teach.teach_patch_repair(pl, g2, log)
        torch.save(dict(cfg=pcfg, g=g2_sd, r=r2_sd, g_res=g_res,
                        r_res=r_res), ppath)
    lift2 = g_res["acc"] - g_res["prior"]
    ok2 = lift2 > 0 and r_res["rep"] > 0
    log(f"[check] patch: G2 lift {lift2:+.4f} (>0), R2 repair "
        f"{r_res['rep']:.4f} (>0)  {'OK' if ok2 else 'STOP'}")
    if not ok2:
        log("patch teaching failed its proceed checks -- stopping.")
        fh.close()
        sys.exit(1)

    # ---- the joint bank ----
    jl = JointLevel(pl)
    jpath = artifacts.path("binary_joint_bank")
    jcfg = dict(jl.cfg(), binarize=cs["cfg"]["binarize"])
    jstore = torch.load(jpath) if jpath.exists() else None
    if jstore is not None and jstore["cfg"] == jcfg:
        log("[teach] joint bank cached")
    else:
        bl = BackboneLevel(STACK_BACKBONE, cs["tr"].long(), cs["ev"].long(),
                           y_tr, y_ev)
        g1 = bl.make_bank(SEED + 100)
        g1.load_state_dict(store[STACK_BACKBONE]["g"])
        g2 = pl.make_bank(SEED + 100)
        g2.load_state_dict(g2_sd)
        truth_tr = torch.cat([pl.g_tr, pl.p_tr], 1)
        truth_ev = torch.cat([pl.g_ev, pl.p_ev], 1)
        maj = torch.cat([bl.maj[1:], pl.maj])
        pool_tr = teach.joint_pool(jl, bl, g1, g2, y_tr, pl.g_tr, pl.p_tr,
                                   torch.Generator().manual_seed(SEED + 500),
                                   log, "tr")
        pool_ev = teach.joint_pool(jl, bl, g1, g2, y_ev, pl.g_ev, pl.p_ev,
                                   torch.Generator().manual_seed(SEED + 550),
                                   log, "ev")
        j_sd, j_res, _ = teach.teach_joint(jl, y_tr, y_ev, truth_tr,
                                           truth_ev, pool_tr, pool_ev, maj,
                                           log)
        torch.save(dict(cfg=jcfg, bank=j_sd, res=j_res), jpath)
        log(f"[check] joint: repair {j_res['rep']:.4f} (>0)  "
            f"{'OK' if j_res['rep'] > 0 else 'STOP'}")
    log("binary bank teaching done")
    fh.close()


if __name__ == "__main__":
    main()
