"""Fit the binary codec: dropout-AQ on 0.1-binarized Fashion-MNIST.

The binary-trained arms of Sec. V C and Table VII. The codec code is
`chill_zen.codec` unchanged; the only difference from experiments 01
and 02 is the data: images binarized at the protocol's source
threshold 0.1 before fitting. Three fits:

* backbone 128x16@p0.5 and 64x16@p0.5 (the one-level arms)
* the residual patch codec 2x16@p0.5 on the 128-backbone residual
  (the two-level stack)

Render convention: the source threshold is 0.1 (the recovered
protocol's); the decode of a {0,1}-target fit rounds at the midpoint
0.5, whatever threshold produced the targets. Renders saved as {0,1}
floats therefore pass the scoring script's 0.1 threshold unchanged.

Expected (eval side; the fit is config-keyed and cached):

    128x16@p0.5  disagree-ev 0.0169   relMSE-ev 0.0788   dead 0
    64x16@p0.5   disagree-ev 0.0266   relMSE-ev 0.1046   dead 0
    two-level    disagree-ev 0.0068   relMSE-ev 0.0464

A disagreement with the 128/64 rows beyond the third decimal means the
data convention is wrong, and the script says so and stops.

Cost: hours on first fit; seconds when the artifacts exist.

    python experiments/17_fit_binary_codec.py
"""
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, codec                             # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion, rel_mse    # noqa: E402

SRC_THRESH = 0.1        # the protocol's source threshold
RENDER_THRESH = 0.5     # decode rounds at the midpoint of {0, 1}
BACKBONES = [(128, 16, 0.5), (64, 16, 0.5)]
PATCH = (2, 0.5)        # books per slot, keep-p -- the deployed patch code
EXPECT = {"128x16@p0.5": (0.0169, 0.0788), "64x16@p0.5": (0.0266, 0.1046)}
TOL = 1e-3              # three decimals


def cfg_of(M, S, p):
    return dict(seed=SEED, m=M, s=S, p=p, iters=codec.ITERS,
                passes=codec.PASSES, km_iters=codec.KM_ITERS,
                ridge=codec.RIDGE, n_train=N_TRAIN, binarize=SRC_THRESH)


def disagree(rec, tgt):
    return ((rec > RENDER_THRESH) != (tgt > 0.5)).float().mean().item()


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("17_fit_binary_codec")
    log(f"binary codec fit ({time.strftime('%Y-%m-%d %H:%M')})")
    X, y = load_fashion(seed=0)
    img = (X[:, 0].clamp(0, 1).reshape(len(X), 784) > SRC_THRESH).float()
    x_tr, x_ev = img[:N_TRAIN], img[N_TRAIN:]
    log(f"[data] binarized at {SRC_THRESH}: on-fraction {img.mean():.4f} "
        f"(expected 0.4521)")

    bpath = artifacts.path("binary_backbone_books")
    cpath = artifacts.path("binary_backbone_codes")
    rpath = artifacts.path("binary_codec_results")
    books = torch.load(bpath) if bpath.exists() else {}
    codes_store = torch.load(cpath) if cpath.exists() else {}
    res = torch.load(rpath)["res"] if rpath.exists() else {}

    ok_all = True
    for M, S, p in BACKBONES:
        tag = f"{M}x{S}@p{p}"
        cfg = cfg_of(M, S, p)
        if tag not in books or books[tag]["cfg"] != cfg:
            log(f"[fit] {tag} ({codec.bits_of(M, S)} bits, binary at "
                f"{SRC_THRESH})")
            bias, atoms = codec.fit_global(x_tr, M, S, p, log=log)
            books[tag] = dict(cfg=cfg, bias=bias, atoms=atoms)
            torch.save(books, bpath)
        b = books[tag]
        if tag not in codes_store or codes_store[tag]["cfg"] != cfg:
            codes_tr = codec.encode_global(x_tr, b["bias"], b["atoms"], p)
            codes_ev = codec.encode_global(x_ev, b["bias"], b["atoms"], p)
            codes_store[tag] = dict(cfg=cfg, tr=codes_tr.to(torch.int8),
                                    ev=codes_ev.to(torch.int8))
            torch.save(codes_store, cpath)
        cs = codes_store[tag]
        rec_ev = codec.decode_global(cs["ev"].long(), b["bias"], b["atoms"], p)
        rec_tr = codec.decode_global(cs["tr"].long(), b["bias"], b["atoms"], p)
        row = dict(cfg=cfg, dis_ev=disagree(rec_ev, x_ev),
                   relmse_ev=rel_mse(rec_ev, x_ev),
                   dis_tr=disagree(rec_tr, x_tr),
                   relmse_tr=rel_mse(rec_tr, x_tr),
                   dead=sum((torch.bincount(cs["tr"][:, m].long(),
                                            minlength=S) == 0).sum()
                            for m in range(M)).item())
        res[tag] = row
        torch.save(dict(res=res), rpath)
        gd, gr = EXPECT[tag]
        ok = abs(row["dis_ev"] - gd) < TOL and abs(row["relmse_ev"] - gr) < TOL
        ok_all &= ok
        log(f"[{tag}] disagree-ev {row['dis_ev']:.4f} (expected {gd}), "
            f"relMSE-ev {row['relmse_ev']:.4f} (expected {gr}), "
            f"dead {row['dead']}  {'OK' if ok else 'MISMATCH'}")

    if not ok_all:
        log("binary codec MISMATCH beyond the third decimal -- the data "
            "convention is suspect; stopping before the patch fit.")
        fh.close()
        sys.exit(1)

    # The residual patch codec on the 128-backbone residual.
    tag = "128x16@p0.5"
    b = books[tag]
    Mp, pp = PATCH
    pcfg = dict(seed=SEED, backbone=tag, grid=f"49x{codec.PX}", m=Mp, s=16,
                p=pp, iters=codec.ITERS, passes=codec.PASSES,
                km_iters=codec.KM_ITERS, ridge=codec.RIDGE,
                n_train=N_TRAIN, binarize=SRC_THRESH)
    ppath = artifacts.path("binary_patch_books")
    kpath = artifacts.path("binary_patch_codes")
    pbooks = torch.load(ppath) if ppath.exists() else {}
    ptag = f"{Mp}x16@p{pp}"
    if ptag not in pbooks or pbooks[ptag]["cfg"] != pcfg:
        cs = codes_store[tag]
        dec_tr = codec.decode_global(cs["tr"].long(), b["bias"], b["atoms"],
                                     0.5)
        dec_ev = codec.decode_global(cs["ev"].long(), b["bias"], b["atoms"],
                                     0.5)
        res_tr = codec.to_patches(x_tr - dec_tr)
        log(f"[fit] patch {ptag} on the {tag} residual (binary)")
        pbias, W = codec.fit_slots(res_tr, Mp, 16, pp, log=log)
        pbooks[ptag] = dict(cfg=pcfg, bias=pbias, W=W)
        torch.save(pbooks, ppath)
        k_tr = codec.encode_slots(codec.to_patches(x_tr - dec_tr), pbias, W, pp)
        k_ev = codec.encode_slots(codec.to_patches(x_ev - dec_ev), pbias, W, pp)
        torch.save({ptag: dict(cfg=pcfg, tr=k_tr.to(torch.int8),
                               ev=k_ev.to(torch.int8))}, kpath)
    pb = pbooks[ptag]
    kd = torch.load(kpath)[ptag]
    cs = codes_store[tag]
    dec_ev = codec.decode_global(cs["ev"].long(), b["bias"], b["atoms"], 0.5)
    comb = dec_ev + codec.from_patches(
        codec.decode_slots(kd["ev"].long(), pb["bias"], pb["W"], pp))
    log(f"[two-level] combined floor: disagree-ev "
        f"{disagree(comb, x_ev):.4f}  relMSE-ev {rel_mse(comb, x_ev):.4f}  "
        f"(backbone alone {res[tag]['dis_ev']:.4f} / "
        f"{res[tag]['relmse_ev']:.4f})")
    log("binary codec fit done")
    fh.close()


if __name__ == "__main__":
    main()
