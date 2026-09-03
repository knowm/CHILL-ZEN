"""Fit the residual patch codec on top of the frozen backbone.

Paper Sec. III A and Sec. IV E (b). The backbone decode leaves a
residual; this level codes it with M books of 16 symbols per slot on a
7x7 grid of 4x4-pixel slots. The deployed point is 2 books per slot,
keep-p 0.5: 392 bits on top of the backbone's 512, for 904 in all.

Reported per point: the COMBINED floor (backbone decode plus patch
decode against the real image -- the number that matters), the residual
floor alone, within-slot cross-book coherence, and dead atoms.
Expected against the deployed 128-book backbone: the backbone alone
sits at 0.0612, the deployed 2x16@p0.5 point at 0.0382, and every
point in the sweep beats the 0.0576 of a per-patch code at 512 bits.

Cost: under a minute per point. Requires 01 to have fitted the
backbone.

    python experiments/02_fit_patch_codec.py                # full sweep
    python experiments/02_fit_patch_codec.py --points 2x16@p0.5
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, codec, config                    # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion, rel_mse   # noqa: E402

S = 16
P_SLOTS = 49


def cfg_of(M, p, backbone):
    return dict(seed=SEED, backbone=backbone, grid=f"{P_SLOTS}x{codec.PX}",
                m=M, s=S, p=p, iters=codec.ITERS, passes=codec.PASSES,
                km_iters=codec.KM_ITERS, ridge=codec.RIDGE, n_train=N_TRAIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--points", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("02_fit_patch_codec")

    if args.points:
        sweep = []
        for t in args.points:
            m, rest = t.split("x")
            sweep.append((int(m), float(rest.split("@p")[1])))
    else:
        sweep = config.PATCH_SWEEP

    X, y = load_fashion(seed=0)
    img = X[:, 0].clamp(0, 1).reshape(len(X), 784)

    gb = torch.load(artifacts.path("backbone_books"))[args.backbone]
    gcodes = torch.load(artifacts.path("backbone_codes"))[args.backbone]
    p_g = gb["cfg"]["p"]
    dec_tr = codec.decode_global(gcodes["tr"].long(), gb["bias"],
                                 gb["atoms"], p_g)
    dec_ev = codec.decode_global(gcodes["ev"].long(), gb["bias"],
                                 gb["atoms"], p_g)
    res_tr = codec.to_patches(img[:N_TRAIN] - dec_tr)
    res_ev = codec.to_patches(img[N_TRAIN:] - dec_ev)
    backbone_floor = rel_mse(dec_ev, img[N_TRAIN:])
    res_energy = ((img[N_TRAIN:] - dec_ev).var() / img[N_TRAIN:].var()).item()
    log(f"[patch codec] backbone {args.backbone}: eval floor "
        f"{backbone_floor:.4f}, residual energy {100 * res_energy:.1f}% of "
        f"image variance")

    bpath, cpath = artifacts.path("patch_books"), artifacts.path("patch_codes")
    rpath = artifacts.path("patch_codec_results")
    books = torch.load(bpath) if bpath.exists() else {}
    codes_store = torch.load(cpath) if cpath.exists() else {}
    res = torch.load(rpath)["res"] if rpath.exists() else {}

    for M, p in sweep:
        tag = f"{M}x{S}@p{p}"
        cfg = cfg_of(M, p, args.backbone)
        if tag not in books or books[tag]["cfg"] != cfg:
            log(f"[fit] {tag} ({P_SLOTS * M * 4} bits/image)")
            bias, W = codec.fit_slots(res_tr, M, S, p, log=log)
            books[tag] = dict(cfg=cfg, bias=bias, W=W)
            torch.save(books, bpath)
        if tag in res and res[tag].get("cfg") == cfg:
            continue
        bias, W = books[tag]["bias"], books[tag]["W"]
        codes_tr = codec.encode_slots(res_tr, bias, W, p)
        codes_ev = codec.encode_slots(res_ev, bias, W, p)
        render_ev = dec_ev + codec.from_patches(
            codec.decode_slots(codes_ev, bias, W, p))
        combined = rel_mse(render_ev, img[N_TRAIN:])
        res_floor = rel_mse(codec.decode_slots(codes_ev, bias, W, p), res_ev)
        coh, bar = codec.coherence_slots(codes_tr, codes_ev, M, S)
        dead = 0
        for m in range(M):
            for k in range(P_SLOTS):
                dead += (torch.bincount(codes_tr[:, k, m],
                                        minlength=S) == 0).sum().item()
        codes_store[tag] = dict(cfg=cfg, tr=codes_tr.to(torch.int8),
                                ev=codes_ev.to(torch.int8))
        torch.save(codes_store, cpath)
        res[tag] = dict(cfg=cfg, M=M, p=p, bits=P_SLOTS * M * 4,
                        combined=combined, res_floor=res_floor,
                        coh=coh, bar=bar, dead=dead)
        torch.save(dict(res=res, sweep=config.PATCH_SWEEP,
                        backbone_floor=backbone_floor), rpath)
        log(f"[{tag}] {P_SLOTS * M * 4} bits  combined {combined:.4f}  "
            f"residual {res_floor:.4f}  coh {coh:.3f} (bar {bar:.3f})  "
            f"dead {dead}/{P_SLOTS * M * S}")

    log(f"\n{'point':>11}{'bits':>6}{'combined':>10}{'residual':>10}"
        f"{'coh-bar':>9}{'dead':>6}")
    for tag, r in sorted(res.items(), key=lambda kv: (kv[1]["bits"],
                                                      kv[1]["p"])):
        cb = r["coh"] - r["bar"] if r["coh"] == r["coh"] else float("nan")
        log(f"{tag:>11}{r['bits']:>6}{r['combined']:>10.4f}"
            f"{r['res_floor']:>10.4f}{cb:>9.3f}{r['dead']:>6}")
    log(f"anchors: backbone alone {backbone_floor:.4f}  |  a per-patch code "
        f"at 512 bits {config.PER_PATCH_CODEC_FLOOR}")
    fh.close()


if __name__ == "__main__":
    main()
