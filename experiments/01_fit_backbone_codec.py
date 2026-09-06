"""Fit the backbone codec: M books of 16 symbols over the whole image.

Paper Sec. III A and Sec. IV E (a, b). One dropout-AQ code over all 784
pixels -- no tiling, so no tile seams to remove later. The sweep spans
16 to 128 books at keep-p in {0.5, 0.75, 1.0} and a deeper 32-symbol
family; the deployed point is 128 books at keep-p 0.5, 512 bits.

Reported per point: the eval roundtrip floor, the cross-book coherence
over the majority bar (the redundancy the banks feed on), and dead
atoms. Expected at the deployed point: floor 0.0612, coherence 0.194
against a 0.137 bar, no dead atoms. At keep-p 0.75 the same 512 bits
reach 0.0500 -- the compression frontier -- and at keep-p 1.0 the code
degrades to 0.0902 with 56 dead atoms, which is the point of the
dropout objective.

Cost: about 12 minutes per 128-book point on eight CPU threads; the
whole sweep is a few hours. Artifacts are config-keyed, so re-running
skips what is already fitted.

    python experiments/01_fit_backbone_codec.py            # deployed point
    python experiments/01_fit_backbone_codec.py --all      # the full sweep
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


def cfg_of(M, S, p):
    return dict(seed=SEED, m=M, s=S, p=p, iters=codec.ITERS,
                passes=codec.PASSES, km_iters=codec.KM_ITERS,
                ridge=codec.RIDGE, n_train=N_TRAIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="fit the whole sweep rather than the deployed point alone")
    ap.add_argument("--points", nargs="*", default=None,
                    help="explicit codes as books x symbols @ keep-p, e.g. 128x16@p0.5")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("01_fit_backbone_codec")

    if args.points:
        from chill_zen.levels import parse_tag
        sweep = [parse_tag(t) for t in args.points]
    elif args.all:
        sweep = config.BACKBONE_SWEEP
    else:
        from chill_zen.levels import parse_tag
        sweep = [parse_tag(config.BACKBONE)]

    X, y = load_fashion(seed=0)
    D = 784
    img = X[:, 0].clamp(0, 1).reshape(len(X), D)
    x_tr, x_ev = img[:N_TRAIN], img[N_TRAIN:]

    bpath, cpath = artifacts.path("backbone_books"), \
        artifacts.path("backbone_codes")
    rpath = artifacts.path("backbone_codec_results")
    books = torch.load(bpath) if bpath.exists() else {}
    codes_store = torch.load(cpath) if cpath.exists() else {}
    res = torch.load(rpath)["res"] if rpath.exists() else {}

    for M, S, p in sweep:
        tag = f"{M}x{S}@p{p}"
        cfg = cfg_of(M, S, p)
        if tag not in books or books[tag]["cfg"] != cfg:
            log(f"[fit] {tag} ({codec.bits_of(M, S)} bits)")
            bias, atoms = codec.fit_global(x_tr, M, S, p, log=log)
            books[tag] = dict(cfg=cfg, bias=bias, atoms=atoms)
            torch.save(books, bpath)
        if tag in res and res[tag].get("cfg") == cfg:
            continue
        bias, atoms = books[tag]["bias"], books[tag]["atoms"]
        codes_tr = codec.encode_global(x_tr, bias, atoms, p)
        codes_ev = codec.encode_global(x_ev, bias, atoms, p)
        floor = rel_mse(codec.decode_global(codes_ev, bias, atoms, p), x_ev)
        coh, bar = codec.coherence_global(codes_tr, codes_ev, M, S)
        dead = sum((torch.bincount(codes_tr[:, m], minlength=S) == 0).sum()
                   for m in range(M)).item()
        codes_store[tag] = dict(cfg=cfg, tr=codes_tr.to(torch.int8),
                                ev=codes_ev.to(torch.int8))
        torch.save(codes_store, cpath)
        res[tag] = dict(cfg=cfg, M=M, S=S, p=p, bits=codec.bits_of(M, S),
                        floor=floor, coh=coh, bar=bar, dead=dead)
        torch.save(dict(res=res, sweep=config.BACKBONE_SWEEP), rpath)
        log(f"[{tag}] {codec.bits_of(M, S)} bits  floor {floor:.4f}  "
            f"coh {coh:.3f} (bar {bar:.3f}, over {coh - bar:+.3f})  "
            f"dead {dead}")

    log(f"\n{'point':>12}{'bits':>6}{'floor':>8}{'coh-over-bar':>14}{'dead':>6}")
    for tag, r in sorted(res.items(), key=lambda kv: (kv[1]["bits"],
                                                      kv[1]["p"])):
        log(f"{tag:>12}{r['bits']:>6}{r['floor']:>8.4f}"
            f"{r['coh'] - r['bar']:>+14.3f}{r['dead']:>6}")
    log(f"anchor: a per-patch code at 512 bits reaches "
        f"{config.PER_PATCH_CODEC_FLOOR}")
    fh.close()


if __name__ == "__main__":
    main()
