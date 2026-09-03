"""Where the draw deviates from real codes.

Paper Sec. IV D. Experiment 07 localizes the whole residual gap in one
component -- the label-to-backbone-code draw. This one measures its
signature by comparing the drawn codes of the n = 1000 batch against
the codes of real images. Nothing is taught.

Five probes:

1. per-book symbol marginals against the training marginals, in total
   variation, with a real evaluation sample as the sampling-noise bar;
2. over-picked symbols against atom contrast -- is the draw reaching
   for rare high-contrast atoms, which would explain the excess
   contrast;
3. book-pair co-occurrence, adjacent and offset-7, against the same bar;
4. re-encode self-consistency: decode a drawn code, encode the result,
   and see how often the same symbol comes back;
5. contrast accounting through the stack -- backbone decode and full
   render, drawn and real, against real images.

Two draws are compared side by side: the deployed one of experiment 07
and, if experiment 09 has been run, the prefix-schedule one. Expected:

                              deployed    prefix-schedule    bar
  marginal TV                   0.2138             0.2032   0.0454
    worst book                0.3409 (2)         0.4029 (5)
  pair-joint TV, adjacent       0.4015             0.3693   0.1622
  pair-joint TV, offset-7       0.4021             0.3690   0.1641
  re-encode, overall            0.8070             0.7794   0.6888
  re-encode, first half         0.8347             0.8060   0.7143
  contrast, backbone decode      1.226              1.198    1.039

with the full render at 1.236 end-to-end against 1.107 on the
reconstruction arm. Sec. IV D of the paper quotes the prefix-schedule
column for the distributional probes.

The over-picked symbols are near-zero-contrast atoms in the earliest
books -- the correlation with contrast is null (-0.002), and the top
over-picks sit in the bottom contrast quantile. It is not a reach for
contrast; it is a collapse onto modal flat atoms at the start of the
chain, where the context is only the label.

Re-encode self-consistency of drawn codes *exceeds* that of real codes.
The draw is not disobeying the codec. It is drawing from the wrong
distribution.

Cost: a couple of minutes. Requires 07 and 09.

    python experiments/10_draw_analysis.py
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, codec, config, judge              # noqa: E402
from chill_zen.data import N_TRAIN, load_fashion                   # noqa: E402
from chill_zen.generate import render, render_backbone             # noqa: E402

N_PER = 100


def tv(p, q):
    return 0.5 * (p - q).abs().sum(-1)


def marginals(codes, M, S):
    out = torch.zeros(M, S)
    for m in range(M):
        out[m] = torch.bincount(codes[:, m].long(), minlength=S).float()
    return out / len(codes)


def pair_joint(codes, pairs, S):
    out = torch.zeros(len(pairs), S * S)
    for k, (i, j) in enumerate(pairs):
        idx = codes[:, i].long() * S + codes[:, j].long()
        out[k] = torch.bincount(idx, minlength=S * S).float()
    return out / len(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("10_draw_analysis")

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    gb = torch.load(artifacts.path("backbone_books"))[args.backbone]
    kb = torch.load(artifacts.path("patch_books"))[args.patch]
    bias, atoms, p_g = gb["bias"], gb["atoms"], gb["cfg"]["p"]
    M, S = atoms.shape[0], atoms.shape[1]
    gcodes = torch.load(artifacts.path("backbone_codes"))[args.backbone]

    deployed = torch.load(artifacts.path("verdict"))
    draws = {"deployed draw": deployed["st"]["g_e2e"].long()}
    ppath = artifacts.path("prefix_verdict")
    if ppath.exists():
        draws["prefix-schedule draw"] = torch.load(ppath)["st"]["g_e2e"].long()
    else:
        log("[draw] no prefix-schedule verdict on disk — run 09 for the "
            "second column")

    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER]
                          for c in range(10)])
    real_img = X[N_TRAIN:, 0][real_idx].clamp(0, 1).reshape(-1, 784)
    g_real = gcodes["ev"].long()[real_idx]
    g_train = gcodes["tr"].long()
    res = {}

    # 1. per-book symbol marginals
    m_train = marginals(g_train, M, S)
    tv_real = tv(marginals(g_real, M, S), m_train)
    log(f"[draw] 1. per-book marginal TV against train (real-eval bar "
        f"{tv_real.mean():.4f}):")
    tv_books = {}
    for name, g in draws.items():
        t = tv(marginals(g, M, S), m_train)
        tv_books[name] = t
        log(f"     {name:<22} mean {t.mean():.4f}  books 0-{M // 2 - 1} "
            f"{t[:M // 2].mean():.4f}  books {M // 2}-{M - 1} "
            f"{t[M // 2:].mean():.4f}  max {t.max():.4f} at book "
            f"{t.argmax().item()}")
    res["tv_books"] = dict(tv_books, real=tv_real)

    # 2. over-pick against atom contrast
    contrast = (p_g * atoms).std(-1)
    log("[draw] 2. symbol over-pick (drawn frequency - train) against atom "
        "contrast:")
    for name, g in draws.items():
        delta = marginals(g, M, S) - m_train
        d, c = delta.flatten(), contrast.flatten()
        r = torch.corrcoef(torch.stack([d, c]))[0, 1].item()
        over = d.topk(5).indices
        cells = ", ".join(
            f"book {i // S} symbol {i % S} +{d[i]:.3f} (contrast quantile "
            f"{(c < c[i]).float().mean():.2f})" for i in over)
        log(f"     {name:<22} correlation {r:+.3f}")
        log(f"     {'':<22} top over-picks: {cells}")
        res.setdefault("overpick", {})[name] = dict(delta=delta, corr=r)
    res["contrast"] = contrast

    # 3. book-pair co-occurrence
    for tag, pairs in (("adjacent", [(m, m + 1) for m in range(M - 1)]),
                       ("offset-7", [(m, m + 7) for m in range(M - 7)])):
        j_train = pair_joint(g_train, pairs, S)
        bar = tv(pair_joint(g_real, pairs, S), j_train).mean().item()
        line = "  ".join(
            f"{name} {tv(pair_joint(g, pairs, S), j_train).mean().item():.4f}"
            for name, g in draws.items())
        log(f"[draw] 3. {tag} pair-joint TV against train:  {line}  "
            f"(real-eval bar {bar:.4f})")
        res.setdefault("pair_tv", {})[tag] = dict(bar=bar)

    # 4. re-encode self-consistency
    log("[draw] 4. re-encode self-consistency (decode, encode, agree):")
    reenc = {}
    for name, g in dict(draws, **{"real codes": g_real}).items():
        dec = codec.decode_global(g, bias, atoms, p_g)
        agree = (codec.encode_global(dec, bias, atoms, p_g) == g).float()
        reenc[name] = agree.mean(0)
        log(f"     {name:<22} overall {agree.mean():.4f}  "
            f"books 0-{M // 2 - 1} {agree[:, :M // 2].mean():.4f}  "
            f"{M // 2}-{M - 1} {agree[:, M // 2:].mean():.4f}")
    res["reenc"] = reenc

    # 5. contrast accounting
    real_std = judge.fg_std(real_img.reshape(-1, 28, 28))
    rows = [("real images", real_img.reshape(-1, 28, 28)),
            ("backbone decode, real codes", render_backbone(g_real, gb))]
    for name, g in draws.items():
        rows.append((f"backbone decode, {name}", render_backbone(g, gb)))
    rows.append(("render, end-to-end",
                 render(deployed["st"]["g_e2e"], deployed["st"]["p_e2e"],
                        gb, kb)[0]))
    rows.append(("render, reconstruction",
                 render(deployed["st"]["g_rec"], deployed["st"]["p_rec"],
                        gb, kb)[0]))
    log(f"[draw] 5. contrast accounting (foreground std; real "
        f"{real_std:.4f}):")
    acct = {}
    for name, img in rows:
        s = judge.fg_std(img)
        acct[name] = s
        log(f"     {name:<30} {s:.4f}  ratio {s / real_std:.3f}")
    res["contrast_accounting"] = dict(acct, real=real_std)
    torch.save(res, artifacts.path("draw_analysis"))
    fh.close()


if __name__ == "__main__":
    main()
