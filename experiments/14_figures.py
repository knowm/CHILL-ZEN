"""The paper's figures, rebuilt from the frozen artifacts.

No experiment runs here: decode, render and plot only.

  fig-samples.png     real | end-to-end | reconstruction strips, n = 1000
  fig-codec.png       (a) backbone / +patch / real  (b) rate-distortion
  fig-backbone.png    (a) backbone-only samples by book count
                      (b) critic against code shape
  fig-teach.png       the joint bank's teaching curve, split by level
  fig-noise.png       read noise against read voltage: the thermal 1/V
                      term over the flicker floor, at three pool sizes

The energy figure is experiment 15 and the binarization figure is
experiment 16; both belong to the head-to-head and need its outputs.

Cost: under a minute. Requires 01-07 (and 05 for fig-teach).

    python experiments/14_figures.py
"""
import argparse
import math
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.data import CLASSES, N_TRAIN, load_fashion          # noqa: E402
from chill_zen.generate import render, render_backbone             # noqa: E402
from chill_zen.energy import (A_FLICKER_UNIT, A_THERMAL_UNIT,      # noqa: E402
                         GAIN_PHYS, GMAX)

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    args = ap.parse_args()
    BACKBONE, PATCH = args.backbone, args.patch

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    st = torch.load(artifacts.path("verdict"))["st"]
    n_per = len(st["g_e2e"]) // 10
    lab = torch.arange(10).repeat_interleave(n_per)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:n_per]
                          for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)

    gb = torch.load(artifacts.path("backbone_books"))[BACKBONE]
    kb = torch.load(artifacts.path("patch_books"))[PATCH]
    e2e, _ = render(st["g_e2e"], st["p_e2e"], gb, kb)
    rec, _ = render(st["g_rec"], st["p_rec"], gb, kb)

    # ---------------- fig-samples ----------------------------------------
    n_show = 6
    cols = [("real references", real),
            ("end-to-end (label only)", e2e),
            ("reconstruction (real backbone code)", rec)]
    grid = [torch.cat([torch.cat(list(s[lab == c][:n_show]), 1)
                       for _, s in cols], 1) for c in range(10)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.imshow(torch.cat(grid, 0), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    for j in range(1, len(cols)):
        ax.axvline(j * n_show * 28 - 0.5, color="w", lw=1.2)
    ax.set_xticks([(j + 0.5) * n_show * 28 for j in range(len(cols))])
    ax.set_xticklabels([n for n, _ in cols])
    ax.set_yticks([(i + 0.5) * 28 for i in range(10)])
    ax.set_yticklabels(CLASSES)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(artifacts.figure("fig-samples.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig-samples.png")

    # ---------------- fig-codec ------------------------------------------
    gcodes = torch.load(artifacts.path("backbone_codes"))[BACKBONE]
    pick = torch.stack([torch.where(y_ev == c)[0][0] for c in range(10)])
    imgs_r = X[N_TRAIN:, 0][pick].clamp(0, 1)
    g_r = gcodes["ev"].long()[pick]
    p_codes_ev = torch.load(artifacts.path("patch_codes"))[PATCH]["ev"].long()
    p_r = p_codes_ev[pick].reshape(len(pick), -1)
    full, back = render(g_r, p_r, gb, kb)
    strip = [torch.cat([back[c], full[c], imgs_r[c]], 1) for c in range(10)]
    pres = torch.load(artifacts.path("patch_codec_results"))
    pc = pres["res"]
    deployed_bits = 512 + pc[PATCH]["bits"]
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(7.0, 2.9),
        gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    a1.imshow(torch.cat([torch.cat(strip[:5], 0),
                         torch.cat(strip[5:], 0)], 1),
              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a1.set_xticks([14, 42, 70, 98, 126, 154])
    a1.set_xticklabels(["backbone", "+patch", "real"] * 2)
    a1.set_yticks([])
    for s in a1.spines.values():
        s.set_visible(False)
    a1.tick_params(length=0)
    a1.set_title("(a) two-level reconstruction")
    for pkeep, mk in ((0.5, "o"), (0.75, "s")):
        pts = [(512 + pc[f"{m}x16@p{pkeep}"]["bits"],
                pc[f"{m}x16@p{pkeep}"]["combined"])
               for m in (1, 2, 3) if f"{m}x16@p{pkeep}" in pc]
        if pts:
            a2.plot([b for b, _ in pts], [v for _, v in pts], mk + "-",
                    label=f"residual patch level, keep-p = {pkeep}")
    bres = torch.load(artifacts.path("backbone_codec_results"))["res"]
    if "128x16@p0.75" in bres:
        a2.plot([512], [bres["128x16@p0.75"]["floor"]], "d", color="k",
                label="global code alone (512 b)")
    a2.plot([512], [config.PER_PATCH_CODEC_FLOOR], "x", color="k",
            label="per-patch code alone (512 b)")
    a2.plot([deployed_bits], [pc[PATCH]["combined"]], "*", ms=10, color="C3",
            label=f"deployed point ({deployed_bits} b)")
    a2.set_xlabel("total code bits per image")
    a2.set_ylabel("reconstruction relMSE")
    a2.set_title("(b) rate–distortion")
    a2.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(artifacts.figure("fig-codec.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig-codec.png")

    # ---------------- fig-backbone ---------------------------------------
    sweep = torch.load(artifacts.path("backbone_sweep"))
    jb_all = torch.load(artifacts.path("backbone_books"))
    n320 = len(next(iter(sweep["states"].values())))
    lab320 = torch.arange(10).repeat_interleave(n320 // 10)
    tags = [t for t in config.FRONTIER if t in sweep["states"]]
    panels = []
    for tag in tags:
        g = sweep["states"][tag].long()[:, 1:]
        dec = render_backbone(g, jb_all[tag])
        rows = [torch.cat(list(dec[lab320 == c][:2]), 1) for c in (0, 2, 8, 9)]
        panels.append(torch.cat(rows, 0))
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(7.0, 2.7),
        gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    a1.imshow(torch.cat(panels, 1), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    a1.set_xticks([(i + 0.5) * 56 for i in range(len(tags))])
    a1.set_xticklabels([f"{t.split('x')[0]} books" if i == 0
                        else t.split("x")[0] for i, t in enumerate(tags)])
    a1.set_yticks([])
    a1.tick_params(length=0)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_title("(a) backbone-only samples")
    res = sweep["res"]
    books = [16, 32, 64, 128]
    for pkeep in (0.5, 0.75):
        pts = [(m, res[f"{m}x16@p{pkeep}"][0]) for m in books
               if f"{m}x16@p{pkeep}" in res]
        if pts:
            a2.plot([m for m, _ in pts], [v for _, v in pts], "o-",
                    label=f"S = 16, keep-p = {pkeep}")
    pts32 = [(m, res[f"{m}x32@p0.5"][0]) for m in (16, 32, 64)
             if f"{m}x32@p0.5" in res]
    if pts32:
        a2.plot([m for m, _ in pts32], [v for _, v in pts32], "s--",
                label="S = 32, keep-p = 0.5")
    if "64x16@p1.0" in res:
        a2.plot([64], [res["64x16@p1.0"][0]], "x", color="k",
                label="keep-p = 1 (plain AQ)")
    a2.set_xscale("log", base=2)
    a2.set_xticks(books)
    a2.set_xticklabels(books)
    a2.set_xlabel("number of books")
    a2.set_ylabel("critic agreement")
    a2.set_title("(b) generation vs code shape")
    a2.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(artifacts.figure("fig-backbone.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig-backbone.png")

    # ---------------- fig-teach ------------------------------------------
    curve = torch.load(artifacts.path("joint_bank"))["curve"]
    ep = list(range(1, len(curve) + 1))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(ep, [c[0] for c in curve], "o-", label="overall repair")
    ax.plot(ep, [c[1][0] for c in curve], "s--", label="global addresses")
    ax.plot(ep, [c[1][1] for c in curve], "d--", label="patch addresses")
    ax.set_xlabel("teaching epoch")
    ax.set_ylabel("repair accuracy")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(artifacts.figure("fig-teach.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig-teach.png")

    # ---------------- fig-noise ------------------------------------------
    op = artifacts.path("operating_point")
    if op.exists():
        cl = torch.load(op)["classes"]
        m_draw = cl["backbone draw (hot)"]["m_med"]
        m_sweep = cl["joint sweep (cold)"]["m_med"]
    else:
        m_draw, m_sweep = 2071.0, 22647.0
    V = torch.logspace(-3.3, 0.0, 300)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m, ls, lbl in ((float(GMAX), "-", r"$m = m_{\rm ref}$ (one pair)"),
                       (m_draw, "--", rf"$m = {m_draw / 1e3:.1f}"
                                      r"{\times}10^3$ (draw pool)"),
                       (m_sweep, ":", rf"$m = {m_sweep / 1e4:.1f}"
                                      r"{\times}10^4$ (sweep pool)")):
        f_m = math.sqrt(GMAX / m)
        s_th = GAIN_PHYS * A_THERMAL_UNIT * f_m / V
        s_fl = GAIN_PHYS * A_FLICKER_UNIT * f_m * 0.9      # (1 - y^2) ~ 0.9
        s = torch.sqrt(s_th ** 2 + s_fl ** 2)
        ln = ax.loglog(V, s, ls, label=lbl)
        ax.axhline(s_fl, color=ln[0].get_color(), lw=0.6, alpha=0.5)
    ax.axvline(0.05, color="k", lw=0.8, alpha=0.6)
    ax.annotate("subthreshold\nread voltage", xy=(0.05, 0.12),
                xycoords=("data", "axes fraction"), fontsize=6,
                ha="left", xytext=(3, 0), textcoords="offset points")
    ax.set_xlabel("read voltage $V$ (V)")
    ax.set_ylabel(r"read noise $\sigma_y$")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(artifacts.figure("fig-noise.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig-noise.png")


if __name__ == "__main__":
    main()
