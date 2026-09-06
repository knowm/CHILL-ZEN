"""fig-energy: the frontier on the performance-versus-energy plane.

Paper Fig. 8 (`fig:energy`, Sec. V C). Every FID here is produced by
the evaluation code,
reference statistics, binarization threshold and sample count of the
DTM paper; the energies are this work's model (chill_zen/energy_geom.py,
Appendix B) at the nominal point, with a band out to the pessimistic
corner.

(a) their own axes -- energy per sample on a log scale against inverse
    FID -- with their five reported series replotted from their data,
    and the CHILL ZEN frontier drawn as the same kind of curve;
(b) the same frontier against the DTM chain alone, as FID, so the
    margin is readable, with the codec ceiling and the real-data floor.

Both panels also carry the three binary-trained arms of Table VII, all
now at the physical operating point of Sec. IV C (two-level at
v_n/V = 0.30, one-level 128 and 64 at the same point) at their
computed energies.

Requires experiment 13. FIDs are read from the physical-point scoring
pass (`head_to_head/comparator-score-results.json`, written by
`07_comparator_score.py`), which is what Table VII reports; the codec
ceiling and the real-train control come from `score-results.json`. The
recorded values below are fallbacks for a clone that has not run the
head-to-head. The emulator-temperature arms in `score-results.json` and
`binary-score-results.json` are superseded and are not plotted.

    python experiments/15_energy_figure.py
"""
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))

from chill_zen import artifacts                                    # noqa: E402
from dtm_figure1_data import SERIES                           # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.dpi": 200,
})

DTM_BEST = (1.568e-8, 24.9)

# Fallbacks only: the recorded physical-point verdicts (NUMBERS.md
# Sec. V C), each grayscale arm at its best level on the register grid,
# binary-trained at P6 (10 mV, 3 mV). Used when the scoring pass has not
# been run here.
RECORDED = {
    "one-level-16": 44.15, "one-level-32": 24.81,
    "one-level-64": 22.62, "one-level-128": 22.11,
    "two-level": 18.09, "codec-ceiling": 14.205, "real-train": 1.907,
}
RECORDED_BINARY = {
    "binary-two-level-T0.3": 9.69,
    "binary-one-level-128-T0.2": 15.46,
    "binary-one-level-64-T0.2": 15.10,
}
# Which physical-point arm each plotted series is. The binary keys keep
# the T-arm names because 22_binary_energy.py's census is stored under
# them; the FID is the physical-point render's.
PHYSICAL = {
    "one-level-16": "comparator-one-level-16",
    "one-level-32": "comparator-one-level-32",
    "one-level-64": "comparator-one-level-64",
    "one-level-128": "comparator-one-level-128",
    "two-level": "comparator-two-level",
    "binary-two-level-T0.3": "comparator-two-level-P6",
    "binary-one-level-128-T0.2": "comparator-binary-one-level-128",
    "binary-one-level-64-T0.2": "comparator-binary-one-level-64",
}
# Binary energies (22_binary_energy.py), nominal point; the model charges
# counts only, so these equal the grayscale twins'.
BINARY_E = {
    "binary-two-level-T0.3": 2.00e-9,
    "binary-one-level-128-T0.2": 7.5e-10,
    "binary-one-level-64-T0.2": 2.1e-10,
}


def main():
    frontier = torch.load(artifacts.path("energy_frontier"))
    h2h = ROOT / "experiments" / "head_to_head"
    spath = h2h / "score-results.json"
    scores = json.loads(spath.read_text()) if spath.exists() else None
    ppath = h2h / "comparator-score-results.json"
    phys = json.loads(ppath.read_text()) if ppath.exists() else None
    epath = artifacts.path("binary_energy")
    if epath.exists():
        er = torch.load(epath)["results"]
        BINARY_E["binary-two-level-T0.3"] = er["two-level"]["nominal"]
        BINARY_E["binary-one-level-128-T0.2"] = er["128x16@p0.5"]["nominal"]
        BINARY_E["binary-one-level-64-T0.2"] = er["64x16@p0.5"]["nominal"]

    def F(arm):
        # Physical-point rows from the comparator scoring pass: each
        # grayscale arm at its best level on the register grid
        # (34_grayscale_grid.py) when the grid has been scored, else the
        # single-point render of 25. The codec ceiling and the real-train
        # control are the same images either way and come from the
        # deployed scoring pass.
        if arm in PHYSICAL and phys is not None:
            grid = [v["fid"] for k, v in phys.items()
                    if k.startswith(f"comparator-gray-{arm}-")]
            if grid:
                return min(grid)
            return phys[PHYSICAL[arm]]["fid"]
        if arm not in PHYSICAL and scores is not None:
            return scores[f"{arm}@0.1"]["theirs"]
        return RECORDED[arm]

    def FB(arm):
        if phys is not None:
            return phys[PHYSICAL[arm]]["fid"]
        return RECORDED_BINARY[arm]

    sizes = [16, 32, 64, 128]
    arms = [f"one-level-{m}" for m in sizes] + ["two-level"]
    x_lo = [frontier[m]["lo"] for m in sizes] + [frontier["deployed"]["lo"]]
    x_hi = [frontier[m]["hi"] for m in sizes] + [frontier["deployed"]["hi"]]
    ys = [F(a) for a in arms]
    ceiling = F("codec-ceiling")
    floor = F("real-train")
    b_arms = ["binary-one-level-64-T0.2", "binary-one-level-128-T0.2",
              "binary-two-level-T0.3"]
    bx = [BINARY_E[a] for a in b_arms]
    by = [FB(a) for a in b_arms]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    styles = {"DTM": "tab:orange", "MEBM": "tab:gray", "GAN": "tab:blue",
              "VAE": "tab:green", "DDPM": "tab:purple"}
    markers = {"DTM": "x", "MEBM": "x", "GAN": "o", "VAE": "o", "DDPM": "o"}
    # Series are labelled on the curves: a legend box large enough for six
    # entries would cover points in every corner of this plane.
    labels = {
        "DTM": ("DTM (prob. comp.)", 2.5e-8, 0.0460, "left"),
        "MEBM": ("MEBM (prob. comp.)", 1.4e-5, 0.0243, "left"),
        "GAN": ("GAN (GPU)", 3.5e-3, 0.0300, "left"),
        "VAE": ("VAE (GPU)", 7e-4, 0.0595, "right"),
        "DDPM": ("DDPM (GPU)", 3e-1, 0.0500, "right"),
    }
    for name, pts in SERIES.items():
        a1.plot([e for e, _ in pts], [1.0 / f for _, f in pts], "--",
                color=styles[name], marker=markers[name], ms=4, lw=0.9)
        txt, lx, ly, ha = labels[name]
        a1.annotate(txt, (lx, ly), color=styles[name], fontsize=6,
                    ha=ha, va="center", linespacing=1.1)
    inv = [1.0 / f for f in ys]
    a1.fill(x_lo + x_hi[::-1], inv + inv[::-1], color="tab:red",
            alpha=0.15, lw=0)
    a1.plot(x_lo, inv, "-", color="tab:red", marker="x", ms=4.5, lw=1.2)
    a1.annotate("CHILL ZEN\n[this work]", (1.2e-12, 0.0615), color="tab:red",
                fontsize=6.5, fontweight="bold", ha="left", va="center",
                linespacing=1.1)
    a1.plot(x_lo[-1], inv[-1], "*", color="tab:red", ms=9)
    a1.plot(x_hi, inv, ":", color="tab:red", lw=0.7, alpha=0.7)
    a1.plot(bx, [1.0 / f for f in by], "--D", color="tab:brown", ms=3.5,
            lw=0.9)
    a1.annotate("binary-trained", (1.2e-12, 0.098), color="tab:brown",
                fontsize=6, ha="left", va="center")
    a1.set_title("(a) the reported plane")
    a1.set_xlim(1e-12, 1e1)
    a1.set_ylim(0, 0.112)
    a1.set_xlabel("Energy Consumption [J/Sample]")
    a1.set_ylabel(r"Performance [FID$^{-1}$]")

    a2.plot([e for e, _ in SERIES["DTM"]], [f for _, f in SERIES["DTM"]],
            "-s", color="tab:orange", ms=4, lw=1.0, label="DTM chain")
    for (e, f), n in zip(SERIES["DTM"], [1, 2, 4, 6, 8]):
        if f < 60:
            a2.annotate(f"{n}", (e, f), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color="tab:orange")
    a2.fill(x_lo + x_hi[::-1], ys + ys[::-1], color="tab:red", alpha=0.15,
            lw=0)
    a2.plot(x_lo, ys, "-o", color="tab:red", ms=4, lw=1.2,
            label="CHILL ZEN, nominal point")
    a2.plot(x_hi, ys, "--^", color="tab:red", ms=3.5, lw=0.8, alpha=0.55,
            label="CHILL ZEN, pessimistic corner")
    a2.plot(x_lo[-1], ys[-1], "*", color="tab:red", ms=10)
    names = ["16", "32", "64", "128", "two-level"]
    offs = [(5, 2), (4, -10), (4, -10), (-16, 4), (7, 1)]
    for x, yv, nm, off in zip(x_lo, ys, names, offs):
        a2.annotate(nm, (x, yv), textcoords="offset points", xytext=off,
                    fontsize=6, color="tab:red")
    a2.plot(bx, by, "--D", color="tab:brown", ms=3.5, lw=0.9,
            label="binary-trained (physical point)")
    b_names = ["64", "128", "two-level"]
    b_offs = [(4, -10), (4, -9), (5, -3)]
    for x, yv, nm, off in zip(bx, by, b_names, b_offs):
        a2.annotate(nm, (x, yv), textcoords="offset points", xytext=off,
                    fontsize=6, color="tab:brown")
    a2.axhline(ceiling, color="tab:green", lw=0.8, ls="-.", alpha=0.9)
    a2.annotate(f"128-book codec ceiling ({ceiling:.1f})",
                (4.5e-9, ceiling + 1), fontsize=6, color="tab:green",
                alpha=0.95)
    a2.axhline(floor, color="k", lw=0.8, ls=":", alpha=0.7)
    a2.annotate(f"real data ({floor:.2f})", (6e-9, floor + 1), fontsize=6,
                alpha=0.8)
    a2.set_title("(b) the head-to-head, zoomed")
    a2.set_xlim(2e-11, 5e-8)
    a2.set_ylim(0, 50)
    a2.set_xlabel("Energy Consumption [J/Sample]")
    a2.set_ylabel("FID (lower is better)")
    a2.legend(loc="upper right", fontsize=6, frameon=True, framealpha=0.92,
              edgecolor="none")

    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
    fig.tight_layout(pad=0.5)
    fig.savefig(artifacts.figure("fig-energy.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig-energy.png")
    for a, xl, xh, f in zip(arms, x_lo, x_hi, ys):
        print(f"  {a}: E [{xl:.2e}, {xh:.2e}] J, FID {f:.2f}")
    for a, x, f in zip(b_arms, bx, by):
        print(f"  {a}: E {x:.2e} J, FID {f:.2f}")
    print(f"  ceiling {ceiling:.2f}   real floor {floor:.2f}   "
          f"their best DTM {DTM_BEST[1]} at {DTM_BEST[0]:.3e} J")


if __name__ == "__main__":
    main()
