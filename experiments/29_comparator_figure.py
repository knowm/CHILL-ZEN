"""fig-comparator-sweep: the comparator-noise sweep, both stacks.

Paper Fig. 4 (`fig:comparatorsweep`, Sec. IV C). Two panels against
v_n/V, the comparator's input-referred noise relative to the read
voltage:

(a) grayscale end-to-end critic (real bar from the same run), read from
    `artifacts/comparator-sweep-results.pt` (`23_comparator_sweep.py`);
(b) binary-trained two-level FID, read from
    `head_to_head/comparator-score-results.json`
    (`07_comparator_score.py`, scoring the renders of
    `24_binary_comparator_sweep.py` and `--extend`), with each point's
    v_n/V taken from the sweep artifacts 24 writes.

Every plotted value comes off disk. Nothing here is transcribed, so
re-running 23, 24 and the scoring pass moves the figure, and a missing
input fails rather than drawing the previous run's curve.

The one constant is the DTM bar, which is not part of this sweep: it is
the reference arm's own FID from the head-to-head's `03_score.py`
(NUMBERS.md, Sec. V C).

The fixed-gain emulator control is marked on each panel as the point it
is: not on the v_n/V axis, since it is not a physical operating point.

    python experiments/29_comparator_figure.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch                                                    # noqa: E402

from chill_zen import artifacts                                 # noqa: E402

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.dpi": 200,
})

SCORES = pathlib.Path(__file__).resolve().parents[1] / \
    "experiments" / "head_to_head" / "comparator-score-results.json"

# The sweep is at one read voltage. P5 is the 50 mV twin of P3 -- the
# control showing v_n/V and not V sets the temperature -- so it shares an
# x with P3 and is not a point on this curve. Sec. IV C quotes it in the
# text instead.
V_SWEEP = 0.010

# The reference arm, not a point of this sweep: head_to_head/03_score.py.
DTM_BAR = 24.90


def need(p, how):
    if not p.exists():
        raise SystemExit(f"missing {p}\n  produce it with: {how}")
    return p


def gray_curve():
    """(v_n/V, critic) per point, the deployed control, and the real bar."""
    d = torch.load(need(artifacts.path("comparator_sweep"),
                        "python experiments/23_comparator_sweep.py"))
    pts = [(v_n / v_read, d["results"][name]["mean"][0])
           for name, (v_read, v_n) in d["points"].items()
           if v_read == V_SWEEP]
    pts.sort()
    return pts, d["results"]["deployed"]["mean"][0], d["real"][0]


def binary_curve():
    """(v_n/V, FID) per point, and the deployed control.

    FIDs come from the scoring pass; each point's v_n/V comes from the
    sweep artifact that rendered it, so the two are matched by the point
    name they were written under rather than by position.
    """
    scores = json.loads(need(
        SCORES, "python experiments/head_to_head/07_comparator_score.py "
                "(in .venv-dtm)").read_text())
    grid = {}
    for key, how in (("binary_comparator_sweep",
                      "python experiments/24_binary_comparator_sweep.py"),
                     ("binary_comparator_sweep_ext",
                      "python experiments/24_binary_comparator_sweep.py --extend")):
        d = torch.load(need(artifacts.path(key), how))
        for name, pt in d["points"].items():
            if pt is not None and pt[0] == V_SWEEP:   # not the control, not the twin
                grid[name.split()[0]] = pt[1] / pt[0]

    pts = []
    for tag, v in scores.items():
        p = tag.rsplit("-", 1)[-1]                # comparator-two-level-P3 -> P3
        if tag.startswith("comparator-two-level-") and p in grid:
            pts.append((grid[p], v["fid"]))
    pts.sort()
    dep = scores["comparator-two-level-deployed"]["fid"]
    return pts, dep


def main():
    gray_pts, gray_dep, gray_real = gray_curve()
    gx, gy = zip(*gray_pts)
    bin_pts, bin_dep = binary_curve()
    bx, by = zip(*bin_pts)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    a.plot(gx, gy, "o-", color="tab:blue", label="physical point")
    a.axhline(gray_real, color="gray", ls=":", lw=1, label="real bar")
    a.axhline(gray_dep, color="tab:red", ls="--", lw=1,
              label="control (emulator $T=0.1$)")
    a.set_xlabel(r"$v_n / V$")
    a.set_ylabel("grayscale end-to-end critic")
    a.legend(frameon=False, loc="center right")
    a.set_title("(a) grayscale")

    b.plot(bx, by, "o-", color="tab:blue", label="physical point")
    b.axhline(DTM_BAR, color="gray", ls=":", lw=1,
              label=f"DTM bar ({DTM_BAR:.2f})")
    b.axhline(bin_dep, color="tab:red", ls="--", lw=1,
              label="control (emulator $T=0.3$)")
    b.set_xlabel(r"$v_n / V$")
    b.set_ylabel("binary-trained two-level FID")
    b.invert_yaxis()
    b.legend(frameon=False, loc="center right")
    b.set_title("(b) binary")

    fig.tight_layout()
    out = artifacts.figure("fig-comparator-sweep.png")
    fig.savefig(out)
    print(f"grayscale: {[(round(x, 3), round(y, 4)) for x, y in zip(gx, gy)]}"
          f"  control {gray_dep:.4f}  real {gray_real:.4f}")
    print(f"binary:    {[(round(x, 3), round(y, 2)) for x, y in bin_pts]}"
          f"  control {bin_dep:.2f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
