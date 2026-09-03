"""fig-comparator-sweep: the comparator-noise sweep, both stacks.

Paper Fig. 4 (`fig:comparatorsweep`, Sec. IV C). Two panels against
v_n/V, the comparator's input-referred noise relative to the read
voltage:

(a) grayscale end-to-end critic (real bar 0.8860), from
    `23_comparator_sweep.py`;
(b) binary-trained two-level FID (DTM bar 24.90), from
    `24_binary_comparator_sweep.py` (+ `--extend`).

The fixed-gain emulator control is marked on each panel as the point
it is: not on the v_n/V axis, since it is not a physical operating
point.

Values are the recorded sweep (NUMBERS.md, Sec. IV C).

    python experiments/29_comparator_figure.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts                                    # noqa: E402

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.dpi": 200,
})

VNV_GRAY = [0, 0.015, 0.05, 0.10, 0.20]
CRITIC = [0.8170, 0.8480, 0.8847, 0.9043, 0.8977]
GRAY_DEPLOYED = 0.8350
GRAY_REAL = 0.8860

VNV_BIN = [0, 0.015, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75]
FID = [22.65, 21.67, 16.39, 12.99, 11.23, 11.10, 11.57, 12.27]
BIN_DEPLOYED = 9.82
DTM_BAR = 24.90


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    a.plot(VNV_GRAY, CRITIC, "o-", color="tab:blue", label="physical point")
    a.axhline(GRAY_REAL, color="gray", ls=":", lw=1, label="real bar")
    a.axhline(GRAY_DEPLOYED, color="tab:red", ls="--", lw=1,
              label="control (emulator $T=0.1$)")
    a.set_xlabel(r"$v_n / V$")
    a.set_ylabel("grayscale end-to-end critic")
    a.legend(frameon=False, loc="center right")
    a.set_title("(a) grayscale")

    b.plot(VNV_BIN, FID, "o-", color="tab:blue", label="physical point")
    b.axhline(DTM_BAR, color="gray", ls=":", lw=1, label="DTM bar (24.90)")
    b.axhline(BIN_DEPLOYED, color="tab:red", ls="--", lw=1,
              label="control (emulator $T=0.3$)")
    b.set_xlabel(r"$v_n / V$")
    b.set_ylabel("binary-trained two-level FID")
    b.invert_yaxis()
    b.legend(frameon=False, loc="center right")
    b.set_title("(b) binary")

    fig.tight_layout()
    out = artifacts.figure("fig-comparator-sweep.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
