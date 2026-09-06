"""fig-binarization: the two binarization conventions, side by side.

Paper Fig. 7 (`fig:binarization`, Sec. V C). Which threshold turns a grayscale render into a binary
image is a property of the data, not a model choice, and the DTM paper
does not state theirs. It is 0.1, recovered from their released code
and confirmed against their own shipped reference statistics by the
head-to-head gate. This figure shows what the difference looks like:
the same rendered images, one sample per class, thresholded at 0.1 and
at the midpoint.

It reads the exact 5120-image sets that were scored at the operating
point of Sec. IV C, so it shows the images the FID numbers describe
rather than a fresh draw.

Requires the head-to-head's `25_comparator_headtohead.py` and `06_binary_arm.py`.

    python experiments/16_binarization_figure.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chill_zen import artifacts                                    # noqa: E402
from chill_zen.data import N_TRAIN, load_fashion                   # noqa: E402

GEN = ROOT / "experiments" / "head_to_head" / "gen"
THEIRS, OURS = 0.1, 0.5
ROWS = [("real", "real images"),
        ("comparator-two-level", "two-level CHILL ZEN"),
        ("comparator-one-level-128", "one-level, 128 books")]
PER_CLASS = 512
PICK = 3


def best_render(arm):
    """The grayscale arm's best-scoring render on the register grid
    (`comparator-gray-{arm}-{P}`, scored by 07_comparator_score.py), else
    the single-point render `comparator-{arm}` of 25."""
    import json
    spath = GEN.parent / "comparator-score-results.json"
    if spath.exists():
        scores = json.loads(spath.read_text())
        pre = f"comparator-gray-{arm[len('comparator-'):]}-"
        grid = {k: v["fid"] for k, v in scores.items() if k.startswith(pre)}
        if grid:
            return min(grid, key=grid.get)
    return arm


def real_train():
    """The first 512 train images of each class, the head-to-head's own
    real-image control, rebuilt from the data rather than a cached file."""
    import torch
    X, y = load_fashion(seed=0)
    y_tr = y[:N_TRAIN].long()
    idx = torch.cat([torch.where(y_tr == c)[0][:PER_CLASS] for c in range(10)])
    return X[:N_TRAIN, 0][idx].clamp(0, 1).numpy().astype(np.float32)


def main():
    fig, axes = plt.subplots(len(ROWS), 1,
                             figsize=(7.0, 0.62 * len(ROWS) + 0.25),
                             gridspec_kw=dict(hspace=0.25))
    for ax, (arm, nice) in zip(np.atleast_1d(axes), ROWS):
        im = real_train() if arm == "real" else np.load(GEN / f"{best_render(arm)}.npy")
        pick = [c * PER_CLASS + PICK for c in range(10)]
        lo = np.concatenate([(im[i] > THEIRS) for i in pick], 1)
        hi = np.concatenate([(im[i] > OURS) for i in pick], 1)
        gap = np.ones((28, 10))
        ax.imshow(np.concatenate([lo, gap, hi], 1), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_ylabel(nice, fontsize=7, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    np.atleast_1d(axes)[0].set_title(
        "threshold 0.1 (the DTM convention)"
        "                                  threshold 0.5",
        fontsize=8, loc="left")
    fig.savefig(artifacts.figure("fig-binarization.png"),
                bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("fig-binarization.png")
    for arm, _ in ROWS:
        im = real_train() if arm == "real" else np.load(GEN / f"{best_render(arm)}.npy")
        print(f"  {arm}: on-fraction {float((im > THEIRS).mean()):.4f} at "
              f"0.1, {float((im > OURS).mean()):.4f} at 0.5")


if __name__ == "__main__":
    main()
