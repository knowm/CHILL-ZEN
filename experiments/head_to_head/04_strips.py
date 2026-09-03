"""What the arms look like under each binarization convention.

The head-to-head thresholds the grayscale renders at 0.1 rather than at
the midpoint. That is a convention match, not a tune, but it changes
every image, so it has to be looked at before any FID is read. One row
per arm: at 0.1 on the left, at the midpoint on the right, the same
sample index within each class.

    .venv-dtm/bin/python experiments/head_to_head/04_strips.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

ARMS = ["real-train", "codec-ceiling", "two-level", "one-level-128",
        "one-level-64", "one-level-32", "one-level-16"]
PER_CLASS = 512
PICK = 3


def main():
    have = [a for a in ARMS if (ev.GEN / f"{a}.npy").exists()]
    figu, axes = plt.subplots(len(have), 1, figsize=(11, 1.35 * len(have)))
    for ax, a in zip(np.atleast_1d(axes), have):
        im = np.load(ev.GEN / f"{a}.npy")
        pick = [c * PER_CLASS + PICK for c in range(10)]
        lo = np.concatenate([(im[i] > ev.THEIR_THRESH) for i in pick], 1)
        hi = np.concatenate([(im[i] > ev.OUR_THRESH) for i in pick], 1)
        gap = np.ones((28, 8))
        ax.imshow(np.concatenate([lo, gap, hi], 1), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_ylabel(a, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"on-fraction  {float((im > ev.THEIR_THRESH).mean()):.4f}"
                     f" at 0.1   |   {float((im > ev.OUR_THRESH).mean()):.4f}"
                     " at 0.5", fontsize=7, loc="right")
    np.atleast_1d(axes)[0].set_title(
        "binarized at 0.1 (theirs)                                "
        "binarized at 0.5 (the midpoint)", fontsize=9, loc="left")
    figu.suptitle("The same renders under the two binarization conventions, "
                  "one sample per class", fontsize=11)
    figu.tight_layout(rect=(0, 0, 1, 0.97))
    figu.savefig(ev.HERE / "strips.png", dpi=140)
    print("saved: strips.png")


if __name__ == "__main__":
    main()
