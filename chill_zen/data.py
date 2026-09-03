"""Fashion-MNIST, in the order every experiment here was fitted on.

The 70,000 images as OpenML serves them (train and test concatenated),
permuted once by numpy's PCG64 at seed 0, then cut 68,000 / 2,000.
Codecs and banks see only the first 68,000; every probe and every judge
number is on the last 2,000.

The head-to-head of Sec. V C uses a different data order, the canonical
idx files with train and test kept apart, because the reference
statistics it is scored against were computed over the canonical
60,000-image train split. It reads them itself, in
`experiments/head_to_head/dtm_fid.py`, so that it needs nothing from
this module and the two orders cannot be mixed by accident.
"""
import numpy as np
import torch

SEED = 0
N_TRAIN, N_EVAL = 68000, 2000
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Boot"]


def load_fashion(seed=0):
    """-> X [70000, 1, 28, 28] float in [0, 1], y [70000] long.

    Downloaded and cached by scikit-learn's OpenML fetcher on first
    call. The permutation is numpy's default_rng(seed), which is a
    stated part of the split: every artifact in this repository was
    fitted on `X[:68000]` of `load_fashion(seed=0)`.
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False,
                      parser="liac-arff")
    X = ds.data.astype(np.float32) / 255.0
    y = ds.target.astype(np.int64)
    p = np.random.default_rng(seed).permutation(len(X))
    return torch.from_numpy(X[p]).reshape(-1, 1, 28, 28), torch.from_numpy(y[p])


def rel_mse(rec, x):
    """Reconstruction error as a fraction of the data variance."""
    return (((rec - x) ** 2).mean() / x.var()).item()
