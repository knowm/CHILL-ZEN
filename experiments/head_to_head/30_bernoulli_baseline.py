"""Bernoulli calibration baselines for the head-to-head.

Two trivial models, rendered so the metric itself can be calibrated:

* **class-conditional** — for each class, every pixel flips
  independently at its frequency over that class's 6,000 images in the
  0.1-binarized canonical train split; 512 samples per class.
* **pooled** — the same, with one pixel-frequency map over all 60,000
  images; 5,120 samples, no class information at all.

No training, no lanes, no structure beyond per-pixel frequencies.
These are measurements of the metric, not models: they say how far
per-pixel statistics alone go under this FID protocol, which is the
context the paper's generated-arm numbers are read in.

Renders to `gen/audit-bernoulli-{class,pooled}.npy`, scored by
`32_audit_score.py`. The `audit-*` prefix is deliberate: nothing with
this prefix is a system arm, and `07_comparator_score.py`'s
`comparator-*` glob and `15_energy_figure.py` never see it.

Cost: under a minute.

    .venv-dtm/bin/python experiments/head_to_head/30_bernoulli_baseline.py
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

SEED = 31000
N_PER = 512


def save(name, imgs, lab, log):
    np.save(ev.GEN / f"{name}.npy", imgs.astype(np.float32))
    np.save(ev.GEN / f"{name}-lab.npy", lab.astype(np.int64))
    log(f"[gen] {name}: {imgs.shape}  on-fraction {imgs.mean():.4f}")


def main():
    log, fh = ev.make_log("30_bernoulli_baseline")
    log(f"\n=== Bernoulli calibration baselines "
        f"({time.strftime('%Y-%m-%d %H:%M')}) — seed {SEED} ===")
    x, y = ev.fashion("train")
    b = ev.binarize(x, ev.THEIR_THRESH)
    rng = np.random.default_rng(SEED)

    imgs = np.empty((10 * N_PER, 28, 28), np.float32)
    lab = np.repeat(np.arange(10), N_PER)
    for c in range(10):
        p_c = b[y == c].mean(0)
        u = rng.random((N_PER, 28, 28))
        imgs[c * N_PER:(c + 1) * N_PER] = (u < p_c[None]).astype(np.float32)
        log(f"  class {c}: n_train {int((y == c).sum())}  "
            f"pixel-frequency mean {p_c.mean():.4f}")
    save("audit-bernoulli-class", imgs, lab, log)

    p = b.mean(0)
    u = rng.random((10 * N_PER, 28, 28))
    pooled = (u < p[None]).astype(np.float32)
    save("audit-bernoulli-pooled", pooled, np.full(10 * N_PER, -1), log)

    log("done; score with 32_audit_score.py")
    fh.close()


if __name__ == "__main__":
    main()
