"""Train the judge's critic, once.

Paper Sec. IV A. A pixel-space MLP, 784 -> 256 -> ReLU -> 10, trained
by Adam at 1e-3 for 8 epochs of batch 256 on the 68,000 training
images. It is frozen from then on and never sees a generated image
during training, which is what lets it be a judge.

Its own accuracy is not the point -- the real-reference row of every
verdict table is the bar a generator is measured against, and that row
is this network scoring real held-out images. Expected raw accuracy on
the 2000 held-out images: 0.8820. The real-reference row of Table II is
a different quantity, 0.8860: the same network on the class-balanced
1000-image reference sample the verdict uses.

Deterministic at a fixed seed. `artifacts/critic.pt` ships with the
repository; this script regenerates it.

Cost: about a minute.

    python experiments/00_train_critic.py
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, judge                             # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="retrain even if artifacts/critic.pt exists")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("00_train_critic")

    path = artifacts.path("critic")
    if path.exists() and not args.force:
        acc = torch.load(path)["acc"]
        log(f"[critic] {path.name} already present (eval {acc:.4f}); "
            "pass --force to retrain")
        fh.close()
        return

    X, y = load_fashion(seed=0)
    flat = X[:, 0].reshape(len(X), 784)
    critic, acc = judge.train_critic(flat[:N_TRAIN], y[:N_TRAIN],
                                     flat[N_TRAIN:], y[N_TRAIN:],
                                     seed=SEED, log=log)
    torch.save(dict(state=critic.state_dict(), acc=acc), path)
    log(f"[critic] saved {path.name}")
    fh.close()


if __name__ == "__main__":
    main()
