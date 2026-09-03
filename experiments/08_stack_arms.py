"""What each component of the stack contributes, at n = 320.

Paper Sec. IV B (the joint sweep and the capacity delta) and Sec. IV E (a).
Four arms on one drawn batch, so the comparison is between components
rather than between batches:

  backbone      the one-level render, before the patch level exists
  patch level   backbone plus one parallel patch read, no reconciliation
  joint         the deployed system: the same state after one cold
                joint sweep over all 226 addresses
  joint-recon   the same, from a real backbone code -- the ceiling and
                the do-no-harm control in one

Expected at the deployed 128-book shape:

    arm            critic     div    tone   seam   std-ratio
    real           0.9031  0.4823   2.360  1.100        1.00
    backbone       0.8344  0.5156   2.582  1.053        1.21
    patch level    0.8375  0.5184   2.609  1.061        1.21
    joint          0.8344  0.4771   2.733  1.051        1.24
    joint-recon    0.9031  0.5005   2.095  1.062        1.11

Two readings. The joint sweep moves diversity onto the real bar
(0.4771 against 0.4823) while leaving good states alone -- it
reconciles, it does not repaint. And end-to-end quality tracks the
backbone: the patch level decorates a skeleton and cannot rescue one.

Run it again with `--backbone 64x16@p0.5` (after teaching that stack)
for the matched 64-book comparison the paper quotes: joint 0.8062
against 0.8344 at 128 books.

Cost: under a minute. Requires 01-05.

    python experiments/08_stack_arms.py
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config, judge                     # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import T_GEN, draw_backbone, render        # noqa: E402
from chill_zen.generate import render_backbone                     # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from importlib import import_module                           # noqa: E402

load_stack = import_module("07_verdict").load_stack

N_PER = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("08_stack_arms")

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(args.backbone, args.patch)
    bl, pl, jl = S["bl"], S["pl"], S["jl"]
    critic = judge.load_critic(artifacts.path("critic"))

    lab = torch.arange(10).repeat_interleave(N_PER)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER]
                          for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    g_real = S["gcodes"]["ev"].long()[real_idx]

    gen = torch.Generator().manual_seed(SEED + 700)
    t0 = time.time()
    g_open = draw_backbone(bl, S["g1"], S["r1"], lab, gen)
    log(f"[arms] backbone drawn ({time.time() - t0:.0f}s)")
    none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
    p_d0 = pl.read_all(S["g2"], lab, g_open, none, T_GEN, gen)
    p_r0 = pl.read_all(S["g2"], lab, g_real, none, T_GEN, gen)

    def sweep(g, p, tag):
        pred = jl.read_joint(S["jbank"], lab, g, p, 0.0, None)
        g_j, p_j = pred[:, :jl.NG], pred[:, jl.NG:]
        log(f"[arms] {tag}: the joint sweep changed "
            f"{(g_j != g).float().mean().item():.3f} of backbone addresses "
            f"and {(p_j != p).float().mean().item():.3f} of patch addresses")
        return g_j, p_j

    g_j, p_j = sweep(g_open, p_d0, "joint")
    g_jr, p_jr = sweep(g_real, p_r0, "joint-recon")

    arms = [("backbone", render_backbone(g_open, S["gb"])),
            ("patch level", render(g_open, p_d0, S["gb"], S["kb"])[0]),
            ("joint", render(g_j, p_j, S["gb"], S["kb"])[0]),
            ("joint-recon", render(g_jr, p_jr, S["gb"], S["kb"])[0])]

    real_std = judge.fg_std(real)
    res = {}
    log(f"\nn = {len(lab)}")
    log(f"{'arm':<14}{'critic':>8}{'div':>8}{'tone':>7}{'seam':>7}{'stdr':>7}")
    ra, rdv = judge.judge(critic, real, lab)
    log(f"{'real':<14}{ra:>8.4f}{rdv:>8.4f}"
        f"{judge.tone_spread(real, lab):>7.3f}"
        f"{judge.seam_ratio(real):>7.3f}{1.0:>7.2f}")
    for name, img in arms:
        res[name] = judge.verdict_row(critic, img, lab, real_std)
        log(f"{name:<14}{res[name][0]:>8.4f}{res[name][1]:>8.4f}"
            f"{res[name][2]:>7.3f}{res[name][3]:>7.3f}{res[name][4]:>7.2f}")
    torch.save(dict(cfg=dict(seed=SEED + 700, backbone=args.backbone,
                             patch=args.patch, t_gen=T_GEN, n_per=N_PER),
                    res=res, real=(ra, rdv)),
               artifacts.ARTIFACTS / "stack-arms-results.pt")
    fh.close()


if __name__ == "__main__":
    main()
