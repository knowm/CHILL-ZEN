"""Seed spread, memorization, and the sweep's flip rate.

Three reporting measurements behind Sec. IV B and Sec. V A:

* **Seeds** -- three fresh seeds of the Table II verdict (end-to-end
  and reconstruction arms, n = 1000 each). Table II carries the mean
  +/- spread (max - min), replacing the previously asserted +/-0.005
  evaluation noise.
* **Memorization and mode coverage** -- n = 5120 (512 per class, its
  own declared seed): distinct joint-code and backbone-code counts,
  exact-copy count, and the nearest-train-neighbour L2 distribution of
  generated renders against the same distribution for held-out real
  images.
* **Flip rate** -- addresses flipped by joint-sweep doses 2, 3, and 4
  on the end-to-end states. This is the measurement behind not
  claiming that the sweep settles: the churn decays but persists, so
  one dose is a reconciliation pass, not a fixed point.

Expected:

    end-to-end       critic 0.8317 spread 0.0130   div 0.4715 / 0.0067
    reconstruction   critic 0.8877 spread 0.0040   div 0.5019 / 0.0007
    distinct codes   5120/5120 joint and backbone, exact copies 0
    NN-to-train L2   generated q01/q50 1.558/2.880, held-out 1.465/3.433
    flip rate        dose 2: 40.8/226 addresses, dose 3: 21.7, dose 4: 13.5

Cost: about ten minutes. Requires 01-05.

    python experiments/21_reporting.py
"""
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from importlib import import_module                           # noqa: E402

load_stack = import_module("07_verdict").load_stack

SEEDS_A = [SEED + 960, SEED + 961, SEED + 962]
SEED_B = SEED + 970
N_PER_A, N_PER_B = 100, 512
DOSES = 4


def gen_states(S, lab, gen, g_real=None):
    """One full two-level pass; g_real substitutes the draw (recon arm)."""
    pl, jl = S["pl"], S["jl"]
    g_open = (draw_backbone(S["bl"], S["g1"], S["r1"], lab, gen)
              if g_real is None else g_real)
    none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
    p_d = pl.read_all(S["g2"], lab, g_open, none, T_GEN, gen)
    pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
    return pj


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("21_reporting")
    log(f"reporting measurements ({time.strftime('%Y-%m-%d %H:%M')})")

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(config.BACKBONE, config.PATCH)
    pl, jl = S["pl"], S["jl"]
    critic = judge.load_critic(artifacts.path("critic"))

    # ---- seeds: three fresh seeds on the verdict ----
    lab = torch.arange(10).repeat_interleave(N_PER_A)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER_A]
                          for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    real_std = judge.fg_std(real)
    g_real = S["gcodes"]["ev"].long()[real_idx]
    log("\nseeds: three fresh seeds, n = 1000")
    log(f"{'arm':<16}{'seed':>6}{'critic':>8}{'div':>8}{'tone':>7}"
        f"{'seam':>7}{'stdr':>7}")
    seeds_out = {}
    for name, greal in (("end-to-end", None), ("reconstruction", g_real)):
        rows = []
        for seed in SEEDS_A:
            gen = torch.Generator().manual_seed(seed)
            pj = gen_states(S, lab, gen, greal)
            img, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
            row = judge.verdict_row(critic, img, lab, real_std)
            rows.append(row)
            log(f"{name:<16}{seed:>6}{row[0]:>8.4f}{row[1]:>8.4f}"
                f"{row[2]:>7.3f}{row[3]:>7.3f}{row[4]:>7.2f}")
        t = torch.tensor(rows)
        seeds_out[name] = dict(rows=rows, mean=t.mean(0).tolist(),
                               spread=(t.max(0).values
                                       - t.min(0).values).tolist())
        m, s = seeds_out[name]["mean"], seeds_out[name]["spread"]
        log(f"{name:<16}{'mean':>6}{m[0]:>8.4f}{m[1]:>8.4f}{m[2]:>7.3f}"
            f"{m[3]:>7.3f}{m[4]:>7.2f}   spread critic {s[0]:.4f} "
            f"div {s[1]:.4f}")

    # ---- memorization / mode coverage at n = 5120 ----
    log(f"\nmemorization: n = 5120 at seed {SEED_B}")
    lab_b = torch.arange(10).repeat_interleave(N_PER_B)
    gen = torch.Generator().manual_seed(SEED_B)
    t0 = time.time()
    pj = gen_states(S, lab_b, gen)
    img, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
    log(f"  drawn+rendered ({time.time() - t0:.0f}s)")
    uj = {tuple(r.tolist()) for r in pj}
    ug = {tuple(r.tolist()) for r in pj[:, :jl.NG]}
    log(f"  distinct joint codes {len(uj)}/{len(pj)}  distinct backbone "
        f"codes {len(ug)}/{len(pj)}")

    x_tr = X[:N_TRAIN, 0].clamp(0, 1).reshape(N_TRAIN, 784)
    tr_sq = (x_tr ** 2).sum(1)

    def nn_dist(q):
        q = q.reshape(len(q), -1)
        out = torch.empty(len(q))
        for lo in range(0, len(q), 256):
            blk = q[lo:lo + 256]
            d = (blk ** 2).sum(1)[:, None] + tr_sq[None, :] \
                - 2.0 * blk @ x_tr.T
            out[lo:lo + 256] = d.clamp(min=0).min(1).values.sqrt()
        return out

    t0 = time.time()
    d_gen = nn_dist(img)
    heldout = X[N_TRAIN:, 0].clamp(0, 1)
    d_real = nn_dist(heldout)
    q = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0])
    qg = torch.quantile(d_gen, q)
    qr = torch.quantile(d_real, q)
    log(f"  NN-to-train L2, generated (n={len(d_gen)}): "
        + "  ".join(f"q{int(x * 100):02d} {v:.3f}" for x, v in zip(q, qg))
        + f"  ({time.time() - t0:.0f}s)")
    log(f"  NN-to-train L2, held-out real (n={len(d_real)}): "
        + "  ".join(f"q{int(x * 100):02d} {v:.3f}" for x, v in zip(q, qr)))
    log(f"  exact-copy count (distance 0): generated "
        f"{(d_gen < 1e-4).sum().item()}, held-out "
        f"{(d_real < 1e-4).sum().item()}")

    # ---- flip rate under repeated sweep doses ----
    log("\nflip rate: joint-sweep doses on the end-to-end states (n = 5120)")
    state = pj.clone()
    flips = []
    for dose in range(2, DOSES + 1):
        nxt = jl.read_joint(S["jbank"], lab_b, state[:, :jl.NG],
                            state[:, jl.NG:], 0.0, None)
        fl = (nxt != state).float().mean().item()
        per_img = (nxt != state).any(1).float().mean().item()
        flips.append((dose, fl, per_img))
        log(f"  dose {dose}: flipped addresses {fl:.5f} "
            f"({fl * jl.NU:.2f}/226 per image), images touched "
            f"{per_img:.4f}")
        state = nxt

    torch.save(dict(seeds=seeds_out,
                    distinct=dict(joint=len(uj), backbone=len(ug),
                                  n=len(pj)),
                    nn=dict(gen=d_gen, real=d_real), flips=flips,
                    seed_ids=dict(a=SEEDS_A, b=SEED_B)),
               artifacts.path("reporting_results"))
    log("\nsaved: reporting-results.pt")
    fh.close()


if __name__ == "__main__":
    main()
