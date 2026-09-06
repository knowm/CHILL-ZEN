"""The cold-read control: the cold sweeps at the physical floor.

Paper Sec. II A and Limitation 3 treat a cold read as T = 0. The deployed
generation implements that literally: the R1 sweep and the joint sweep
are noiseless argmax reads, with no device term and no comparator term.
Physically a cold read sits at the flicker floor of the devices, at the
settling-limited thermal term, and at the comparator's own floor, the
register's code 0 (100 uV, or v_n/V = 0.002 at the 50 mV cold read).
The paper's justification, a median winner margin about 42 times the
flicker floor, says nothing about the tail: on the shipped states some
8% of the joint sweep's addresses sit within three floors of a tie.

This script runs the deployed two-level generation on the same hot
draws under two cold-read conditions and reports the difference:

  * reported  -- both cold sweeps as noiseless argmax (as deployed);
  * physical  -- both cold sweeps through `chill_zen.physical` at
                 50 mV, device terms on at the settling-limited pulse,
                 comparator at register code 0, 10 and 45
                 (100 uV, 300 uV, 1 mV; v_n/V = 0.002, 0.006, 0.02).

The hot reads are identical across conditions (their own generator,
same seed), so every difference is the cold sweeps'. Per seed and per
cold level it reports the judge row of the end-to-end and the
reconstruction arm, the fraction of addresses whose verdict changes
when the *same* input is swept with noise (R1 and joint separately),
and the R1 and joint winner-margin tails. The reconstruction arm here
draws its patch level through the physical read, as the end-to-end arm
does, rather than through the fixed-gain emulator read of
`21_reporting.py`.

Cost: about five minutes at n = 1000 and three seeds. Requires 01-05.

    python experiments/35_cold_noise_control.py
"""
import argparse
import os
import pathlib
import sys
import time
from importlib import import_module

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config, judge                     # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import render                              # noqa: E402
from chill_zen.physical import (draw_backbone_hot_only,            # noqa: E402
                                read_all_backbone_physical,
                                read_joint_physical,
                                read_patches_physical, register_level)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
load_stack = import_module("07_verdict").load_stack

N_PER = 100
SEEDS = [SEED + 990, SEED + 991, SEED + 992]
V_HOT, V_N_HOT = 0.010, 2e-3          # the operating point, code 95
V_COLD = 0.050
COLD_LEVELS = {"code 0 (100uV)": 100e-6,
               "code 10 (300uV)": 300e-6,
               "code 45 (1mV)": 1e-3}
# the combined physical floor at code 0, referred to the 50 mV read:
# flicker ~2.2e-3 (gain 0.02, joint pool) in quadrature with 2.0e-3
FLOOR = 3.0e-3


def margins(bank, aat, nu, S=16):
    """Winner margin (top1 - top2) per group of the noiseless read."""
    yv, _ = bank._y_m(aat)
    top2 = yv.reshape(len(aat), nu, S).topk(2, dim=-1).values
    return (top2[..., 0] - top2[..., 1]).flatten()


def tail(m):
    q = torch.quantile(m.double(), torch.tensor([0.01, 0.05, 0.5], dtype=torch.float64))
    return (f"median {q[2]:.4f}  p5 {q[1]:.4f}  p1 {q[0]:.4f}  "
            f"< 1x floor {(m < FLOOR).float().mean():.4f}  "
            f"< 3x {(m < 3 * FLOOR).float().mean():.4f}")


def r1_sweep(S, lab, units, v_n, gen):
    """The cold R1 sweep under one condition. v_n None = noiseless."""
    bl = S["bl"]
    if v_n is None:
        g = bl.read_all(S["r1"], units, 0.0, None)
    else:
        g = read_all_backbone_physical(bl, S["r1"], units, V_COLD, v_n, gen)
    g[:, 0] = lab
    return g[:, 1:]


def joint_sweep(S, lab, g, p, v_n, gen):
    """The cold joint sweep under one condition. v_n None = noiseless."""
    jl = S["jl"]
    if v_n is None:
        return jl.read_joint(S["jbank"], lab, g, p, 0.0, None)
    return read_joint_physical(jl, S["jbank"], lab, g, p, V_COLD, v_n, gen)


def sweep(S, lab, units, g_open, p_d, v_n, gen):
    """Both cold sweeps under one condition, each on the given inputs.
    Returns (g after R1, joint verdict)."""
    return (r1_sweep(S, lab, units, v_n, gen),
            joint_sweep(S, lab, g_open, p_d, v_n, gen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=N_PER)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("35_cold_noise_control")
    log(f"\n=== 35_cold_noise_control.py ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    log(f"hot reads at ({V_HOT * 1e3:.0f} mV, {V_N_HOT * 1e3:.0f} mV), "
        f"code {register_level(V_N_HOT)[0]}; cold sweeps at {V_COLD * 1e3:.0f} "
        f"mV, device terms on, settling-limited pulse, comparator at "
        + ", ".join(f"{k}" for k in COLD_LEVELS) + f"; floor {FLOOR:.1e}")

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(config.BACKBONE, config.PATCH)
    bl, pl, jl = S["bl"], S["pl"], S["jl"]
    critic = judge.load_critic(artifacts.path("critic"))
    n_per = args.n_per
    lab = torch.arange(10).repeat_interleave(n_per)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:n_per] for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    real_std = judge.fg_std(real)
    g_real = S["gcodes"]["ev"].long()[real_idx]
    ra, rdv = judge.judge(critic, real, lab)
    log(f"real bar: critic {ra:.4f}  div {rdv:.4f}  n = {len(lab)}")

    conds = [("reported", None)] + list(COLD_LEVELS.items())
    # end-to-end: noisy joint sweep on the reported (g, p); recon: the
    # same on the real-code arm; chain: the whole cold path with noise,
    # noisy R1 -> patch read on that g -> noisy joint.
    rows = {arm: {c: [] for c, _ in conds}
            for arm in ("end-to-end", "recon", "chain")}
    flips = {c: dict(r1=[], joint=[], joint_rec=[]) for c, _ in conds}
    hdr = f"{'arm':<12}{'cold read':<18}{'seed':>6}{'critic':>8}{'div':>8}{'tone':>7}{'stdr':>7}"

    for seed in SEEDS:
        t0 = time.time()
        gen_hot = torch.Generator().manual_seed(seed)
        units = draw_backbone_hot_only(bl, S["g1"], lab, gen_hot, V_HOT, V_N_HOT)
        # margins of the R1 sweep's input, never measured before
        log(f"\nseed {seed}: hot pass {time.time() - t0:.0f}s")
        log("  R1 sweep margins   " + tail(margins(S["r1"], bl.build_aat(units), bl.NU)))

        # the reported condition fixes g_open and p_d for every condition
        g_open = r1_sweep(S, lab, units, None, None)
        gen_p = torch.Generator().manual_seed(seed + 7)
        p_d, _ = read_patches_physical(pl, S["g2"], lab, g_open, gen_p, V_HOT, V_N_HOT)
        gen_p = torch.Generator().manual_seed(seed + 7)
        p_r, _ = read_patches_physical(pl, S["g2"], lab, g_real, gen_p, V_HOT, V_N_HOT)
        log("  joint sweep margins" + tail(margins(S["jbank"], pl.build_aat(lab, g_open, p_d), jl.NU)))

        log("  " + hdr)
        ref = {}
        for name, v_n in conds:
            gen_c = torch.Generator().manual_seed(seed + 11)
            g_c, pj = sweep(S, lab, units, g_open, p_d, v_n, gen_c)
            _, pjr = sweep(S, lab, units, g_real, p_r, v_n, gen_c)
            if v_n is None:
                ref = dict(g=g_c, pj=pj, pjr=pjr)
                pj_chain = pj
            else:
                flips[name]["r1"].append((g_c != ref["g"]).float().mean().item())
                flips[name]["joint"].append((pj != ref["pj"]).float().mean().item())
                flips[name]["joint_rec"].append((pjr != ref["pjr"]).float().mean().item())
                gen_p = torch.Generator().manual_seed(seed + 7)
                p_c, _ = read_patches_physical(pl, S["g2"], lab, g_c, gen_p,
                                               V_HOT, V_N_HOT)
                pj_chain = joint_sweep(S, lab, g_c, p_c, v_n, gen_c)
            for arm, st in (("end-to-end", pj), ("recon", pjr),
                            ("chain", pj_chain)):
                img, _ = render(st[:, :jl.NG], st[:, jl.NG:], S["gb"], S["kb"])
                row = judge.verdict_row(critic, img, lab, real_std)
                rows[arm][name].append(row)
                log(f"  {arm:<12}{name:<18}{seed:>6}{row[0]:>8.4f}{row[1]:>8.4f}"
                    f"{row[2]:>7.3f}{row[4]:>7.2f}")

    log("\nsummary over seeds (mean +- max-min):")
    log(f"{'arm':<12}{'cold read':<18}{'critic':>18}{'div':>18}{'tone':>8}{'stdr':>6}")
    out = {}
    for arm in rows:
        for name, _ in conds:
            t = torch.tensor(rows[arm][name])
            mu, sp = t.mean(0), t.max(0).values - t.min(0).values
            out[(arm, name)] = dict(mean=mu.tolist(), spread=sp.tolist())
            log(f"{arm:<12}{name:<18}{mu[0]:>9.4f} +-{sp[0]:.4f}"
                f"{mu[1]:>9.4f} +-{sp[1]:.4f}{mu[2]:>8.3f}{mu[4]:>6.2f}")
    log("\naddresses whose verdict changes on the SAME input when the sweep "
        "runs with noise (mean over seeds):")
    log(f"{'cold read':<18}{'R1 (of 128)':>14}{'joint e2e (of 226)':>20}{'joint recon':>14}")
    for name, _ in COLD_LEVELS.items():
        f = flips[name]
        r1, jt, jr = (sum(f[k]) / len(f[k]) for k in ("r1", "joint", "joint_rec"))
        log(f"{name:<18}{r1:>8.4f} ({r1 * bl.M:4.1f}){jt:>10.4f} ({jt * jl.NU:5.1f})"
            f"{jr:>8.4f} ({jr * jl.NU:5.1f})")
        out[("flips", name)] = dict(r1=r1, joint=jt, joint_rec=jr)
    torch.save(dict(seeds=SEEDS, n_per=n_per, v_hot=(V_HOT, V_N_HOT),
                    v_cold=V_COLD, levels=COLD_LEVELS, floor=FLOOR,
                    rows={f"{a}|{c}": v for (a, c), v in out.items()}),
               artifacts.path("verdict").parent / "cold-noise-control.pt")
    log("\nsaved: cold-noise-control.pt")
    fh.close()


if __name__ == "__main__":
    main()
