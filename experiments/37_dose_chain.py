"""The joint sweep as a chain, with and without read noise (Sec. III B).

A11 of the hostile review: one joint dose is
a reconciliation, not a fixed point (dose 2 flips 82 of 226 addresses,
dose 3 44, dose 4 26). The predecessor pipeline found that extra doses
collapsed renders toward class prototypes, but its sweeps were
noiseless. This script asks two things on the deployed banks:

  1. What do doses 1..8 of the noiseless joint sweep do to the judge?
  2. Does read noise in the sweep change that -- does a noisy chain
     settle to a stationary churn instead of collapsing?

Same hot draws (backbone draw + R1 + patch read at the operating point,
10 mV / 2 mV) for every condition; then the joint sweep is iterated
under one read condition, from the deployed noiseless argmax through
the physical floor (50 mV, code 0) up to the draw's own operating point
(10 mV, code 95). After each dose: judge row (critic, div, tone,
std-ratio), addresses flipped by that dose, images unchanged by it,
distinct joint codes, and the within-class mean pairwise L2 between
renders (a graded diversity; `div` saturates). Dose 0 is the state
before any joint sweep. The reconstruction arm (real backbone codes)
runs the same chain under two conditions as a control for collapse.

Cost: about 15 minutes at n = 1000 and three seeds. Requires 01-05.

    python experiments/37_dose_chain.py
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
from chill_zen.physical import (draw_backbone_physical,            # noqa: E402
                                read_joint_physical,
                                read_patches_physical, register_level)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
load_stack = import_module("07_verdict").load_stack

N_PER = 100
SEEDS = [SEED + 960, SEED + 961, SEED + 962]     # 21_reporting's seeds
V_HOT, V_N_HOT = 0.010, 2e-3                     # the operating point
DOSES = 8
# sweep read conditions: name -> (V_read, v_n); None = noiseless argmax
CONDS = {
    "noiseless (deployed)": (None, None),
    "50 mV, code 0   (v_n/V 0.002)": (0.050, 100e-6),
    "50 mV, code 45  (v_n/V 0.02)": (0.050, 1e-3),
    "50 mV, code 245 (v_n/V 0.10)": (0.050, 5e-3),
    "10 mV, code 45  (v_n/V 0.10)": (0.010, 1e-3),
    "10 mV, code 95  (v_n/V 0.20)": (0.010, 2e-3),
}
RECON_CONDS = ["noiseless (deployed)", "10 mV, code 95  (v_n/V 0.20)"]
# --quench: noisy chains, each dose followed by one cold dose, judged both
# ways. Asks whether "relax hot, finish cold" beats the single cold dose.
QUENCH_CONDS = {
    "50 mV, code 95  (v_n/V 0.04)": (0.050, 2e-3),
    "50 mV, code 145 (v_n/V 0.06)": (0.050, 3e-3),
    "50 mV, code 245 (v_n/V 0.10)": (0.050, 5e-3),
}


def sweep(S, lab, g, p, cond, gen):
    jl = S["jl"]
    v_read, v_n = cond
    if v_n is None:
        return jl.read_joint(S["jbank"], lab, g, p, 0.0, None)
    return read_joint_physical(jl, S["jbank"], lab, g, p, v_read, v_n, gen)


@torch.no_grad()
def l2_div(img, y):
    """Within-class mean pairwise L2 between renders, class-averaged."""
    out = torch.zeros(10)
    for c in range(10):
        s = img[y == c].reshape(-1, 784)
        d = torch.cdist(s, s)
        iu = torch.triu_indices(len(s), len(s), offset=1)
        out[c] = d[iu[0], iu[1]].mean()
    return out.mean().item()


def measure(S, critic, lab, real_std, g, p):
    jl = S["jl"]
    img, _ = render(g, p, S["gb"], S["kb"])
    a, dv, tone, _, stdr = judge.verdict_row(critic, img, lab, real_std)
    codes = torch.cat([g, p], 1)
    distinct = len({tuple(r.tolist()) for r in codes}) / len(codes)
    return dict(critic=a, div=dv, tone=tone, stdr=stdr,
                l2=l2_div(img, lab), distinct=distinct)


def run_chain(S, critic, lab, real_std, g0, p0, cond, gen, log, tag,
              quench=False):
    jl = S["jl"]
    state = torch.cat([g0, p0], 1)
    rows = [dict(dose=0, flips=float("nan"), fixed=float("nan"),
                 **measure(S, critic, lab, real_std, g0, p0))]
    log(f"    {'dose':>4}{'critic':>8}{'div':>8}{'tone':>7}{'stdr':>6}"
        f"{'L2div':>7}{'distinct':>9}{'flips/226':>10}{'unchanged':>10}"
        + (f"{'| quenched:':>12}{'critic':>8}{'div':>8}{'tone':>7}{'stdr':>6}"
           f"{'L2div':>7}{'flips':>7}" if quench else ""))
    r = rows[0]
    log(f"    {0:>4}{r['critic']:>8.4f}{r['div']:>8.4f}{r['tone']:>7.3f}"
        f"{r['stdr']:>6.2f}{r['l2']:>7.3f}{r['distinct']:>9.3f}"
        f"{'--':>10}{'--':>10}")
    for dose in range(1, DOSES + 1):
        nxt = sweep(S, lab, state[:, :jl.NG], state[:, jl.NG:], cond, gen)
        ch = nxt != state
        flips = ch.float().mean().item() * jl.NU
        fixed = 1.0 - ch.any(1).float().mean().item()
        r = dict(dose=dose, flips=flips, fixed=fixed,
                 **measure(S, critic, lab, real_std, nxt[:, :jl.NG],
                           nxt[:, jl.NG:]))
        rows.append(r)
        line = (f"    {dose:>4}{r['critic']:>8.4f}{r['div']:>8.4f}{r['tone']:>7.3f}"
                f"{r['stdr']:>6.2f}{r['l2']:>7.3f}{r['distinct']:>9.3f}"
                f"{flips:>10.1f}{fixed:>10.3f}")
        if quench:
            q = sweep(S, lab, nxt[:, :jl.NG], nxt[:, jl.NG:], (None, None), None)
            qf = (q != nxt).float().mean().item() * jl.NU
            qr = measure(S, critic, lab, real_std, q[:, :jl.NG], q[:, jl.NG:])
            r["quench"] = dict(flips=qf, **qr)
            line += (f"{'|':>12}{qr['critic']:>8.4f}{qr['div']:>8.4f}"
                     f"{qr['tone']:>7.3f}{qr['stdr']:>6.2f}{qr['l2']:>7.3f}"
                     f"{qf:>7.1f}")
        log(line)
        state = nxt
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=N_PER)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--quench", action="store_true",
                    help="noisy chains with a cold dose after each dose")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("37_dose_chain"
                                 + ("_quench" if args.quench else ""))
    conds = QUENCH_CONDS if args.quench else CONDS
    recon_conds = [] if args.quench else RECON_CONDS
    log(f"\n=== 37_dose_chain.py ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    log(f"hot reads at ({V_HOT * 1e3:.0f} mV, {V_N_HOT * 1e3:.0f} mV), code "
        f"{register_level(V_N_HOT)[0]}; joint sweep chained {DOSES} doses "
        f"under: " + "; ".join(conds)
        + ("; each dose followed by one cold dose (quench)" if args.quench
           else ""))

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(config.BACKBONE, config.PATCH)
    pl, jl = S["pl"], S["jl"]
    critic = judge.load_critic(artifacts.path("critic"))
    n_per = args.n_per
    lab = torch.arange(10).repeat_interleave(n_per)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:n_per] for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    real_std = judge.fg_std(real)
    g_real = S["gcodes"]["ev"].long()[real_idx]
    ra, rdv = judge.judge(critic, real, lab)
    log(f"real bar: critic {ra:.4f}  div {rdv:.4f}  tone "
        f"{judge.tone_spread(real, lab):.3f}  L2div {l2_div(real, lab):.3f}"
        f"  n = {len(lab)}")

    results = {}
    for seed in SEEDS[:args.seeds]:
        t0 = time.time()
        gen = torch.Generator().manual_seed(seed)
        g_open, _ = draw_backbone_physical(S["bl"], S["g1"], S["r1"], lab,
                                           gen, V_HOT, V_N_HOT)
        p_d, _ = read_patches_physical(pl, S["g2"], lab, g_open, gen,
                                       V_HOT, V_N_HOT)
        p_r, _ = read_patches_physical(pl, S["g2"], lab, g_real, gen,
                                       V_HOT, V_N_HOT)
        log(f"\nseed {seed}: hot pass {time.time() - t0:.0f}s")
        for name, cond in conds.items():
            log(f"  end-to-end, sweep read {name}")
            gen_c = torch.Generator().manual_seed(seed + 31)
            results[("e2e", name, seed)] = run_chain(
                S, critic, lab, real_std, g_open, p_d, cond, gen_c, log, name,
                quench=args.quench)
        for name in recon_conds:
            log(f"  reconstruction, sweep read {name}")
            gen_c = torch.Generator().manual_seed(seed + 31)
            results[("recon", name, seed)] = run_chain(
                S, critic, lab, real_std, g_real, p_r, conds[name], gen_c,
                log, name)
        log(f"  seed {seed} done ({time.time() - t0:.0f}s)")

    # summary: mean over seeds, dose 1 vs dose DOSES, per condition
    log("\nsummary over seeds (mean): dose 1 -> dose "
        f"{DOSES}; delta is paired within seed")
    log(f"{'arm':<6}{'sweep read':<32}{'critic 1':>9}{'critic 8':>9}"
        f"{'delta':>8}{'div 1':>7}{'div 8':>7}{'L2 1':>7}{'L2 8':>7}"
        f"{'stdr 8':>7}{'flips@8':>8}")
    summary = {}
    for arm in ("e2e", "recon"):
        for name in conds:
            keys = [k for k in results if k[0] == arm and k[1] == name]
            if not keys:
                continue
            r1 = torch.tensor([[results[k][1][f] for f in
                                ("critic", "div", "l2", "stdr")] for k in keys])
            r8 = torch.tensor([[results[k][DOSES][f] for f in
                                ("critic", "div", "l2", "stdr")] for k in keys])
            fl = torch.tensor([results[k][DOSES]["flips"] for k in keys])
            d = (r8[:, 0] - r1[:, 0])
            summary[(arm, name)] = dict(d1=r1.mean(0).tolist(),
                                        d8=r8.mean(0).tolist(),
                                        delta=d.mean().item(),
                                        delta_spread=(d.max() - d.min()).item(),
                                        flips8=fl.mean().item())
            log(f"{arm:<6}{name:<32}{r1[:, 0].mean():>9.4f}"
                f"{r8[:, 0].mean():>9.4f}{d.mean():>+8.4f}"
                f"{r1[:, 1].mean():>7.4f}{r8[:, 1].mean():>7.4f}"
                f"{r1[:, 2].mean():>7.3f}{r8[:, 2].mean():>7.3f}"
                f"{r8[:, 3].mean():>7.2f}{fl.mean():>8.1f}")

    if args.quench:
        log("\nquenched (noisy doses then one cold dose), mean over seeds:")
        log(f"{'sweep read':<32}" + "".join(f"{'d' + str(d):>9}" for d in range(1, DOSES + 1)))
        for name in conds:
            keys = [k for k in results if k[1] == name]
            qc = [sum(results[k][d]["quench"]["critic"] for k in keys) / len(keys)
                  for d in range(1, DOSES + 1)]
            log(f"{name:<32}" + "".join(f"{v:>9.4f}" for v in qc))
    torch.save(dict(seeds=SEEDS[:args.seeds], n_per=n_per, conds=conds,
                    doses=DOSES,
                    results={f"{a}|{c}|{s}": v for (a, c, s), v in results.items()},
                    summary={f"{a}|{c}": v for (a, c), v in summary.items()}),
               artifacts.path("verdict").parent
               / ("dose-chain-quench.pt" if args.quench else "dose-chain.pt"))
    log("\nsaved: dose-chain" + ("-quench" if args.quench else "") + ".pt")
    fh.close()


if __name__ == "__main__":
    main()
