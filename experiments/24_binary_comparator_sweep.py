"""The comparator-noise sweep on the binary two-level stack (Sec. IV C).

Same physical read as 23 (`chill_zen.physical`: the emulator's own
`sample_read` at read_noise 0.02, evaluated at (V, t_b), SnCr settling,
the comparator term as `NoiseParams.v_cmp`), on the binary two-level
stack (`128x16@p0.5` backbone, `2x16@p0.5` patch). Six points P0-P5 plus
the deployed control (backbone T = 0.3, patch T = 0.1); `--extend` adds
P6-P7 (v_n/V = 0.3, 0.5), declared after the first grid did not turn
over. Every level is one the comparator's 8-bit register reaches exactly
(100 uV floor, 20 uV step, 5.20 mV ceiling).

Part 1, screen: n = 1000, three seeds, the binary-verdict judge on
0.5-midpoint binarized renders. Part 2, render: n = 5120 per point into
`head_to_head/gen/comparator-two-level-*.npy` for
`head_to_head/07_comparator_score.py` (the head-to-head venv). Must
reproduce the binary column of the Sec. IV C table in NUMBERS.md, P0-P5
plus the P6-P7 rows that `--extend` adds.

Cost: about 20 minutes (screen + render, emulator venv only; FID
scoring is a separate step). Requires 17-19.

    OMP_NUM_THREADS=8 nice -19 python experiments/24_binary_comparator_sweep.py [--extend]
"""
import argparse
import os
import pathlib
import sys
import time
from importlib import import_module

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
assert not any("drafts" in p for p in sys.path), "R1: no drafts imports"

from chill_zen import artifacts, judge                             # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import draw_backbone                       # noqa: E402
from chill_zen.physical import (draw_backbone_physical, log_comp,  # noqa: E402
                                read_patches_physical, register_level)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
bv = import_module("19_binary_verdict")
load_binary_stack, bin_render_two = bv.load_binary_stack, bv.bin_render_two
SRC_THRESH = bv.SRC_THRESH

GEN = pathlib.Path(__file__).resolve().parents[1] / \
    "experiments" / "head_to_head" / "gen"
POINTS = {                            # name: (V, v_n) -- same grid as 23
    "P0 10mV/0":      (0.010, 0.0),
    "P1 10mV/160uV":  (0.010, 160e-6),
    "P2 10mV/500uV":  (0.010, 500e-6),
    "P3 10mV/1mV":    (0.010, 1e-3),
    "P4 10mV/2mV":    (0.010, 2e-3),
    "P5 50mV/5mV":    (0.050, 5e-3),
}
EXTEND = {                            # indices continue from P5
    "P6 10mV/3mV":    (0.010, 3e-3),
    "P7 10mV/5mV":    (0.010, 5e-3),
}
# The grid stops at P7. A ninth point at v_n = 7.5 mV was swept before the
# comparator level was a register; 7.5 mV is above the register's ceiling
# (code 255 = 5.20 mV), so it is not a level a comparator could be
# programmed to and it is not reported. The turnover it was declared to
# find is already in P6 -> P7. Render seeds are keyed on the point's index
# (SEED_BLOCK + 31 * i), so dropping the last point leaves P6 and P7
# untouched.
T_BACKBONE, T_PATCH = 0.3, 0.1        # the deployed arm
N_PER_SCREEN, N_PER_SCORE = 100, 512
SEEDS = [SEED + 950, SEED + 951, SEED + 952]
SEED_BLOCK = SEED + 22000
DIV_FRAC = 0.75


def states_physical(S, lab, gen, v_read, v_n):
    pl, jl = S["pl"], S["jl"]
    g_open, comp = draw_backbone_physical(S["bl"], S["g1"], S["r1"], lab, gen,
                                          v_read, v_n)
    p_d, psh = read_patches_physical(pl, S["g2"], lab, g_open, gen, v_read, v_n)
    pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
    comp["patch"] = psh
    return pj, comp


def states_deployed(S, lab, gen):
    pl, jl = S["pl"], S["jl"]
    g_open = draw_backbone(S["bl"], S["g1"], S["r1"], lab, gen, t_gen=T_BACKBONE)
    none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
    p_d = pl.read_all(S["g2"], lab, g_open, none, T_PATCH, gen)
    return jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None), None


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", action="store_true")
    args = ap.parse_args()
    log, fh = artifacts.make_log("24_binary_comparator_sweep")
    log(f"\n=== 24_binary_comparator_sweep.py "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")
    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    S = load_binary_stack(y_tr, y_ev)
    jl = S["jl"]
    critic = judge.load_critic(artifacts.path("critic"))

    lab = torch.arange(10).repeat_interleave(N_PER_SCREEN)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER_SCREEN] for c in range(10)])
    real = (X[N_TRAIN:, 0][real_idx].clamp(0, 1) > SRC_THRESH).float()
    ra, rdv = judge.judge(critic, real, lab)
    log(f"real bar (0.1-binarized eval, n = {len(lab)}): critic {ra:.4f}  "
        f"div {rdv:.4f}  seam {judge.seam_ratio(real):.3f}")

    arms = dict(EXTEND) if args.extend else {"deployed T=0.3/0.1": None, **POINTS}
    offset = 1 + len(POINTS) if args.extend else 0
    if args.extend:
        log("[extend] P6-P7, grid extension to find the turnover")
    results = {}
    log(f"\n--- screen, n = 1000, three seeds ---")
    log(f"{'point':<20}{'seed':>6}{'critic':>8}{'div':>8}{'seam':>7}")
    for name, pt in arms.items():
        if pt is not None:
            code, v_cmp = register_level(pt[1])
            log(f"{name}: V = {pt[0] * 1e3:.0f} mV, v_n = {v_cmp * 1e6:.0f} uV "
                f"(code {code if code is not None else '-- (not modeled)'}), "
                f"v_n/V = {v_cmp / pt[0]:.3f}")
        rows, comp = [], None
        for seed in SEEDS:
            gen = torch.Generator().manual_seed(seed)
            t0 = time.time()
            if pt is None:
                pj, c = states_deployed(S, lab, gen)
            else:
                pj, c = states_physical(S, lab, gen, *pt)
            comp = comp or c
            img = bin_render_two(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
            a, dv = judge.judge(critic, img, lab)
            row = (a, dv, judge.seam_ratio(img))
            rows.append(row)
            log(f"{name:<20}{seed:>6}{row[0]:>8.4f}{row[1]:>8.4f}{row[2]:>7.3f}"
                f"   ({time.time() - t0:.0f}s)")
        t = torch.tensor(rows)
        m = t.mean(0).tolist()
        sp = (t.max(0).values - t.min(0).values).tolist()
        passed = m[1] >= DIV_FRAC * rdv
        results[name] = dict(rows=rows, mean=m, spread=sp, comp=comp,
                             div_pass=passed)
        log(f"{name:<20}{'mean':>6}{m[0]:>8.4f}{m[1]:>8.4f}{m[2]:>7.3f}"
            f"   spread critic {sp[0]:.4f}   div screen "
            f"{'PASS' if passed else 'FAIL'}")
        if comp:
            log_comp(comp, log)

    log(f"\n--- render, n = {10 * N_PER_SCORE}, for 07_comparator_score.py ---")
    GEN.mkdir(exist_ok=True)
    lab5 = torch.arange(10).repeat_interleave(N_PER_SCORE)
    for i, (name, pt) in enumerate(arms.items(), start=offset):
        gen = torch.Generator().manual_seed(SEED_BLOCK + 31 * i)
        t0 = time.time()
        pj, _ = states_deployed(S, lab5, gen) if pt is None else \
            states_physical(S, lab5, gen, *pt)
        img = bin_render_two(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
        tag = "comparator-two-level-" + name.split()[0]
        np.save(GEN / f"{tag}.npy", img.numpy().astype(np.float32))
        np.save(GEN / f"{tag}-lab.npy", lab5.numpy().astype(np.int64))
        log(f"[gen] {tag}: {tuple(img.shape)}  on-fraction {img.mean():.4f}  "
            f"({time.time() - t0:.0f}s)")

    torch.save(dict(results=results, real=(ra, rdv), points=arms, seeds=SEEDS),
               artifacts.path("binary_comparator_sweep_ext" if args.extend
                              else "binary_comparator_sweep"))
    log("saved; renders in head_to_head/gen/")
    fh.close()


if __name__ == "__main__":
    main()
