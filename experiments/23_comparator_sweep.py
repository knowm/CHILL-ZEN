"""The comparator-noise sweep on the grayscale draw (paper Sec. IV C).

The grid was declared before any result was seen. The physical hot
read:

    sigma_y^2 = (0.02 * sigma_unit(y, m; V, t_b))^2 + (v_cmp / V)^2

is the emulator's own read (`_lane.sample_read`), with read_noise
pinned at the device (0.02), evaluated at the read voltage V and the
settling-limited pulse t_b (SnCr band, energy-model convention). The
comparator term is `NoiseParams.v_cmp`, flat in y and m. No temperature
dial. Nothing in kt-ram-neural-core is changed. The read itself is
`chill_zen.physical` (`physical_read`, `draw_backbone_physical`,
`read_patches_physical`), library code so the same read drives the
one-level frontier (25) and the binary stack (24) without duplication.

Every swept level is one the comparator's 8-bit trim register reaches
exactly (100 uV floor, 20 uV step, 5.20 mV ceiling); `chill_zen.physical`
raises on any level it does not. P0 is the modeling switch -- no
comparator term at all -- not code 0, which is the quietest comparator
modeled.

Six points (V, v_n) plus the deployed T = 0.1 control, three seeds,
n = 1000, judged as in 07. Must reproduce the grayscale column of the
Sec. IV C table in NUMBERS.md row for row.

Cost: about 15 minutes. Requires 01-05.

    OMP_NUM_THREADS=8 nice -19 python experiments/23_comparator_sweep.py
"""
import os
import pathlib
import sys
import time
from importlib import import_module

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
assert not any("drafts" in p for p in sys.path), "R1: no drafts imports"

from chill_zen import artifacts, config, judge                     # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import draw_backbone, render                # noqa: E402
from chill_zen.physical import (draw_backbone_physical, log_comp,  # noqa: E402
                                read_patches_physical, register_level)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
load_stack = import_module("07_verdict").load_stack

N_PER = 100
SEEDS = [SEED + 950, SEED + 951, SEED + 952]
T_DEP = 0.1
POINTS = {                            # name: (V, v_n) -- v_n on the register
    "P0 10mV/0":      (0.010, 0.0),
    "P1 10mV/160uV":  (0.010, 160e-6),
    "P2 10mV/500uV":  (0.010, 500e-6),
    "P3 10mV/1mV":    (0.010, 1e-3),
    "P4 10mV/2mV":    (0.010, 2e-3),
    "P5 50mV/5mV":    (0.050, 5e-3),
}
LOG_CTX = (0, 16, 64, 127)


def run_point(S, lab, seed, v_read, v_n):
    pl, jl = S["pl"], S["jl"]
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    g_open, comp = draw_backbone_physical(S["bl"], S["g1"], S["r1"], lab, gen,
                                          v_read, v_n, log_ctx=LOG_CTX)
    p_d, psh = read_patches_physical(pl, S["g2"], lab, g_open, gen, v_read, v_n)
    pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
    img, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
    comp["patch"] = psh
    return img, comp, time.time() - t0


def run_deployed(S, lab, seed):
    pl, jl = S["pl"], S["jl"]
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    g_open = draw_backbone(S["bl"], S["g1"], S["r1"], lab, gen, t_gen=T_DEP)
    none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
    p_d = pl.read_all(S["g2"], lab, g_open, none, T_DEP, gen)
    pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
    img, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
    return img, time.time() - t0


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("23_comparator_sweep")
    log(f"\n=== 23_comparator_sweep.py ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    log("read_noise 0.02 (device), band SnCr, t_b = max(2 ns, 5RC) per read, "
        "emulator sample_read: sigma^2 = lane^2 + (v_cmp/V)^2; "
        "v_cmp on the 8-bit register (100 uV + code * 20 uV)")
    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(config.BACKBONE, config.PATCH)
    critic = judge.load_critic(artifacts.path("critic"))
    log(f"bank noise params: {S['g1'].noise.state()}")

    lab = torch.arange(10).repeat_interleave(N_PER)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER] for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    real_std = judge.fg_std(real)
    ra, rdv = judge.judge(critic, real, lab)
    log(f"real bar: critic {ra:.4f}  div {rdv:.4f}")

    results = {}
    hdr = (f"{'point':<16}{'seed':>6}{'critic':>8}{'div':>8}{'tone':>7}"
           f"{'seam':>7}{'stdr':>7}")

    def finish(name, rows):
        t = torch.tensor(rows)
        results[name] = dict(rows=rows, mean=t.mean(0).tolist(),
                             spread=(t.max(0).values - t.min(0).values).tolist())

    log(f"\n{hdr}")
    log("deployed control (T = 0.1 as run)")
    rows = []
    for seed in SEEDS:
        img, dt = run_deployed(S, lab, seed)
        row = judge.verdict_row(critic, img, lab, real_std)
        rows.append(row)
        log(f"{'deployed':<16}{seed:>6}{row[0]:>8.4f}{row[1]:>8.4f}{row[2]:>7.3f}"
            f"{row[3]:>7.3f}{row[4]:>7.2f}")
    finish("deployed", rows)

    for name, (v_read, v_n) in POINTS.items():
        code, v_cmp = register_level(v_n)
        log(f"\n{name}: V = {v_read * 1e3:.0f} mV, v_n = {v_cmp * 1e6:.0f} uV "
            f"(code {code if code is not None else '-- (not modeled)'}), "
            f"v_n/V = {v_cmp / v_read:.3f}")
        rows = []
        comp = None
        for seed in SEEDS:
            img, c, dt = run_point(S, lab, seed, v_read, v_n)
            comp = comp or c
            row = judge.verdict_row(critic, img, lab, real_std)
            rows.append(row)
            log(f"{name:<16}{seed:>6}{row[0]:>8.4f}{row[1]:>8.4f}{row[2]:>7.3f}"
                f"{row[3]:>7.3f}{row[4]:>7.2f}")
        finish(name, rows)
        results[name]["comp"] = comp
        log_comp(comp, log)

    log("\nsummary (mean over seeds)")
    log(f"{'point':<16}{'code':>6}{'v_n/V':>7}{'critic':>8}{'div':>8}{'tone':>7}"
        f"{'seam':>7}{'stdr':>7}")
    for name, r in results.items():
        if name in POINTS:
            code, v_cmp = register_level(POINTS[name][1])
            vnv, cs = v_cmp / POINTS[name][0], ("--" if code is None else str(code))
        else:
            vnv, cs = float("nan"), ""
        m = r["mean"]
        log(f"{name:<16}{cs:>6}{vnv:>7.3f}{m[0]:>8.4f}{m[1]:>8.4f}{m[2]:>7.3f}"
            f"{m[3]:>7.3f}{m[4]:>7.2f}")
    torch.save(dict(results=results, real=(ra, rdv), points=POINTS, seeds=SEEDS,
                    gain=0.02, band="SnCr"), artifacts.path("comparator_sweep"))
    log("saved: comparator-sweep-results.pt")
    fh.close()


if __name__ == "__main__":
    main()
