"""The draw ablations: flat sigma, and a temperature schedule.

Paper Sec. IV D (the schedule ablations) and Sec. V A (the calibration
scope). Two ablations of the deployed backbone draw, both declared
before any result was seen:

* **flat-sigma** -- the device read is y + T*sigma(y, m)*randn, the
  state-dependent law of Eq. (4). This arm replaces sigma(y, m) with a
  constant chosen so the effective noise matches the deployed median:
  backbone unit sigma 0.29 and patch 0.165 (both from the recorded
  sigma_eff medians 2.9e-2 and 1.65e-2 at T = 0.1). The per-step
  device-sigma distribution is measured and logged so the match is
  auditable.
* **temperature schedule** -- two schedules along the 128-step
  autoregressive chain. S-lin: T linear from 0.20 (step 0) to 0.05
  (step 127). S-eff: T_m chosen per step so the effective noise is
  constant at the deployed median, clamped to [0.02, 0.5], with the
  per-step unit sigma measured from the deployed arm's own contexts.
  The patch read stays at the deployed T = 0.1 in every arm; the
  ablation is of the backbone draw.

Three seeds per arm at n = 1000, with the deployed arm run at the same
three seeds. Expected (mean over seeds; spread is max - min of critic):

    arm             critic (spread)     div   tone   stdr
    deployed        0.8350 (0.013)   0.4671  2.498   1.25
    flat-sigma      0.8417 (0.009)   0.4384  2.403   1.28
    S-lin           0.8253 (0.022)   0.5000  1.983   1.15
    S-eff           0.8330 (0.023)   0.4197  2.209   1.32

Real bar: critic 0.8860, div 0.4866. Readings: the flat-sigma critic
is a wash within seed spread, and flat sigma costs diversity -- the
state-dependence supplies dispersion, not label agreement. Neither
schedule closes the critic gap; S-lin moves tone, std-ratio, and
diversity onto the reconstruction arm's values at a small critic cost.

Cost: about ten minutes. Requires 01-05.

    python experiments/20_draw_ablations.py
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
from chill_zen.generate import draw_backbone, render               # noqa: E402
from chill_zen.levels import _sample                               # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from importlib import import_module                           # noqa: E402

load_stack = import_module("07_verdict").load_stack

N_PER = 100
SEEDS = [SEED + 950, SEED + 951, SEED + 952]
T_DEP = 0.1
SIG_FLAT_BACKBONE = 0.29      # 2.9e-2 sigma_eff median / T = 0.1
SIG_FLAT_PATCH = 0.165        # 1.65e-2 / 0.1
SCHED_LIN = torch.linspace(0.20, 0.05, 128)


@torch.no_grad()
def measure_sigma(bl, g1, lab, gen, log):
    """Deployed draw at T = 0.1, recording per-step unit-sigma medians
    at the committing address's lanes. Returns [M] medians."""
    B = len(lab)
    units = torch.full((B, bl.NU), -1, dtype=torch.long)
    units[:, 0] = lab
    med = torch.zeros(bl.M)
    for m in range(bl.M):
        yv, mm = g1._y_m(bl.build_aat(units))
        sig = g1.noise.sigma_unit(yv, mm)
        med[m] = sig.reshape(B, bl.NU, bl.S)[:, 1 + m].median()
        yv = _sample(yv, mm, T_DEP, g1.noise, gen, None)
        units[:, 1 + m] = yv.reshape(B, bl.NU, bl.S)[:, 1 + m].argmax(-1)
    q = torch.quantile(med, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    log(f"[sigma] per-step unit-sigma medians: min {q[0]:.4f} q25 {q[1]:.4f} "
        f"median {q[2]:.4f} q75 {q[3]:.4f} max {q[4]:.4f}")
    log(f"[sigma] sigma_eff at T=0.1: step0 {T_DEP * med[0]:.4f} ... "
        f"step127 {T_DEP * med[-1]:.4f}")
    return med


def run_arm(S, lab, seed, draw_kw, patch_sigma_flat, name, log):
    pl, jl = S["pl"], S["jl"]
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    g_open = draw_backbone(S["bl"], S["g1"], S["r1"], lab, gen, **draw_kw)
    none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
    p_d = pl.read_all(S["g2"], lab, g_open, none, T_DEP, gen,
                      sigma_flat=patch_sigma_flat)
    pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
    img, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
    log(f"  [{name}] seed {seed}: drawn+rendered "
        f"({time.time() - t0:.0f}s)")
    return img


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("20_draw_ablations")
    log(f"draw ablations ({time.strftime('%Y-%m-%d %H:%M')})")

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(config.BACKBONE, config.PATCH)
    critic = judge.load_critic(artifacts.path("critic"))

    lab = torch.arange(10).repeat_interleave(N_PER)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER]
                          for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    real_std = judge.fg_std(real)
    ra, rdv = judge.judge(critic, real, lab)
    log(f"real bar: critic {ra:.4f}  div {rdv:.4f}")

    sig_med = measure_sigma(S["bl"], S["g1"], lab,
                            torch.Generator().manual_seed(SEED + 949), log)
    sched_eff = (2.9e-2 / sig_med).clamp(0.02, 0.5)
    log(f"[S-eff] schedule: step0 {sched_eff[0]:.3f} ... step127 "
        f"{sched_eff[-1]:.3f} (median {sched_eff.median():.3f})")

    arms = {
        "deployed": (dict(t_gen=T_DEP), None),
        "flat-sigma": (dict(t_gen=T_DEP, sigma_flat=SIG_FLAT_BACKBONE),
                       SIG_FLAT_PATCH),
        "S-lin": (dict(t_gen=T_DEP, t_sched=SCHED_LIN), None),
        "S-eff": (dict(t_gen=T_DEP, t_sched=sched_eff), None),
    }
    results = {}
    log(f"\n{'arm':<16}{'seed':>6}{'critic':>8}{'div':>8}{'tone':>7}"
        f"{'seam':>7}{'stdr':>7}")
    for name, (draw_kw, psf) in arms.items():
        rows = []
        for seed in SEEDS:
            img = run_arm(S, lab, seed, draw_kw, psf, name, log)
            row = judge.verdict_row(critic, img, lab, real_std)
            rows.append(row)
            log(f"{name:<16}{seed:>6}{row[0]:>8.4f}{row[1]:>8.4f}"
                f"{row[2]:>7.3f}{row[3]:>7.3f}{row[4]:>7.2f}")
        t = torch.tensor(rows)
        results[name] = dict(rows=rows, mean=t.mean(0).tolist(),
                             spread=(t.max(0).values
                                     - t.min(0).values).tolist())
        m, s = results[name]["mean"], results[name]["spread"]
        log(f"{name:<16}{'mean':>6}{m[0]:>8.4f}{m[1]:>8.4f}{m[2]:>7.3f}"
            f"{m[3]:>7.3f}{m[4]:>7.2f}   spread(max-min) critic "
            f"{s[0]:.4f}")
    torch.save(dict(results=results, real=(ra, rdv),
                    sig_med=sig_med, sched_eff=sched_eff,
                    flat=(SIG_FLAT_BACKBONE, SIG_FLAT_PATCH), seeds=SEEDS),
               artifacts.path("draw_ablations"))
    log("\nsaved: draw-ablations-results.pt")
    fh.close()


if __name__ == "__main__":
    main()
