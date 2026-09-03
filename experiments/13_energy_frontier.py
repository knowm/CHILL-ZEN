"""Per-image energy for each point on the one-level frontier.

Paper Sec. V B and Fig. 9. The same model as experiment 12, evaluated
for the backbone-only systems at 16, 32, 64 and 128 books, whose census
per image is M x 16 hot autoregressive reads (the context grows from 6
to 6 + M selected addresses along the chain) plus M x 16 cold sweep
reads. The deployed two-level point is carried over from experiment 12.

Expected, joules per image (the bounded column is the SnCr band; on
tungsten the 64- and 128-book points are 4.68e-12 and 1.47e-11, and
the two-level point 6.85e-11):

    system              bounded    SnCr anchor    W anchor
    16 books           8.65e-13       5.81e-12    1.20e-10
    32 books           1.73e-12       2.12e-11    4.49e-10
    64 books           3.47e-12       7.97e-11    1.72e-09
    128 books          6.97e-12       3.28e-10    7.08e-09
    two-level          1.87e-11       1.58e-09    3.37e-08

Cost: a couple of minutes. Requires 03, 06 and 12.

    python experiments/13_energy_frontier.py
"""
import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.energy import (BANDS, C_LINE, E_ADDR, E_CMP, KBT,   # noqa: E402
                         LREP, S, V_COLD, V_FLOOR, V_LV, PW_ANCHOR,
                         g_sum_from_bytes, pooled_read, sigma_target,
                         t_read)
from chill_zen.levels import parse_tag                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="*", default=config.FRONTIER)
    args = ap.parse_args()
    log, fh = artifacts.make_log("13_energy_frontier")

    banks = torch.load(artifacts.path("backbone_banks"))
    sweep = torch.load(artifacts.path("backbone_sweep"))
    out = {}
    for tag in args.points:
        M, _, _ = parse_tag(tag)
        ga_g, gb_g = banks[tag]["g"]["ga"], banks[tag]["g"]["gb"]
        ga_r, gb_r = banks[tag]["r"]["ga"], banks[tag]["r"]["gb"]
        states = sweep["states"][tag].long()
        lab, codes = states[:100, 0], states[:100, 1:]
        KSP = LREP + M
        steps = []
        for k in range(M):
            aat = torch.full((100, KSP), -1, dtype=torch.long)
            aat[:, :LREP] = lab[:, None]
            if k > 0:
                aat[:, LREP:LREP + k] = codes[:, :k]
            lanes = slice((1 + k) * S, (2 + k) * S)
            yv, mv = pooled_read(ga_g[lanes], gb_g[lanes], aat)
            steps.append((LREP + k, mv.median().item(),
                          sigma_target(yv, mv).median().item()))
        aat = torch.cat([lab[:, None].expand(-1, LREP), codes], 1)
        _, mv = pooled_read(ga_r[S:], gb_r[S:], aat)
        m_cold = mv.median().item()

        v_hot_op = max(V_FLOOR, 0.012)
        e_drive = (sum(LREP + k for k in range(M)) * C_LINE * v_hot_op ** 2
                   + KSP * C_LINE * V_COLD ** 2)
        e_periph = e_drive + 2 * M * S * E_CMP + 2 * M * E_ADDR

        row = dict(periph=e_periph, M=M)
        for band in BANDS:
            e_b = e_a = 0.0
            for n_sel, m_med, st in steps:
                gs = g_sum_from_bytes(n_sel, m_med, band)
                t_b = t_read(gs)
                v_op = max(V_FLOOR,
                           math.sqrt(2 * KBT / (t_b * gs * st ** 2)))
                e_b += S * v_op ** 2 * gs * t_b
                e_a += S * V_LV ** 2 * gs * PW_ANCHOR
            gs = g_sum_from_bytes(KSP, m_cold, band)
            e_b += M * S * V_COLD ** 2 * gs * t_read(gs)
            e_a += M * S * V_LV ** 2 * gs * PW_ANCHOR
            row[band] = dict(bounded=e_b + e_periph, anchor=e_a + e_periph)
            log(f"{M:>3} books [{band}]: bounded {row[band]['bounded']:.3e} J"
                f"  anchor {row[band]['anchor']:.3e} J")
        row["lo"] = min(row[b]["bounded"] for b in BANDS)
        row["hi"] = row["SnCr"]["anchor"]
        row["hi_w"] = row["W"]["anchor"]
        out[M] = row

    e12 = torch.load(artifacts.path("energy_model"))["totals"]
    out["deployed"] = dict(lo=e12["lo"], hi=e12["hi"], hi_w=e12["hi_w"])
    log(f"deployed two-level: bounded {e12['lo']:.3e} J  anchor "
        f"(tin/chromium) {e12['hi']:.3e} J  anchor (tungsten) "
        f"{e12['hi_w']:.3e} J")
    torch.save(out, artifacts.path("energy_frontier"))
    log(f"saved: {artifacts.path('energy_frontier').name}")
    fh.close()


if __name__ == "__main__":
    main()
