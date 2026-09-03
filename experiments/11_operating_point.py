"""Map the emulator's read settings onto physical (V, t, R).

Paper Sec. V B, first half. Before any energy can be quoted, the
emulator's dimensionless noise settings have to be given a physical
reading, and the actual read statistics of the deployed system have to
be measured rather than assumed. This script does both, from the frozen
banks and the cached n = 1000 generation batch.

It reports, per instruction class (the autoregressive draw at several
chain positions, the two cold sweeps, the parallel patch draw): how
many input addresses are selected, the pooled magnitude the read sees,
the divider output, and the thermal and flicker parts of the noise.

Expected readings:

* the emulator's default noise gain at its reference read implies a
  Johnson source of 1.2 MOhm -- the top of the working band, so the
  emulator is not modelling an implausibly quiet device;
* draw noise, measured over the contexts actually deployed, has median
  sigma about 0.029 for the backbone chain (0.15 at step 0 falling to
  0.016 at step 127) and 0.0165 for the patch read -- flicker-dominated
  in this model;
* the cold verdict reads have a winner margin of 0.097, about 46 times
  the physical flicker floor, so treating them as noiseless is a
  justified idealization rather than a convenient one;
* settling is 0.1 to 1.9 ns across the band at 100 fF, which is what
  makes the 2 ns timing floor the binding one.

Cost: about a minute, reads only. Requires 03-05 and 07.

    python experiments/11_operating_point.py
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.energy import (BANDS, C_LANE, GAIN_PHYS, GMAX, KBT,  # noqa: E402
                         KSP_BACKBONE, KSP_PATCH, LREP, NG, NPU,
                         PW_ANCHOR, S, T_DRAW, V_FLOOR, V_LV,
                         A_THERMAL_UNIT, g_sum_from_bytes, pooled_read,
                         sigma_parts)

PW_SWEEP = [1e-9, 1e-8, 1e-7, 1e-6]


def stats(name, y, m, n_sel, hot, out, log):
    s_th, s_fl = sigma_parts(y, m)
    s_unit = torch.sqrt(s_th ** 2 + s_fl ** 2)
    gain = T_DRAW if hot else 0.0
    row = dict(n_sel=n_sel, m_med=m.median().item(),
               m_q10=m.quantile(0.10).item(), m_q90=m.quantile(0.90).item(),
               y_absmed=y.abs().median().item(),
               s_th_med=s_th.median().item(), s_fl_med=s_fl.median().item(),
               s_unit_med=s_unit.median().item(), gain=gain,
               s_eff_med=(gain * s_unit.median()).item() if hot else 0.0)
    if hot:
        s_eff = gain * s_unit.median().item()
        row["E_floor_J"] = 2 * KBT / (s_eff ** 2)
        row["E_floor_kBT"] = 2 / (s_eff ** 2)
    out[name] = row
    log(f"{name:<20} n_sel={n_sel:>3}  pooled magnitude median "
        f"{row['m_med']:.0f} [{row['m_q10']:.0f}..{row['m_q90']:.0f}]  "
        f"|y| median {row['y_absmed']:.4f}")
    log(f"{'':20} sigma(thermal/flicker/total) = {row['s_th_med']:.2e} / "
        f"{row['s_fl_med']:.2e} / {row['s_unit_med']:.2e}"
        + (f"   delivered = {row['s_eff_med']:.2e}   thermodynamic floor "
           f"2kBT/sigma^2 = {row['E_floor_J']:.2e} J = "
           f"{row['E_floor_kBT']:.1e} kBT" if hot else
           "   (cold: the verdict read sits at the flicker floor)"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    args = ap.parse_args()
    log, fh = artifacts.make_log("11_operating_point")
    log("operating point — assumptions A1-A4 are in chill_zen/energy.py")

    # the emulator's thermal gain, read as an equivalent Johnson source
    for gain, tag in ((GAIN_PHYS, "substrate default"), (T_DRAW, "draw")):
        s_ref = gain * A_THERMAL_UNIT / V_LV
        r_impl = (s_ref ** 2) * PW_ANCHOR * V_LV ** 2 / (2 * KBT)
        log(f"anchor: at gain {gain} ({tag}), sigma_thermal at the reference "
            f"read = {s_ref:.4f} -> implied Johnson source R = "
            f"{r_impl:.3e} Ohm at V = {V_LV} V, t = {PW_ANCHOR:.0e} s, "
            f"m = {GMAX}")
    log("")

    bb = torch.load(artifacts.path("backbone_banks"))[args.backbone]
    g1a, g1b = bb["g"]["ga"], bb["g"]["gb"]
    r1a, r1b = bb["r"]["ga"], bb["r"]["gb"]
    pb = torch.load(artifacts.path("patch_banks"))
    g2a, g2b = pb["g"]["ga"], pb["g"]["gb"]
    jb = torch.load(artifacts.path("joint_bank"))
    ja, jbb = jb["bank"]["ga"], jb["bank"]["gb"]
    st = torch.load(artifacts.path("verdict"))["st"]
    g_codes, p_codes = st["g_e2e"].long(), st["p_e2e"].long()
    lab = torch.arange(10).repeat_interleave(len(g_codes) // 10)

    B = 200
    g_c, p_c, y_c = g_codes[:B], p_codes[:B], lab[:B]
    out = {}
    log("read classes, measured on the contexts the deployed system runs:")

    y_all, m_all = [], []
    for k in (0, 16, 48, 96, NG - 1):
        aat = torch.full((B, KSP_BACKBONE), -1, dtype=torch.long)
        aat[:, :LREP] = y_c[:, None]
        if k > 0:
            aat[:, LREP:LREP + k] = g_c[:, :k]
        lanes = slice((1 + k) * S, (2 + k) * S)
        yv, mv = pooled_read(g1a[lanes], g1b[lanes], aat)
        y_all.append(yv.flatten())
        m_all.append(mv.flatten())
        s_th, s_fl = sigma_parts(yv, mv)
        su = torch.sqrt(s_th ** 2 + s_fl ** 2)
        log(f"  draw step {k:>3}: n_sel={LREP + k:>3}  magnitude median "
            f"{mv.median():.0f}  delivered sigma median "
            f"{T_DRAW * su.median():.2e}")
    stats("backbone draw (hot)", torch.cat(y_all), torch.cat(m_all),
          LREP + NG // 2, True, out, log)

    aat = torch.cat([y_c[:, None].expand(-1, LREP), g_c], 1)
    yv, mv = pooled_read(r1a[S:], r1b[S:], aat)
    stats("backbone sweep (cold)", yv.flatten(), mv.flatten(),
          KSP_BACKBONE, False, out, log)

    aat = torch.cat([y_c[:, None].expand(-1, LREP), g_c,
                     torch.full((B, NPU), -1, dtype=torch.long)], 1)
    yv, mv = pooled_read(g2a, g2b, aat)
    stats("patch draw (hot)", yv.flatten(), mv.flatten(), LREP + NG, True,
          out, log)

    aat = torch.cat([y_c[:, None].expand(-1, LREP), g_c, p_c], 1)
    yv, mv = pooled_read(ja, jbb, aat)
    stats("joint sweep (cold)", yv.flatten(), mv.flatten(), KSP_PATCH,
          False, out, log)

    # how much margin the cold reads have over the physical noise floor
    yg = yv.reshape(B, -1, S)
    top2 = yg.topk(2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).flatten()
    _, s_fl = sigma_parts(yv, mv)
    floor = GAIN_PHYS * s_fl.reshape(B, -1, S).median().item()
    log(f"\ncold verdict margins: winner margin median {margin.median():.4f} "
        f"against a physical flicker floor of {floor:.2e} (gain "
        f"{GAIN_PHYS}) -> {margin.median() / floor:.1f}x")
    out["margins"] = dict(margin_med=margin.median().item(), floor=floor,
                          gain_phys=GAIN_PHYS)

    log(f"\nper-lane-read energy E = V^2 G_sum t at V = {V_LV} V, "
        "conductance from A1 (tungsten band):")
    log(f"{'class':<22}{'t':>8}{'E(G_lo) J':>12}{'E(G_hi) J':>12}"
        f"{'E(measured) J':>15}")
    table = {}
    g_lo, g_hi = BANDS["W"]
    for name in ("backbone draw (hot)", "backbone sweep (cold)",
                 "patch draw (hot)", "joint sweep (cold)"):
        row = out[name]
        n_sel = row["n_sel"]
        gp_meas = g_sum_from_bytes(n_sel, row["m_med"], "W") / n_sel
        rows = []
        for pw in PW_SWEEP:
            e_lo = V_LV ** 2 * n_sel * (2 * g_lo) * pw
            e_hi = V_LV ** 2 * n_sel * (2 * g_hi) * pw
            e_ms = V_LV ** 2 * n_sel * gp_meas * pw
            rows.append(dict(pw=pw, e_lo=e_lo, e_hi=e_hi, e_meas=e_ms))
            log(f"{name:<22}{pw:>8.0e}{e_lo:>12.2e}{e_hi:>12.2e}"
                f"{e_ms:>15.2e}")
        table[name] = dict(g_pair_meas=gp_meas, rows=rows)
    for tag, gsum in (("G_lo", KSP_BACKBONE * 2 * g_lo),
                      ("G_hi", KSP_BACKBONE * 2 * g_hi)):
        rc = C_LANE / gsum
        log(f"settling floor (C = {C_LANE:.0e} F, {KSP_BACKBONE}-pair lane, "
            f"{tag}): 5RC = {5 * rc:.2e} s")

    se = out["backbone draw (hot)"]["s_eff_med"]
    log(f"\nreading for the paper: the draw delivers sigma of order "
        f"{se:.1e}, so the thermodynamically limited energy at that noise "
        f"is {out['backbone draw (hot)']['E_floor_kBT']:.1e} kBT per read.")
    log(f"census: the joint bank is {NG} + {NPU} = {NG + NPU} target "
        f"addresses = {(NG + NPU) * S} lanes.")

    torch.save(dict(assumptions=dict(bands=BANDS, v_lv=V_LV,
                                     v_floor=V_FLOOR, pw_sweep=PW_SWEEP,
                                     c_lane=C_LANE),
                    classes=out, energy=table),
               artifacts.path("operating_point"))
    log(f"\nsaved: {artifacts.path('operating_point').name}")
    fh.close()


if __name__ == "__main__":
    main()
