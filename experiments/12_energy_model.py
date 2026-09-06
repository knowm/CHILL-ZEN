"""Joules per image for the deployed two-level system.

Paper Sec. V B, Table V and Appendix B. A physical estimate from a
stated geometry (chill_zen/energy_geom.py), not a measurement. Only the
census enters: lanes and spaces per bank, lane reads and their voltage.

The census is exact, taken from the banks' lane and space counts:
2048 hot autoregressive reads, 2048 cold backbone-sweep reads, 1568 hot
patch reads, 3616 cold joint-sweep reads, and 784 cold decode reads,
10,064 lane reads per image, and 1,781,920 (space, lane) assertions.

Expected, at the nominal point: E_sel 1.82e-9 J, E_read 1.97e-11 J,
E_readout 1.51e-10 J, total 2.00e-9 J, 7.9x below the DTM chain's
1.568e-8 J. Low-swing select 8.8e-10 J (17.7x); pessimistic corner
8.1e-9 J (1.9x).

Cost: seconds. Requires nothing.

    python experiments/12_energy_model.py
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts                                    # noqa: E402
from chill_zen.energy_geom import (DTM_BEST_J, NOMINAL, POINTS,     # noqa: E402
                                   SENSITIVITIES, all_points, energy,
                                   excluded_bounds, fmt_row,
                                   two_level_census)


def main():
    log, fh = artifacts.make_log("12_energy_model")
    log("energy model — geometry and constants in chill_zen/energy_geom.py")
    c = NOMINAL
    log(f"nominal geometry: {c.ucx}x{c.ucx} unit crossbar, crossbar pitch "
        f"{c.d_um * 1e3:.0f} nm, transistor pitch {c.t_um:.2f} um, lane "
        f"pitch {c.lane_pitch_um:.2f} um, C_seg {c.c_seg * 1e15:.2f} fF, "
        f"C_node {c.c_node(134) * 1e15:.0f} fF (K=134) / "
        f"{c.c_node(232) * 1e15:.0f} fF (K=232), V_sel {c.v_sel} V, "
        f"readout {c.e_readout * 1e15:.0f} fJ")

    banks = two_level_census()
    nom = energy(banks, c)
    log("\nper-bank census and nominal contributions:")
    log(f"{'bank':<20}{'lanes':>7}{'spaces':>8}{'LK':>10}{'reads':>7}"
        f"{'V':>7}{'E_sel':>10}{'E_read':>10}{'E_rdout':>10}")
    for p in nom["per_bank"]:
        log(f"{p['name']:<20}{p['lanes']:>7}{p['spaces']:>8}"
            f"{p['assertions']:>10}{p['reads']:>7}{p['v_read'] * 1e3:>5.0f}mV"
            f"{p['sel']:>10.2e}{p['read']:>10.2e}{p['readout']:>10.2e}")
    log(f"{'total':<20}{'':>7}{'':>8}{nom['assertions']:>10}"
        f"{nom['reads']:>7}{'':>7}{nom['sel']:>10.2e}{nom['read']:>10.2e}"
        f"{nom['readout']:>10.2e}")
    log(f"\nshares at the nominal point: sel {nom['sel'] / nom['total']:.1%}, "
        f"readout {nom['readout'] / nom['total']:.1%}, "
        f"read {nom['read'] / nom['total']:.1%}")

    log("\noperating points (Table V):")
    points = all_points(banks)
    for name, e in points.items():
        log("  " + fmt_row(name, e))
    log("\nsingle-knob sensitivities from the nominal row:")
    sens = {name: energy(banks, cc) for name, cc in SENSITIVITIES.items()}
    for name, e in sens.items():
        log("  " + fmt_row(name, e))
    log(f"\ntheir deepest DTM chain: {DTM_BEST_J:.3e} J/sample")

    bnd = excluded_bounds(banks, c)
    tot = nom["total"]
    log("\nbounds on the excluded terms (Appendix B 7), nominal:")
    log(f"  rails: {bnd['rail_c'] * 1e9:.2f} nF over {nom['reads'] and sum(b.lanes for b in banks)} lanes; "
        f"charged per read step at the read voltage: {bnd['rail_charge']:.1e} J "
        f"({bnd['rail_charge'] / tot:.2%} of the total)")
    log(f"  leakage: {bnd['per_space_gates']} off gates per space on a DC path, "
        f"{bnd['gates']:.2e} gates; at 5 pA, 10 ns pulses per read step: "
        f"{bnd['leak_pulsed']:.1e} J ({bnd['leak_pulsed'] / tot:.2%}); "
        f"if the rails were held for 30 us at 50 mV: {bnd['leak_held']:.1e} J "
        f"({bnd['leak_held'] / tot:.1%})")
    log(f"  synapse pairs {nom['assertions'] * 16:.3e}, memristors "
        f"{nom['assertions'] * 32:.3e}")
    g = nom["per_bank"][0]
    log(f"  g bank: {g['assertions']} assertions, 142336 do work; segmenting "
        f"per group saves {(1 - 142336 / g['assertions']):.1%} of the g bank's "
        f"E_sel, {(g['sel'] * (1 - 142336 / g['assertions'])) / nom['sel']:.1%} "
        f"of E_sel")
    sens_d = energy(banks, SENSITIVITIES["crossbar pitch 100 nm"])
    log(f"  crossbar pitch 100 nm moves the total by "
        f"{sens_d['total'] / tot - 1:.1%}")

    torch.save(dict(census=[p for p in nom["per_bank"]],
                    points={k: {kk: vv for kk, vv in v.items()
                                if kk != "per_bank"}
                            for k, v in points.items()},
                    sensitivities={k: v["total"] for k, v in sens.items()},
                    totals=dict(lo=points["nominal"]["total"],
                                hi=points["pessimistic corner"]["total"],
                                low_swing=points["low-swing 0.5 V"]["total"]),
                    constants={k: str(v) for k, v in POINTS.items()}),
               artifacts.path("energy_model"))
    log(f"\nsaved: {artifacts.path('energy_model').name}")
    fh.close()


if __name__ == "__main__":
    main()
