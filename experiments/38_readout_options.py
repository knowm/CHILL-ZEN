"""The readout priced as a circuit, and what each alternative costs.

Appendix B. The model prices one comparator per lane as a
continuous-time comparator, a preamplifier and latch biased at 200 nA
from the 0.8 V supply for a 62.5 ns read window (10 fJ), plus the lane
buffer (5 fJ). It trips once, when the group's shared falling reference
crosses the lane, so the read is one noise sample per lane, which is
what the emulator computes, provided the reference falls faster than
the comparator's noise correlation time. This script prints:

  * the readout options: continuous-time at two bias levels, a clocked
    comparator strobed down a ramp (5 strobes for the winner by binary
    search, 16 for a 4-bit rank), and a clocked pairwise tournament;
    per lane read, readout share, total, ratio to the DTM chain;
  * the continuous-time noise-energy bound: the energy a comparator
    needs to reach a given input-referred noise at fixed bandwidth-time
    product, at the register levels the paper uses;
  * the ramp condition at the hot operating point.

Cost: seconds. Requires nothing.

    python experiments/38_readout_options.py
"""
import pathlib
import sys

import torch
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts                                    # noqa: E402
from chill_zen.energy_geom import (E_BUFFER, NOMINAL, READOUT_OPTIONS,  # noqa: E402
                                   ct_noise_bound, energy, fmt_row,
                                   ramp_slope_min, two_level_census)


def main():
    log, fh = artifacts.make_log("38_readout_options")
    banks = two_level_census()
    log("readout options on the deployed census (Appendix B):")
    table = {}
    for name, (e_cmp, returns, samples) in READOUT_OPTIONS.items():
        e = energy(banks, replace(NOMINAL, e_readout=e_cmp + E_BUFFER))
        table[name] = dict(per_read=e_cmp + E_BUFFER, total=e["total"],
                           ratio=e["ratio"], share=e["readout"] / e["total"],
                           returns=returns, samples=samples)
        log("  " + fmt_row(name, e))
        log(f"      {(e_cmp + E_BUFFER) * 1e15:5.1f} fJ per lane read; "
            f"readout {e['readout'] / e['total']:.0%} of the total; returns "
            f"{returns}; noise samples per lane: {samples}")

    log("\ncontinuous-time noise-energy bound, E = 4kT gamma (n V_T) V_dd (BT)"
        " / v_n^2 (gamma 1, n V_T 35 mV, V_dd 0.8 V, BT 3):")
    levels = {"2 mV (code 95, the hot read)": 2e-3, "1 mV (code 45)": 1e-3,
              "300 uV (code 10)": 300e-6, "100 uV (code 0)": 100e-6}
    bound = {}
    for name, v in levels.items():
        b = ct_noise_bound(v)
        bound[name] = b
        log(f"  {name:<30} {b * 1e15:8.2f} fJ per lane read")

    slope = ramp_slope_min(2e-3, 500e6)
    log(f"\nramp condition at v_n = 2 mV, B = 500 MHz: slope >= "
        f"{slope * 1e-6:.0f} mV/ns; the 20 mV range in "
        f"{0.020 / slope * 1e9:.2f} ns")

    torch.save(dict(options=table, bound=bound, slope=slope),
               artifacts.path("verdict").parent / "readout-options.pt")
    log("\nsaved: readout-options.pt")
    fh.close()


if __name__ == "__main__":
    main()
