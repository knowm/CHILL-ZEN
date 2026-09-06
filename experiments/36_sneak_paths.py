"""Sneak paths in the unit crossbar, and what each way out costs.

Paper Limitation 2 and Appendix B. Two computations, no data and no
banks:

1. A nodal solve of an n x n crossbar with every device at a common
   conductance G, one row driven to the rail and one column to the node,
   the other lines floating. Prints the conductance the node sees, the
   selected device's share of it, the first-order weight each unselected
   device carries in the read (row/column neighbours against the rest),
   and the fraction of a pulse each unselected device sees. Then the
   two half-select variants: node-side unselected lines held at a fixed
   potential (exact differential, but a shunt on the node: the gain and
   the hot read voltage that restores 10 mV at the comparator) and
   node-side lines tracking V_y (exact, with the current the driver
   sinks).

2. The energy model (chill_zen/energy_geom.py) at each SNEAK_OPTIONS
   geometry on the deployed census: floating lines as priced, the two
   half-select schemes, and the 16 x 1 selector column under both mux
   layouts.

Expected: 4 x 4 floating, node sees 2.286 G, selected share 0.4375;
weights 1 / 0.184 (6 neighbours) / 0.020 (9 others), shares 44 / 48 / 8%;
pulse fractions 3/7 and 1/7. 16 x 16 share 0.121. Fixed ground: shunt
2.25 G per side, gain 0.31, hot read 32.5 mV. Energies: 2.00e-9 (7.9x)
as priced; 3.0e-9 (5.2x) fixed ground; 3.0e-9 (5.2x) tracking; 1.6e-9
(9.9x) and 9.1e-10 (17.2x) for the selector column.

Cost: seconds. Requires nothing.

    python experiments/36_sneak_paths.py
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts                                    # noqa: E402
from chill_zen.energy_geom import (SNEAK_OPTIONS, V_HOT, energy,   # noqa: E402
                                   fmt_row, two_level_census)


def solve(n, G, v_rail=1.0, v_node=0.0, rows=None, cols=None):
    """Nodal solve of one n x n unit crossbar. Row 0 is driven to v_rail
    and column 0 to v_node; `rows` / `cols` hold the unselected lines at
    a potential, or float them when None. Returns node potentials and
    the current each driven node supplies."""
    N = 2 * n
    A = np.zeros((N, N))
    for r in range(n):
        for c in range(n):
            g, i, j = G[r, c], r, n + c
            A[i, i] += g
            A[j, j] += g
            A[i, j] -= g
            A[j, i] -= g
    fixed = {0: v_rail, n: v_node}
    for r in range(1, n):
        if rows is not None:
            fixed[r] = rows
    for c in range(1, n):
        if cols is not None:
            fixed[n + c] = cols
    free = [k for k in range(N) if k not in fixed]
    V = np.zeros(N)
    for k, v in fixed.items():
        V[k] = v
    if free:
        fk = list(fixed)
        rhs = -A[np.ix_(free, fk)] @ np.array([fixed[k] for k in fk])
        V[free] = np.linalg.solve(A[np.ix_(free, free)], rhs)
    return V, A @ V


def node_conductance(n, G, **kw):
    """Conductance the node sees from the driven row, in units of G."""
    V, I = solve(n, G, **kw)
    return -I[n]


def main():
    log, fh = artifacts.make_log("36_sneak_paths")
    log("sneak paths in a floating-line unit crossbar; every device at a "
        "common conductance G, one row and one column driven")

    out = {}
    for n in (4, 16):
        G = np.ones((n, n))
        V, I = solve(n, G)
        g_total = -I[n]
        g_sel = G[0, 0] * (V[0] - V[n])
        stages = (n - 1, (n - 1) ** 2, n - 1)
        sneak = 1 / sum(1 / s for s in stages)
        log(f"\n{n} x {n}: stages {stages[0]}G / {stages[1]}G / {stages[2]}G "
            f"in series = {sneak:.3f}G in parallel with the selected G; "
            f"node sees {g_total:.3f}G; selected share {g_sel / g_total:.4f}"
            f" = {2 * n - 1}/{n * n}")
        log(f"   floating columns at {V[n + 1]:.4f} V, floating rows at "
            f"{V[1]:.4f} V of the 1 V read; the {2 * (n - 1)} row/column "
            f"neighbours see {1 - V[n + 1]:.4f} of a pulse, the "
            f"{(n - 1) ** 2} others {V[n + 1] - V[1]:.4f}")
        # first-order weight of each device in the read
        eps = 1e-6
        S = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                Gp = G.copy()
                Gp[r, c] += eps
                S[r, c] = (node_conductance(n, Gp) - g_total) / eps
        nbr = S[0, 1:].sum() + S[1:, 0].sum()
        rest = S[1:, 1:].sum()
        log(f"   first-order weights (selected = 1): each row/column "
            f"neighbour {S[0, 1]:.3f}, each of the others {S[1, 1]:.3f}; "
            f"shares of the read: selected {S[0, 0] / S.sum():.3f}, "
            f"neighbours {nbr / S.sum():.3f}, others {rest / S.sum():.3f}")
        out[n] = dict(node_g=g_total, share=g_sel / g_total, sneak=sneak,
                      w_nbr=S[0, 1], w_rest=S[1, 1],
                      shares=(S[0, 0] / S.sum(), nbr / S.sum(),
                              rest / S.sum()),
                      pulse_nbr=1 - V[n + 1], pulse_rest=V[n + 1] - V[1])

    # half-select on the 4 x 4: only the node-side (column) lines are held
    n, G = 4, np.ones((4, 4))
    log("\nhalf-select, 4 x 4, node-side unselected lines held, rail-side "
        "lines floating:")
    # (a) fixed ground. No neighbour current reaches the node from the
    # rail, so the differential read is exact, but the three column
    # neighbours shunt the node toward ground through the floating rows.
    # Shunt per side: drive the node to 1 V with the rail row and the
    # unselected columns at ground; the current the node supplies, less
    # the selected device's own 1 G, is the shunt.
    Vs, Is = solve(n, G, v_rail=0.0, v_node=1.0, cols=0.0)
    shunt = Is[n] - G[0, 0]
    # waste: rail row at 1 V into the three grounded columns
    Vg, Ig = solve(n, G, cols=0.0)
    waste = sum(G[0, c] * (Vg[0] - Vg[n + c]) ** 2 for c in range(1, n))
    # the pool: K pairs, each side G, so the node's own conductance is 2KG
    # against a shunt of 2 * shunt * K G
    gain = 2.0 / (2.0 + 2 * shunt)
    v_hot_restored = V_HOT / gain
    log(f"  fixed ground: shunt on the node {shunt:.3f}G per side (the "
        f"three column neighbours through the floating rows); the pool's "
        f"own 2G per pair against {2 * shunt:.2f}G of shunt gives a read "
        f"gain of {gain:.3f}; restoring 10 mV at the comparator needs a "
        f"hot read of {v_hot_restored * 1e3:.1f} mV. Driver sinks "
        f"{waste:.1f} G V^2 per side, three times the selected device.")
    # (b) node-side lines track V_y
    Vt, It = solve(n, G, cols=0.0)   # node at 0, columns at 0: exact
    g_into_node = -It[n]
    waste_t = sum(G[0, c] * (Vt[0] - Vt[n + c]) ** 2 for c in range(1, n))
    log(f"  tracking V_y: node sees {g_into_node:.3f}G, the selected "
        f"device alone; floating rows sit at V_y and carry nothing; the "
        f"replica driver sinks {waste_t:.1f} G V^2 per side, "
        f"{waste_t:.0f}KGV per lane at K spaces.")
    log("  16 x 1 selector column: no second line exists for current to "
        "sneak onto; the read is the selected device, with no driver.")

    log("\nthe energy model at each option (deployed census; Limitation 2):")
    table = {}
    for name, c in SNEAK_OPTIONS.items():
        v_hot = v_hot_restored if name.startswith("node-side lines at fixed") \
            else V_HOT
        e = energy(two_level_census(v_hot=v_hot), c)
        table[name] = dict(total=e["total"], sel=e["sel"], read=e["read"],
                           readout=e["readout"], ratio=e["ratio"],
                           v_hot=v_hot, lane_pitch_um=c.lane_pitch_um,
                           c_seg=c.c_seg, segments=c.segments,
                           c_node_134=c.c_node(134), c_node_232=c.c_node(232))
        log("  " + fmt_row(name, e))
        log(f"      hot read {v_hot * 1e3:.1f} mV; lines per assertion "
            f"{c.segments}; lane pitch {c.lane_pitch_um:.2f} um; C_seg "
            f"{c.c_seg * 1e15:.2f} fF; C_node {c.c_node(134) * 1e15:.0f} / "
            f"{c.c_node(232) * 1e15:.0f} fF (K = 134 / 232); leakage paths "
            f"per space {2 * (16 if c.selector_column else 3)}")

    torch.save(dict(crossbar=out, fixed_ground=dict(shunt=shunt, gain=gain,
                                                    v_hot=v_hot_restored,
                                                    waste=waste),
                    tracking=dict(waste=waste_t), options=table),
               artifacts.path("sneak_paths"))
    log(f"\nsaved: {artifacts.path('sneak_paths').name}")
    fh.close()


if __name__ == "__main__":
    main()
