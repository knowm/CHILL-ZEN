"""The energy model (paper Sec. V B and Appendix B).

A physical estimate from a stated geometry, never a measurement. The
model has four terms and no fitted parameter:

    E = E_sel + E_read + E_readout + E_rail

  E_sel      address delivery: for each bank, each space asserted per
             image charges one row-select and one column-select line
             across every lane of the bank, at the select supply.
  E_read     the divider read: each lane read settles its V_y node
             through the selected devices for t = 5RC, so
             E = V^2 G t = 5 C_node V^2. Device conductance cancels.
  E_readout  one clocked comparator plus the lane buffer, per lane read.
  E_rail     bounded below 1e-11 J per image and carried as zero.

Geometry (Appendix B, from Chapter 4 of the neural-lane series): a unit
crossbar is a small memristor crossbar with a row mux and a column mux
and no selector at the memristor; a synapse is an a-side and a b-side
unit crossbar in series between the rails; a lane is K such pairs, one
per address space, sharing one output line. The array is an integrated
crossbar with a 20 nm via on a 50 nm pitch. Each space has 16 symbols,
so the nominal unit crossbar is 4 x 4 with two 4:1 pass-transistor
muxes. The a- and b-side muxes share select lines.

Along a select line one lane is one unit crossbar wide: the array plus
one stack of four pass transistors, p = 4 d + 4 t. The V_y node is 2K
lane pitches long and carries 8K mux drains.

Sneak paths (paper Limitation 2). With the unselected lines of a 4 x 4
floating, the selected device is 7/16 of the conductance the node sees;
the rest is a series-parallel network of the other fifteen devices. The
model prices that geometry as is (`NOMINAL`), and prices three ways out
(`SNEAK_OPTIONS`): the node-side unselected lines pulled to a fixed
potential or to a replica of V_y by a pull-down gate on a bank-wide
complement line (`node_pulldown`; the drivers sink three times the
selected pool's current, `read_waste`), and a 16 x 1 selector column,
one 1-of-16 multiplexer on the rail side and no multiplexer on the node
side, which has no second line for current to sneak onto
(`selector_column`, with the 16 pass gates either beside the array along
the select line, the convention of the 4 x 4, or stacked along V_y).
Experiment 36 prints the table.

Only counts enter: lanes and spaces per bank, lane reads and their
voltage. No bank magnitude, no device band. The noise/operating-point
model that maps the emulator's settings onto (V, t, G) remains in
`energy.py` and feeds experiments 11 and 28; it does not enter the
joule accounting.
"""
from dataclasses import dataclass, replace

DTM_BEST_J = 1.568e-8            # their deepest chain, from their Fig. 1

S, LREP, NG, NPU, N_DECODE = 16, 6, 128, 98, 784
V_HOT, V_COLD = 0.010, 0.050     # the paper's read voltages


@dataclass(frozen=True)
class Constants:
    """Every constant of the model, with its source in Appendix B."""
    d_um: float = 0.050          # crossbar pitch (20 nm via); insensitive
    ucx: int = 4                 # unit crossbar side (4 x 4 holds 16 symbols)
    t_um: float = 0.20           # contacted transistor pitch, periphery
    c_gate: float = 0.3e-15      # pass-gate capacitance, min-size device
    c_wire: float = 0.2e-15      # F per um, min-width metal with neighbors
    c_drain: float = 0.1e-15     # off pass-transistor drain on a node
    v_sel: float = 0.8           # select supply
    e_readout: float = 15e-15    # comparator (10 fJ) + lane buffer (5 fJ)
    shared_ab: bool = True       # a- and b-side muxes share select lines
    node_pulldown: bool = False  # node-side unselected lines held by a
                                 # pull-down gate on a complement line
    selector_column: bool = False  # 16 x 1: one 1-of-16 mux, rail side
    stacked_mux: bool = False    # selector column: pass gates along V_y
    read_waste: float = 0.0      # half-select sink, relative to the pool

    @property
    def symbols(self):
        return self.ucx * self.ucx

    @property
    def lane_pitch_um(self):
        """Lane pitch along a select line: the array plus one mux stack."""
        if self.selector_column:
            stack = 1 if self.stacked_mux else self.symbols
            return self.d_um + stack * self.t_um
        return self.ucx * self.d_um + self.ucx * self.t_um

    @property
    def node_len_um(self):
        """Length of V_y per side of a pair: the array plus the node-side
        mux stack (the pull-down doubles that stack)."""
        if self.selector_column:
            return (self.symbols * self.t_um if self.stacked_mux
                    else self.symbols * self.d_um)
        stack = 2 * self.ucx if self.node_pulldown else self.ucx
        return self.ucx * self.d_um + stack * self.t_um

    @property
    def node_drains(self):
        """Off pass-transistor drains on V_y per side of a pair."""
        if self.selector_column:
            return 0
        return self.ucx + (1 if self.node_pulldown else 0)

    @property
    def c_seg(self):
        """One select-line segment per lane: the gates it drives + wire."""
        gates = 2 if self.shared_ab else 1
        return gates * self.c_gate + self.c_wire * self.lane_pitch_um

    @property
    def segments(self):
        """Select lines charged per (space, lane) assertion: one per mux
        (row and column; a and b if not shared), plus the complement line
        of the node-side mux when its unselected lines are pulled down."""
        per_mux = 1 if self.shared_ab else 2
        muxes = 1 if self.selector_column else 2
        pulldown = per_mux if self.node_pulldown else 0
        return muxes * per_mux + pulldown

    def c_node(self, K):
        return K * (2 * self.node_len_um * self.c_wire
                    + 2 * self.node_drains * self.c_drain)


NOMINAL = Constants()
POINTS = {
    "nominal": NOMINAL,
    "low-swing 0.5 V": replace(NOMINAL, v_sel=0.5),
    "7 nm-class periphery": replace(NOMINAL, t_um=0.06, c_gate=0.1e-15),
    "7 nm-class, 0.5 V": replace(NOMINAL, t_um=0.06, c_gate=0.1e-15,
                                 v_sel=0.5),
    "pessimistic corner": replace(NOMINAL, ucx=16, shared_ab=False,
                                  v_sel=1.0),
}
# The readout (Appendix B, Table on readout options). The model prices a
# continuous-time comparator: preamplifier and latch biased from the core
# supply for the read window, tripping once when the group's shared
# falling reference crosses the lane. E = I_bias V_dd T_window, plus the
# lane buffer. The alternatives are priced from the same constants.
K_B, T_K = 1.380649e-23, 300.0
V_DD = 0.8
E_BUFFER = 5e-15                 # lane buffer: 100 nA for the window
I_BIAS, T_WINDOW = 200e-9, 62.5e-9  # the priced continuous-time comparator
E_STROBE = 10e-15                # one clocked-comparator decision


def ct_energy(i_bias=I_BIAS, t_window=T_WINDOW, v_dd=V_DD):
    """Continuous-time comparator: standing bias over the read window."""
    return i_bias * v_dd * t_window


def ct_noise_bound(v_n, v_dd=V_DD, gamma=1.0, n_vt=0.035, bt=3.0):
    """Energy a continuous-time comparator needs to reach input noise v_n
    at a fixed bandwidth-time product: v_n^2 = 4kT gamma B / g_m with
    g_m = I / (n V_T), so E = I V_dd T = 4kT gamma (n V_T) V_dd (BT) / v_n^2.
    gamma 1 (conservative), n V_T 35 mV, BT 3."""
    return 4 * K_B * T_K * gamma * n_vt * v_dd * bt / v_n ** 2


def ramp_slope_min(v_n, bandwidth):
    """Slope at which each lane trips once: the reference crosses the
    comparator's +-3 sigma band within one noise correlation time."""
    import math
    return 6 * v_n * 2 * math.pi * bandwidth


# name -> (comparator energy per lane read, what the read returns,
#          noise samples per lane per read)
READOUT_OPTIONS = {
    "continuous-time, 200 nA for 62.5 ns (as priced)":
        (ct_energy(), "winner and rank", "one, if the ramp is fast"),
    "continuous-time, 1 uA for 62.5 ns":
        (ct_energy(1e-6), "winner and rank", "one; noise / sqrt(5)"),
    "clocked, binary search on the winner, 5 strobes":
        (5 * E_STROBE, "winner only", "five, first crossing"),
    "clocked, full ramp, 16 strobes":
        (16 * E_STROBE, "winner and 4-bit rank", "sixteen, first crossing"),
    "clocked tournament, 15 decisions per group":
        (15 / 16 * E_STROBE, "winner only", "one per pairwise comparison"),
}

# Limitation 2: the sneak-path options, priced from the nominal point.
# The fixed-ground row raises the hot read to undo the node-side shunt
# (experiment 36 computes the gain and the voltage); the others keep it.
SNEAK_OPTIONS = {
    "floating lines, as priced": NOMINAL,
    "node-side lines at fixed ground": replace(NOMINAL, node_pulldown=True,
                                               read_waste=3.0),
    "node-side lines track V_y": replace(NOMINAL, node_pulldown=True,
                                         read_waste=3.0),
    "16 x 1 selector column, mux beside the array":
        replace(NOMINAL, selector_column=True),
    "16 x 1 selector column, mux stacked along V_y":
        replace(NOMINAL, selector_column=True, stacked_mux=True),
}
SENSITIVITIES = {
    "separate a/b select lines": replace(NOMINAL, shared_ab=False),
    "16 x 16 unit crossbar": replace(NOMINAL, ucx=16),
    "readout 45 fJ": replace(NOMINAL, e_readout=45e-15),
    "wire 0.3 fF/um, gate 0.5 fF": replace(NOMINAL, c_wire=0.3e-15,
                                           c_gate=0.5e-15),
    "crossbar pitch 100 nm": replace(NOMINAL, d_um=0.100),
    "transistor pitch 0.4 um": replace(NOMINAL, t_um=0.40),
}


@dataclass(frozen=True)
class Bank:
    name: str
    lanes: int        # L: lanes that receive the assertion
    spaces: int       # K: spaces asserted per image
    reads: int        # lane reads per image
    v_read: float


def two_level_census(v_hot=V_HOT):
    """The deployed two-level system: five banks, 10,064 lane reads."""
    return [
        Bank("backbone draw (g)", (1 + NG) * S, LREP + NG, NG * S, v_hot),
        Bank("backbone sweep (r)", NG * S, LREP + NG, NG * S, V_COLD),
        Bank("patch draw", NPU * S, LREP + NG, NPU * S, v_hot),
        Bank("joint sweep", (NG + NPU) * S, LREP + NG + NPU,
             (NG + NPU) * S, V_COLD),
        Bank("decode", N_DECODE, LREP + NG + NPU, N_DECODE, V_COLD),
    ]


def one_level_census(M):
    """A backbone-only system of M books: draw, sweep, decode."""
    return [
        Bank("backbone draw (g)", (1 + M) * S, LREP + M, M * S, V_HOT),
        Bank("backbone sweep (r)", M * S, LREP + M, M * S, V_COLD),
        Bank("decode", N_DECODE, LREP + M, N_DECODE, V_COLD),
    ]


def energy(banks, c=NOMINAL):
    """Per-image energy of a census at constants c. Joules."""
    per_bank = []
    e_sel = e_read = e_rdout = 0.0
    for b in banks:
        sel = b.lanes * b.spaces * c.segments * c.c_seg * c.v_sel ** 2
        read = (b.reads * 5 * c.c_node(b.spaces) * b.v_read ** 2
                * (1 + c.read_waste))
        rdout = b.reads * c.e_readout
        per_bank.append(dict(name=b.name, lanes=b.lanes, spaces=b.spaces,
                             assertions=b.lanes * b.spaces, reads=b.reads,
                             v_read=b.v_read, sel=sel, read=read,
                             readout=rdout))
        e_sel += sel
        e_read += read
        e_rdout += rdout
    total = e_sel + e_read + e_rdout
    return dict(sel=e_sel, read=e_read, readout=e_rdout, total=total,
                ratio=DTM_BEST_J / total, per_bank=per_bank,
                assertions=sum(p["assertions"] for p in per_bank),
                reads=sum(p["reads"] for p in per_bank))


def excluded_bounds(banks, c=NOMINAL, i_gate=5e-12, t_pulse=10e-9,
                    t_image=30e-6):
    """Bounds on the terms the model excludes (Appendix B 7). Rails are
    asserted per read at the read voltage and sit at the node potential
    between reads, so a lane conducts only during its own pulse.

      rail_c        wire capacitance of both rails over every lane
      rail_charge   charging a bank's rail segments once per read step
                    at that read's voltage (the pulse is the swing)
      gates         off pass gates on a DC path from an asserted rail:
                    both muxes' unselected gates, both sides (4 x 4:
                    2 x 2 x 3 per space; selector column: 2 x 15)
      leak_pulsed   those gates leaking at i_gate for t_pulse per read
                    step, at the bank's read voltage (the model's case)
      leak_held     the same gates leaking for the whole image at
                    50 mV, the case the model does not assume
    """
    per_space_gates = (2 * (c.symbols - 1) if c.selector_column
                       else 2 * 2 * (c.ucx - 1))
    rail_c = rail_charge = leak_pulsed = leak_held = gates = 0.0
    for b in banks:
        seg = 2 * 2 * b.spaces * c.lane_pitch_um * c.c_wire  # both rails
        rail_c += b.lanes * seg
        # the g bank reads 16 lanes per step over 128 steps and its
        # rails are bank-wide, so each step charges them; the others read
        # every lane at once
        n_steps = 128 if b.name.startswith("backbone draw") else 1
        rail_charge += n_steps * b.lanes * seg * b.v_read ** 2
        g = b.lanes * b.spaces * per_space_gates
        gates += g
        leak_pulsed += g * i_gate * b.v_read * n_steps * t_pulse
        leak_held += g * i_gate * 0.050 * t_image
    return dict(rail_c=rail_c, rail_charge=rail_charge, gates=gates,
                leak_pulsed=leak_pulsed, leak_held=leak_held,
                per_space_gates=per_space_gates)


def all_points(banks):
    return {name: energy(banks, c) for name, c in POINTS.items()}


def fmt_row(name, e):
    return (f"{name:<28} sel {e['sel']:.2e}  read {e['read']:.2e}  "
            f"readout {e['readout']:.2e}  total {e['total']:.2e}  "
            f"DTM/ours {e['ratio']:5.1f}x")
