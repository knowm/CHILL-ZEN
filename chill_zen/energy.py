"""The energy model (paper Sec. V B).

A physical model, never a measurement. It maps the emulator's read
settings onto (V, t, G) through the substrate's read-noise law and the
measured conductance bands of fabricated memristors, then multiplies by
the exact per-image instruction census.

The load-bearing identity: for a read whose noise is thermally limited,

    sigma^2 = 2 kB T / (t V^2 G_sum)   and   E = V^2 G_sum t,

so E * sigma^2 = 2 kB T, independent of the devices. A stochastic
sample at noise sigma costs 2 kB T / sigma^2, for a pooled lane read
exactly as for a single device.

Stated assumptions, all parameters here rather than buried in a script:

  A1  device conductance band, per chemistry, from the datasheet's
      LRS/HRS against write compliance current: tungsten spans
      50 kOhm-1 MOhm; tin/chromium at write compliance at or below
      1 uA reaches 1-30 MOhm.
  A2  sub-threshold read voltage 50 mV (the emulator's own), with a
      10 mV sense floor below which a comparator cannot resolve.
  A3  read pulse width: 2 ns electronics floor or 5RC settling,
      whichever is longer, at 100 fF of lane capacitance. The
      emulator's literal 1 us reference read is the conservative
      anchor.
  A4  room temperature, 298 K.

  P1  input-line charging, 50 fF per selected line per read event.
  P2  one comparator per lane per read event, 1 fJ.
  P3  address select and clock share, 0.5 fJ per group read.

Three accountings are carried everywhere and all three are reported:
FLOOR (the thermodynamic 2kBT/sigma^2 at the delivered noise), BOUNDED
(the floors-bounded operating point), and ANCHOR (the emulator's
literal reference read), the last on both device bands.
"""
import math

import torch

KB = 1.380649e-23
T_ROOM = 298.0                      # A4
KBT = KB * T_ROOM

GMAX = 255                          # byte weight model
V_LV = 0.05                         # A2 sub-threshold read voltage
V_FLOOR = 0.010                     # A2 sense floor
V_COLD = 0.05                       # cold verdict read voltage
T_SETTLE_MIN = 2e-9                 # A3 electronics timing floor
C_LANE = 100e-15                    # A3 lane capacitance
PW_ANCHOR = 1e-6                    # emulator reference pulse width

# A1: per-device conductance bands (siemens), by chemistry.
BANDS = {"W": (1e-6, 20e-6), "SnCr": (3.33e-8, 1e-6)}

C_LINE = 50e-15                     # P1
E_CMP = 1e-15                       # P2
E_ADDR = 0.5e-15                    # P3

# Emulator noise-model constants (ktram_neural_core.torch._lane defaults).
A_THERMAL_UNIT = 0.005              # thermal weight at unit gain
A_FLICKER_UNIT = 1.0                # flicker weight at unit gain
SQRT_REF_M = math.sqrt(GMAX)
FLICKER_LN_REF = 6.0 * math.log(10.0)
T_DRAW = 0.1                        # the draw temperature
GAIN_PHYS = 0.02                    # the substrate's master noise gain

# Bank geometry of the deployed system.
LREP, NG, NPU, S = 6, 128, 98, 16
KSP_BACKBONE = LREP + NG            # 134
KSP_PATCH = LREP + NG + NPU         # 232


def pooled_read(ga, gb, aat):
    """The divider read, done directly on the byte weights.

    ga/gb: [L, K, S] int32. aat: [B, K] long, -1 = open. Returns
    (y, m) as float64 [B, L] -- the same arithmetic the emulator's
    own read performs, recomputed here so the pooled magnitude m is
    available for the noise and conductance model.
    """
    L, K, Ssz = ga.shape
    diff = (ga - gb).to(torch.float64).reshape(L, K * Ssz)
    mag = (ga + gb).to(torch.float64).reshape(L, K * Ssz)
    B = aat.shape[0]
    act = torch.zeros(B, K * Ssz, dtype=torch.float64)
    cols = torch.arange(K) * Ssz + aat.clamp(min=0)
    act.scatter_(1, cols, (aat >= 0).to(torch.float64))
    d = act @ diff.T
    m = act @ mag.T
    y = torch.where(m > 0, d / m.clamp(min=1e-30), torch.zeros_like(d))
    return y, m


def sigma_parts(y, m, v_read=V_LV, pw=PW_ANCHOR):
    """Unit-gain read noise, split into thermal and flicker terms."""
    bw_th = math.sqrt(PW_ANCHOR / pw)
    ln_band = FLICKER_LN_REF + math.log(PW_ANCHOR / pw)
    bw_fl = math.sqrt(ln_band / FLICKER_LN_REF) if ln_band > 0 else 0.0
    f_m = SQRT_REF_M / torch.sqrt(m.clamp(min=1e-30))
    s_th = A_THERMAL_UNIT * f_m / v_read * bw_th
    s_fl = A_FLICKER_UNIT * (1.0 - y * y) * f_m * bw_fl
    return s_th, s_fl


def sigma_target(y, m):
    """The noise the emulator actually delivers at the draw temperature."""
    s_th, s_fl = sigma_parts(y, m)
    return T_DRAW * torch.sqrt(s_th ** 2 + s_fl ** 2)


def g_sum_from_bytes(n_sel, m_bytes, band="W"):
    """Selected-pool conductance from the byte magnitudes under A1."""
    g_lo, g_hi = BANDS[band]
    return n_sel * 2 * g_lo + (m_bytes / GMAX) * (g_hi - g_lo)


def t_read(gs):
    """Read pulse width: the timing floor or 5RC settling, whichever wins."""
    return max(T_SETTLE_MIN, 5 * C_LANE / gs)


def hot_read_energies(y, m, n_sel, band):
    """(floor, bounded, anchor, sigma_target, sigma_delivered) for a draw read."""
    gs = g_sum_from_bytes(n_sel, m.median().item(), band)
    t_b = t_read(gs)
    st = sigma_target(y, m).median().item()
    e_floor = 2 * KBT / st ** 2
    v_want = math.sqrt(2 * KBT / (t_b * gs * st ** 2))
    v_op = max(V_FLOOR, v_want)
    e_b = v_op ** 2 * gs * t_b
    return e_floor, e_b, V_LV ** 2 * gs * PW_ANCHOR, st, \
        math.sqrt(2 * KBT / e_b)


def cold_read_energies(m, n_sel, band):
    """(bounded, anchor, sigma_delivered) for a verdict read."""
    gs = g_sum_from_bytes(n_sel, m.median().item(), band)
    t_b = t_read(gs)
    e_b = V_COLD ** 2 * gs * t_b
    return e_b, V_LV ** 2 * gs * PW_ANCHOR, math.sqrt(2 * KBT / e_b)
