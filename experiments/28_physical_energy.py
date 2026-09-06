"""Energy at the physical operating point, not a T-derived target.

`12_energy_model.py` and `22_binary_energy.py` compute a hot read's
bounded energy by solving backward from `sigma_target` -- the
emulator's `T = 0.1`/`0.3` draw noise evaluated at the *reference*
read (50 mV, 1 us) -- for the voltage that would deliver it thermally,
then clamp at `V_FLOOR` (10 mV) whenever that voltage is sub-floor.
That voltage is sub-floor for essentially every hot read this system
reports (`V_need` 1.4-12.7 mV against a 10 mV floor), so
`v_op = V_FLOOR` already, in every one of those rows: the "target" was
never reachable and the clamp did the real work.
`e_bound = V_FLOOR^2 * G_sum * t_b` is therefore already the physical
operating point's energy; nothing about the number the paper states
changes. What changes is what we say produced the read's
noise: not a solved-for target, but the composite physical sigma at
`(V_FLOOR, t_b)` -- the lane term (`chill_zen.physical`'s
`GAIN_DEV * sigma_unit`, here `energy.sigma_parts` evaluated directly
at the operating point instead of at the reference point) plus the
comparator's flat `v_n / V_FLOOR`, in quadrature.

This script recomputes the census of `12_energy_model.py` (grayscale)
and `22_binary_energy.py` (binary) and reports, per hot-read class:
the old target/delivered pair (model B, the thermodynamic identity,
backward-solved), the new physical lane sigma (model E, the emulator's
own law, evaluated directly at the operating point -- these agree
within about 25%, a finding this script re-confirms rather than fits),
the comparator's sigma_cmp at the grayscale point (v_n = 2 mV, P4) and
the binary point (v_n = 3 mV, P6), and the composite. It confirms
`e_bound` is unchanged (to floating-point noise) and adds the
comparator's own energy line
(E_CMP, already charged in the periphery total) explicitly, plus the
reference-read row at v_n = 5 mV (P5, the 50 mV twin).

Cost: under a minute; analysis on existing artifacts. Requires 03-05,
07, 11, 17, 18 (backbone/patch/joint banks and the binary banks).

    python experiments/28_physical_energy.py
"""
import math
import pathlib
import sys
import time
from importlib import import_module

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.data import SEED                                    # noqa: E402
from chill_zen.energy import (BANDS, KBT, KSP_BACKBONE, KSP_PATCH,  # noqa: E402
                              LREP, NG, NPU, S, V_FLOOR,
                              g_sum_from_bytes, pooled_read,
                              sigma_parts, t_read)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
load_stack = import_module("07_verdict").load_stack
load_binary_stack = import_module("19_binary_verdict").load_binary_stack

GAIN_DEV = 0.02                # the device (never varied)
V_N_GRAY = 2e-3                # P4, register code 95
V_N_BIN = 3e-3                 # P6
V_N_REF = 5e-3                 # P5, at the 50 mV reference-read twin
V_REF = 0.050
BAND = "SnCr"                  # the band the emulator's constant reads as
B = 200


def hot_row(yv, mv, n_sel, v_read, pw=None):
    """(sigma_lane, gs, t_b) at the given (v_read, pw); pw=None means
    settling-limited (the physical draw's own timing)."""
    gs = g_sum_from_bytes(n_sel, mv.median().item(), BAND)
    t_b = pw if pw is not None else t_read(gs)
    s_th, s_fl = sigma_parts(yv, mv, v_read=v_read, pw=t_b)
    sigma_lane = (GAIN_DEV * torch.sqrt(s_th ** 2 + s_fl ** 2)).median().item()
    return sigma_lane, gs, t_b


def old_target_delivered(yv, mv, n_sel, band=BAND):
    """The earlier T-derived target/delivered pair (model B, solved back)."""
    from chill_zen.energy import sigma_target
    gs = g_sum_from_bytes(n_sel, mv.median().item(), band)
    t_b = t_read(gs)
    st = sigma_target(yv, mv).median().item()
    v_want = math.sqrt(2 * KBT / (t_b * gs * st ** 2))
    v_op = max(V_FLOOR, v_want)
    e_b = v_op ** 2 * gs * t_b
    sig_del = math.sqrt(2 * KBT / e_b)
    return st, v_want, e_b, sig_del


def report_class(name, n_sel, yv, mv, v_n, log):
    st, v_want, e_b_old, sig_del_old = old_target_delivered(yv, mv, n_sel)
    sig_lane, gs, t_b = hot_row(yv, mv, n_sel, V_FLOOR)
    sig_cmp = v_n / V_FLOOR
    sig_comp = math.sqrt(sig_lane ** 2 + sig_cmp ** 2)
    e_b_new = V_FLOOR ** 2 * gs * t_b
    ratio = e_b_new / e_b_old if e_b_old else float("nan")
    log(f"{name:<18}{'old':>6} target {st:.4f}  v_want {v_want * 1e3:.2f}mV"
        f"  e_bound {e_b_old:.3e}  sigma_del(B) {sig_del_old:.4f}")
    log(f"{'':<18}{'new':>6} sigma_lane(E) {sig_lane:.4f}  sigma_cmp {sig_cmp:.4f}"
        f"  sigma_composite {sig_comp:.4f}  e_bound {e_b_new:.3e}"
        f"  (ratio to old {ratio:.4f})")


def grayscale():
    log, fh = artifacts.make_log("28_physical_energy")
    log(f"\n=== physical-point energy reconciliation (grayscale) "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")
    log(f"GAIN_DEV {GAIN_DEV} (device), V_FLOOR {V_FLOOR * 1e3:.0f} mV, "
        f"band {BAND}; v_n grayscale {V_N_GRAY * 1e3:.1f} mV (P4), "
        f"binary {V_N_BIN * 1e3:.1f} mV (P6), reference {V_N_REF * 1e3:.1f} "
        f"mV at {V_REF * 1e3:.0f} mV (P5)")

    bb = torch.load(artifacts.path("backbone_banks"))[config.BACKBONE]
    pb = torch.load(artifacts.path("patch_banks"))
    jb = torch.load(artifacts.path("joint_bank"))
    st = torch.load(artifacts.path("verdict"))["st"]
    g_codes, p_codes = st["g_e2e"].long(), st["p_e2e"].long()
    lab = torch.arange(10).repeat_interleave(len(g_codes) // 10)
    g_c, p_c, y_c = g_codes[:B], p_codes[:B], lab[:B]

    log("\n-- backbone draw (ctx 0, 16, 64, 127) --")
    g1a, g1b = bb["g"]["ga"], bb["g"]["gb"]
    for k in (0, 16, 64, 127):
        aat = torch.full((B, KSP_BACKBONE), -1, dtype=torch.long)
        aat[:, :LREP] = y_c[:, None]
        if k > 0:
            aat[:, LREP:LREP + k] = g_c[:, :k]
        lanes = slice((1 + k) * S, (2 + k) * S)
        yv, mv = pooled_read(g1a[lanes], g1b[lanes], aat)
        report_class(f"backbone ctx{k}", LREP + k, yv.flatten(), mv.flatten(),
                     V_N_GRAY, log)

    log("\n-- patch draw --")
    g2a, g2b = pb["g"]["ga"], pb["g"]["gb"]
    aat = torch.cat([y_c[:, None].expand(-1, LREP), g_c,
                     torch.full((B, NPU), -1, dtype=torch.long)], 1)
    yv, mv = pooled_read(g2a, g2b, aat)
    report_class("patch", LREP + NG, yv.flatten(), mv.flatten(), V_N_GRAY, log)

    log("\n-- reference-read row (P5 twin: V=50mV, v_n=5mV, settling t_b "
        "unchanged) --")
    sig_lane_ref, gs_ref, t_b_ref = hot_row(yv.flatten(), mv.flatten(),
                                            LREP + NG, V_REF)
    sig_cmp_ref = V_N_REF / V_REF
    log(f"patch @ 50mV        sigma_lane {sig_lane_ref:.4f}  sigma_cmp "
        f"{sig_cmp_ref:.4f}  sigma_composite "
        f"{math.sqrt(sig_lane_ref ** 2 + sig_cmp_ref ** 2):.4f}  "
        f"(P5 measured 97-100% comparator, matches P3 on every judge)")

    log("\ncomparator energy: E_CMP = 1 fJ per lane per read (P2), already "
        "charged in the periphery total (12_energy_model.py); a raw-latch "
        "class starved to v_n = 2-3 mV rms prices the same as one held at "
        "full bias -- the noise setting is a bias schedule, not an extra "
        "component.")
    fh.close()
    return log


def binary():
    log, fh = artifacts.make_log("28_physical_energy")
    log(f"\n=== physical-point energy reconciliation (binary) "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")
    from chill_zen.data import N_TRAIN, load_fashion
    from chill_zen.generate import draw_backbone
    _, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    S_ = load_binary_stack(y_tr, y_ev)
    lab = torch.arange(10).repeat_interleave(B // 10)
    gen = torch.Generator().manual_seed(SEED + 980)
    g_c = draw_backbone(S_["bl"], S_["g1"], S_["r1"], lab, gen, t_gen=0.3)

    bstore = torch.load(artifacts.path("binary_backbone_banks"))
    bb = bstore["128x16@p0.5"]
    g1a, g1b = bb["g"]["ga"], bb["g"]["gb"]
    log("\n-- binary backbone draw (ctx 0, 16, 64, 127) --")
    for k in (0, 16, 64, 127):
        aat = torch.full((B, KSP_BACKBONE), -1, dtype=torch.long)
        aat[:, :LREP] = lab[:, None]
        if k > 0:
            aat[:, LREP:LREP + k] = g_c[:, :k]
        lanes = slice((1 + k) * S, (2 + k) * S)
        yv, mv = pooled_read(g1a[lanes], g1b[lanes], aat)
        report_class(f"bin backbone ctx{k}", LREP + k, yv.flatten(),
                     mv.flatten(), V_N_BIN, log)
    fh.close()


if __name__ == "__main__":
    grayscale()
    binary()
