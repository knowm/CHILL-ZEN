"""Joules per image for the deployed two-level system.

Paper Sec. V B. A physical model, not a measurement. Three accountings
are carried for every read class and all three are reported:

  FLOOR    the thermodynamic 2kBT/sigma^2 at the noise the read
           actually delivers -- device-independent, by Eq. (2)
  BOUNDED  the floors-bounded operating point: a 10 mV sense floor and
           settling-limited timing, t = max(2 ns, 5RC)
  ANCHOR   the emulator's literal reference read, 50 mV for 1 us, on
           each of the two device bands

The census is exact, taken from the bank shapes rather than estimated:
2048 hot autoregressive reads, 1568 hot patch reads, 2048 and 3616 cold
sweep reads, and 784 cold decode reads -- 10,064 lane reads per image
over 226 addresses, in 132 sequential steps.

Periphery (P1-P3: line charging, one comparator per lane, address
select and clock) is added on top, mirroring the E_bias / E_clock /
E_comm decomposition the DTCA paper uses, so the two totals are
comparable in kind.

Expected: a device-level bounded total of 8.24e-12 J on the tin/chromium
band and 5.80e-11 J on tungsten; periphery of 1.05e-11 J, comparator-
dominated -- at the bounded point the periphery, not the devices, is the
cost. With periphery: 1.87e-11 J bounded, 1.58e-09 J at the tin/chromium
anchor, and 3.37e-08 J at the tungsten anchor. That last number is
larger than the DTM chain's 1.57e-08 and is stated as such.

Per drawn bit the thermal draw costs 9e-17 to 4.5e-16 J (2e4 to 1e5
kBT), against 2e-15 J per Bernoulli sample for the DTCA's dedicated RNG
cell.

Cost: about a minute. Requires 03-05, 07 and 11.

    python experiments/12_energy_model.py
"""
import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.energy import (BANDS, C_LINE, E_ADDR, E_CMP, KBT,   # noqa: E402
                         KSP_BACKBONE, KSP_PATCH, LREP, NG, NPU, S,
                         V_COLD, V_FLOOR, V_LV, PW_ANCHOR,
                         g_sum_from_bytes, pooled_read, sigma_target,
                         t_read)

DTM_BEST_J = 1.568e-8          # their deepest chain, from their Fig. 1
DTM_RNG_J = 2e-15              # their per-Bernoulli-sample RNG cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    args = ap.parse_args()
    log, fh = artifacts.make_log("12_energy_model")
    log("energy model — A1-A4 and P1-P3 are in chill_zen/energy.py")

    bb = torch.load(artifacts.path("backbone_banks"))[args.backbone]
    pb = torch.load(artifacts.path("patch_banks"))
    jb = torch.load(artifacts.path("joint_bank"))
    st = torch.load(artifacts.path("verdict"))["st"]
    g_codes, p_codes = st["g_e2e"].long(), st["p_e2e"].long()
    lab = torch.arange(10).repeat_interleave(len(g_codes) // 10)
    B = 200
    g_c, p_c, y_c = g_codes[:B], p_codes[:B], lab[:B]

    classes = {}
    g1a, g1b = bb["g"]["ga"], bb["g"]["gb"]
    hot_rows = []
    for k in range(NG):
        aat = torch.full((B, KSP_BACKBONE), -1, dtype=torch.long)
        aat[:, :LREP] = y_c[:, None]
        if k > 0:
            aat[:, LREP:LREP + k] = g_c[:, :k]
        lanes = slice((1 + k) * S, (2 + k) * S)
        yv, mv = pooled_read(g1a[lanes], g1b[lanes], aat)
        hot_rows.append((LREP + k, yv.flatten(), mv.flatten()))
    classes["backbone draw"] = dict(kind="hot", nreads=NG * S, steps=hot_rows)

    r1a, r1b = bb["r"]["ga"], bb["r"]["gb"]
    aat = torch.cat([y_c[:, None].expand(-1, LREP), g_c], 1)
    yv, mv = pooled_read(r1a[S:], r1b[S:], aat)
    classes["backbone sweep"] = dict(
        kind="cold", nreads=NG * S,
        steps=[(KSP_BACKBONE, yv.flatten(), mv.flatten())])

    g2a, g2b = pb["g"]["ga"], pb["g"]["gb"]
    aat = torch.cat([y_c[:, None].expand(-1, LREP), g_c,
                     torch.full((B, NPU), -1, dtype=torch.long)], 1)
    yv, mv = pooled_read(g2a, g2b, aat)
    classes["patch draw"] = dict(
        kind="hot", nreads=NPU * S,
        steps=[(LREP + NG, yv.flatten(), mv.flatten())])

    ja, jbb = jb["bank"]["ga"], jb["bank"]["gb"]
    aat = torch.cat([y_c[:, None].expand(-1, LREP), g_c, p_c], 1)
    yv, mv = pooled_read(ja, jbb, aat)
    classes["joint sweep"] = dict(
        kind="cold", nreads=(NG + NPU) * S,
        steps=[(KSP_PATCH, yv.flatten(), mv.flatten())])

    # The decode, charged as a lane operation (review E; E3d).
    # decode_global is bias + p * sum_m atoms[m][code_m] -- the pooled
    # read of Eq. (5) with a constant denominator, since every book is
    # present at decode time. 784 output lanes over the same 226-address
    # context as the joint sweep, one parallel group read. Magnitudes
    # are proxied by the joint sweep's pooled read at the same context
    # size. It runs host-side today (Limitation 7); it is charged here
    # so neither side of the comparison carries an unbudgeted step.
    classes["decode"] = dict(kind="cold", nreads=784,
                             steps=[(KSP_PATCH, yv.flatten(), mv.flatten())])

    log("\nper-class device-level energies (per lane read, medians):")
    log(f"{'band':<6}{'class':<18}{'floor':>10}{'bounded':>10}{'anchor':>10}"
        f"{'sigma tgt':>11}{'sigma del':>11}")
    dev = {}
    for band in BANDS:
        dev[band] = {}
        for name, c in classes.items():
            eF, eB, eA, sig_t, sig_d = [], [], [], [], []
            for n_sel, yv, mv in c["steps"]:
                gs = g_sum_from_bytes(n_sel, mv.median().item(), band)
                t_b = t_read(gs)
                if c["kind"] == "hot":
                    st_tgt = sigma_target(yv, mv).median().item()
                    e_floor = 2 * KBT / st_tgt ** 2
                    v_op = max(V_FLOOR,
                               math.sqrt(2 * KBT / (t_b * gs * st_tgt ** 2)))
                    e_b = v_op ** 2 * gs * t_b
                    eF.append(e_floor)
                    eB.append(e_b)
                    sig_t.append(st_tgt)
                    sig_d.append(math.sqrt(2 * KBT / e_b))
                else:
                    e_b = V_COLD ** 2 * gs * t_b
                    eB.append(e_b)
                    eF.append(e_b)          # a cold read has no draw floor
                    sig_t.append(0.0)
                    sig_d.append(math.sqrt(2 * KBT / e_b))
                eA.append(V_LV ** 2 * gs * PW_ANCHOR)
            w = ([S] * len(c["steps"]) if name == "backbone draw"
                 else [c["nreads"]])
            tot = sum(w)
            dev[band][name] = dict(
                kind=c["kind"], nreads=c["nreads"],
                e_floor=sum(e * wi for e, wi in zip(eF, w)) / tot,
                e_bound=sum(e * wi for e, wi in zip(eB, w)) / tot,
                e_anchor=sum(e * wi for e, wi in zip(eA, w)) / tot,
                sig_tgt=sum(sig_t) / len(sig_t),
                sig_del=sum(sig_d) / len(sig_d))
            d = dev[band][name]
            log(f"{band:<6}{name:<18}{d['e_floor']:>10.2e}"
                f"{d['e_bound']:>10.2e}{d['e_anchor']:>10.2e}"
                f"{d['sig_tgt']:>11.1e}{d['sig_del']:>11.1e}")

    n_reads = sum(d["nreads"] for d in dev["W"].values())
    hot_reads = classes["backbone draw"]["nreads"] + \
        classes["patch draw"]["nreads"]
    log(f"\ncensus: {n_reads} lane reads per image "
        f"(hot {hot_reads}, cold {n_reads - hot_reads}) over "
        f"{NG + NPU} addresses, in {NG + 4} sequential steps")
    log(f"  of which decode: {classes['decode']['nreads']} cold lane reads, "
        "one parallel group read (magnitudes proxied by the joint sweep's "
        "pooled read; runs host-side today, Limitation 7)")
    band_tot = {}
    for band in BANDS:
        band_tot[band] = dict(
            floor=sum(d["e_floor"] * d["nreads"] for d in dev[band].values()),
            bounded=sum(d["e_bound"] * d["nreads"] for d in dev[band].values()),
            anchor=sum(d["e_anchor"] * d["nreads"] for d in dev[band].values()))
        log(f"  [{band}] floor {band_tot[band]['floor']:.3e}  bounded "
            f"{band_tot[band]['bounded']:.3e}  anchor "
            f"{band_tot[band]['anchor']:.3e} J")
    # Per band, not min() over bands (review E): on the tin/chromium
    # band the 5RC settling time binds; on tungsten the 2 ns
    # electronics floor binds, so "bounded below by 5CV^2 regardless of
    # band" holds only where settling binds.
    tot_B = band_tot["SnCr"]["bounded"]
    tot_B_w = band_tot["W"]["bounded"]
    tot_A = band_tot["SnCr"]["anchor"]
    tot_A_w = band_tot["W"]["anchor"]
    tot_F = band_tot["W"]["floor"]

    # periphery, P1-P3
    v_hot_op = max(V_FLOOR, 0.012)
    e_drive = (sum(LREP + k for k in range(NG)) * C_LINE * v_hot_op ** 2
               + KSP_BACKBONE * C_LINE * V_COLD ** 2
               + (LREP + NG) * C_LINE * v_hot_op ** 2
               + KSP_PATCH * C_LINE * V_COLD ** 2)
    e_cmp = n_reads * E_CMP
    n_group_reads = NG + NG + NPU + (NG + NPU) + 1   # + the decode
    e_addr = n_group_reads * E_ADDR
    e_periph = e_drive + e_cmp + e_addr
    log(f"\nperiphery per image: line charging {e_drive:.2e} + comparators "
        f"{e_cmp:.2e} + address/clock {e_addr:.2e} = {e_periph:.2e} J")

    lo, hi, hi_w = tot_B + e_periph, tot_A + e_periph, tot_A_w + e_periph
    lo_w = tot_B_w + e_periph
    log(f"\nWITH PERIPHERY, per image (tin/chromium band): {lo:.2e} .. "
        f"{hi:.2e} J")
    log(f"  tungsten band: bounded {lo_w:.2e} .. anchor {hi_w:.2e} J")
    log(f"  devices alone: bounded SnCr {tot_B:.2e} / W {tot_B_w:.2e}; "
        f"anchor SnCr {tot_A:.2e} / W {tot_A_w:.2e}")

    log("\ncomparisons:")
    log(f"  their deepest DTM chain: {DTM_BEST_J:.3e} J/sample -> "
        f"bounded {DTM_BEST_J / lo:.0f}x less, tin/chromium anchor "
        f"{DTM_BEST_J / hi:.1f}x less, tungsten anchor "
        f"{hi_w / DTM_BEST_J:.1f}x MORE")
    log(f"  at the tungsten band's bounded point: {DTM_BEST_J / lo_w:.0f}x "
        "less")
    hot = [d for d in dev["W"].values() if d["kind"] == "hot"]
    e_hot_bit = (sum(d["e_bound"] * d["nreads"] for d in hot)
                 / sum(d["nreads"] for d in hot)) * S / 4
    e_flr_bit = (sum(d["e_floor"] * d["nreads"] for d in hot)
                 / sum(d["nreads"] for d in hot)) * S / 4
    log(f"  per drawn bit (16 lanes carry 4 bits): bounded {e_hot_bit:.2e} J "
        f"= {e_hot_bit / KBT:.1e} kBT; at the thermodynamic floor "
        f"{e_flr_bit:.2e} J = {e_flr_bit / KBT:.1e} kBT")
    log(f"  their RNG cell: {DTM_RNG_J:.0e} J = {DTM_RNG_J / KBT:.1e} kBT "
        "per Bernoulli sample")

    torch.save(dict(dev=dev, periph=dict(drive=e_drive, cmp=e_cmp,
                                         addr=e_addr),
                    band_tot=band_tot,
                    totals=dict(floor=tot_F, bounded=tot_B,
                                bounded_w=tot_B_w, anchor=tot_A,
                                anchor_w=tot_A_w, periph=e_periph,
                                lo=lo, lo_w=lo_w, hi=hi, hi_w=hi_w),
                    census=dict(n_reads=n_reads,
                                per_class={n: d["nreads"]
                                           for n, d in dev["W"].items()})),
               artifacts.path("energy_model"))
    log(f"\nsaved: {artifacts.path('energy_model').name}")
    fh.close()


if __name__ == "__main__":
    main()
