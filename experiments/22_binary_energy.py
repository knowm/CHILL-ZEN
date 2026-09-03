"""The energy model on the binary census.

The same physical model as `12_energy_model.py` (decode charged,
per-band bounded point), evaluated on the binary banks' magnitudes
with contexts replayed from a fresh 200-state draw at the scored
optimum T = 0.3. Three systems: the two-level stack (census identical
in shape to the grayscale deployed system, 10,064 reads with the
decode), and the one-level 128 and 64 arms (M x 16 hot autoregressive
reads + M x 16 cold sweep reads + the 784-read decode).

Expected (Table VII and the Sec. V B flag):

    two-level      with periphery, SnCr bounded 1.871e-11 J
                   (grayscale post-decode 1.87e-11 -- within 1%)
    one-level 128  device bounded SnCr 3.64e-12 J
    one-level 64   device bounded SnCr 2.31e-12 J

The one-level rows land about 2.5x below their grayscale twins on like
terms (6.97e-12 / 3.47e-12 pre-decode), tracking the smaller
magnitudes the binary banks learned; the deviation is flagged in
Sec. V B and is not explained.

Cost: a few minutes. Requires 17 and 18.

    python experiments/22_binary_energy.py
"""
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts                                    # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.energy import (BANDS, C_LINE, E_ADDR, E_CMP, KBT,   # noqa: E402
                              KSP_BACKBONE, KSP_PATCH, LREP, NG, NPU, S,
                              V_COLD, V_FLOOR, V_LV, PW_ANCHOR,
                              g_sum_from_bytes, pooled_read, sigma_target,
                              t_read)
from chill_zen.generate import draw_backbone                       # noqa: E402
from chill_zen.levels import BackboneLevel                         # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from importlib import import_module                           # noqa: E402

load_binary_stack = import_module("19_binary_verdict").load_binary_stack

T_STAR = 0.3
B = 200


def band_totals(classes, log, tag):
    dev = {}
    for band in BANDS:
        dev[band] = {}
        for name, c in classes.items():
            eF, eB, eA = [], [], []
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
                else:
                    e_b = V_COLD ** 2 * gs * t_b
                    eB.append(e_b)
                    eF.append(e_b)
                eA.append(V_LV ** 2 * gs * PW_ANCHOR)
            w = ([S] * len(c["steps"]) if len(c["steps"]) > 1
                 else [c["nreads"]])
            tot = sum(w)
            dev[band][name] = dict(
                nreads=c["nreads"],
                e_floor=sum(e * wi for e, wi in zip(eF, w)) / tot,
                e_bound=sum(e * wi for e, wi in zip(eB, w)) / tot,
                e_anchor=sum(e * wi for e, wi in zip(eA, w)) / tot)
    n_reads = sum(d["nreads"] for d in dev["W"].values())
    out = {}
    for band in BANDS:
        out[band] = dict(
            bounded=sum(d["e_bound"] * d["nreads"]
                        for d in dev[band].values()),
            anchor=sum(d["e_anchor"] * d["nreads"]
                       for d in dev[band].values()))
    log(f"  [{tag}] census {n_reads} reads; device bounded "
        f"SnCr {out['SnCr']['bounded']:.3e} / W {out['W']['bounded']:.3e}; "
        f"anchor SnCr {out['SnCr']['anchor']:.3e} / W "
        f"{out['W']['anchor']:.3e} J")
    return out, n_reads


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("22_binary_energy")
    log(f"binary energy census ({time.strftime('%Y-%m-%d %H:%M')}) "
        f"T={T_STAR}")
    _, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    S_ = load_binary_stack(y_tr, y_ev)
    pl, jl = S_["pl"], S_["jl"]
    lab = torch.arange(10).repeat_interleave(B // 10)
    gen = torch.Generator().manual_seed(SEED + 980)
    g_c = draw_backbone(S_["bl"], S_["g1"], S_["r1"], lab, gen, t_gen=T_STAR)
    none = torch.full((B, pl.NPU), -1, dtype=torch.long)
    p_d = pl.read_all(S_["g2"], lab, g_c, none, 0.1, gen)
    pj = jl.read_joint(S_["jbank"], lab, g_c, p_d, 0.0, None)
    g_c, p_c = pj[:, :jl.NG], pj[:, jl.NG:]

    bstore = torch.load(artifacts.path("binary_backbone_banks"))
    pstore = torch.load(artifacts.path("binary_patch_banks"))
    jstore = torch.load(artifacts.path("binary_joint_bank"))
    results = {}

    # ---- two-level census (mirrors 12) ----
    bb = bstore["128x16@p0.5"]
    classes = {}
    g1a, g1b = bb["g"]["ga"], bb["g"]["gb"]
    hot_rows = []
    for k in range(NG):
        aat = torch.full((B, KSP_BACKBONE), -1, dtype=torch.long)
        aat[:, :LREP] = lab[:, None]
        if k > 0:
            aat[:, LREP:LREP + k] = g_c[:, :k]
        lanes = slice((1 + k) * S, (2 + k) * S)
        yv, mv = pooled_read(g1a[lanes], g1b[lanes], aat)
        hot_rows.append((LREP + k, yv.flatten(), mv.flatten()))
    classes["backbone draw"] = dict(kind="hot", nreads=NG * S,
                                    steps=hot_rows)
    r1a, r1b = bb["r"]["ga"], bb["r"]["gb"]
    aat = torch.cat([lab[:, None].expand(-1, LREP), g_c], 1)
    yv, mv = pooled_read(r1a[S:], r1b[S:], aat)
    classes["backbone sweep"] = dict(
        kind="cold", nreads=NG * S,
        steps=[(KSP_BACKBONE, yv.flatten(), mv.flatten())])
    g2a, g2b = pstore["g"]["ga"], pstore["g"]["gb"]
    aat = torch.cat([lab[:, None].expand(-1, LREP), g_c,
                     torch.full((B, NPU), -1, dtype=torch.long)], 1)
    yv, mv = pooled_read(g2a, g2b, aat)
    classes["patch draw"] = dict(kind="hot", nreads=NPU * S,
                                 steps=[(LREP + NG, yv.flatten(),
                                         mv.flatten())])
    ja, jbb = jstore["bank"]["ga"], jstore["bank"]["gb"]
    aat = torch.cat([lab[:, None].expand(-1, LREP), g_c, p_c], 1)
    yv, mv = pooled_read(ja, jbb, aat)
    classes["joint sweep"] = dict(kind="cold", nreads=(NG + NPU) * S,
                                  steps=[(KSP_PATCH, yv.flatten(),
                                          mv.flatten())])
    classes["decode"] = dict(kind="cold", nreads=784,
                             steps=[(KSP_PATCH, yv.flatten(), mv.flatten())])
    log("\ntwo-level (binary banks):")
    two, n_reads = band_totals(classes, log, "two-level")
    v_hot_op = max(V_FLOOR, 0.012)
    e_drive = (sum(LREP + k for k in range(NG)) * C_LINE * v_hot_op ** 2
               + KSP_BACKBONE * C_LINE * V_COLD ** 2
               + (LREP + NG) * C_LINE * v_hot_op ** 2
               + KSP_PATCH * C_LINE * V_COLD ** 2)
    e_periph = e_drive + n_reads * E_CMP + (NG + NG + NPU + (NG + NPU)
                                            + 1) * E_ADDR
    log(f"  periphery {e_periph:.2e} J; WITH PERIPHERY bounded "
        f"SnCr {two['SnCr']['bounded'] + e_periph:.3e} J "
        f"(grayscale post-decode 1.87e-11)")
    results["two-level"] = dict(bands=two, periph=e_periph,
                                with_periph=two["SnCr"]["bounded"] + e_periph,
                                n_reads=n_reads)

    # ---- one-level arms ----
    codes = torch.load(artifacts.path("binary_backbone_codes"))
    for tag, gray in (("128x16@p0.5", 6.97e-12), ("64x16@p0.5", 3.47e-12)):
        cs = codes[tag]
        bl = BackboneLevel(tag, cs["tr"].long(), cs["ev"].long(),
                           y_tr, y_ev)
        g1 = bl.make_bank(SEED + 100)
        g1.load_state_dict(bstore[tag]["g"])
        r1 = bl.make_bank(SEED + 200)
        r1.load_state_dict(bstore[tag]["r"])
        gen = torch.Generator().manual_seed(SEED + 981)
        gc = draw_backbone(bl, g1, r1, lab, gen, t_gen=T_STAR)
        M = bl.M
        ksp = LREP + M
        cl = {}
        ga, gb = bstore[tag]["g"]["ga"], bstore[tag]["g"]["gb"]
        rows = []
        for k in range(M):
            aat = torch.full((B, ksp), -1, dtype=torch.long)
            aat[:, :LREP] = lab[:, None]
            if k > 0:
                aat[:, LREP:LREP + k] = gc[:, :k]
            lanes = slice((1 + k) * S, (2 + k) * S)
            yv, mv = pooled_read(ga[lanes], gb[lanes], aat)
            rows.append((LREP + k, yv.flatten(), mv.flatten()))
        cl["backbone draw"] = dict(kind="hot", nreads=M * S, steps=rows)
        ra, rb = bstore[tag]["r"]["ga"], bstore[tag]["r"]["gb"]
        aat = torch.cat([lab[:, None].expand(-1, LREP), gc], 1)
        yv, mv = pooled_read(ra[S:], rb[S:], aat)
        cl["backbone sweep"] = dict(kind="cold", nreads=M * S,
                                    steps=[(ksp, yv.flatten(),
                                            mv.flatten())])
        cl["decode"] = dict(kind="cold", nreads=784,
                            steps=[(ksp, yv.flatten(), mv.flatten())])
        log(f"\none-level {tag} (binary banks; grayscale device-bounded "
            f"pre-decode {gray:.2e}):")
        out, nr = band_totals(cl, log, tag)
        results[tag] = dict(bands=out, n_reads=nr)

    torch.save(dict(results=results, t_star=T_STAR),
               artifacts.path("binary_energy"))
    log("\nsaved: binary-energy-results.pt")
    fh.close()


if __name__ == "__main__":
    main()
