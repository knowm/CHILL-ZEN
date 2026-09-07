"""Render every grayscale arm of Table VII at every level of the register grid.

`25_comparator_headtohead.py` renders the grayscale arms at one point,
the critic's optimum for the two-level stack. The critic and the FID do
not agree on that point, and the one-level frontier was never swept at
all, so this script renders each arm -- the two-level stack and the
one-level 16/32/64/128-book frontier -- at each level of the grid of
Sec. IV C at `V = 10 mV`: P0 (lanes only), P1 160 uV, P2 500 uV,
P3 1 mV, P4 2 mV, P6 3 mV. Same recipe as 25 in every other respect
(frozen banks, no teaching, no tuning), same seeds per arm, so the P4
renders reproduce 25's bit for bit. Renders go to
`gen/comparator-gray-{two-level,one-level-M}-{P}.npy`;
`07_comparator_score.py` scores them and prints each arm's best level,
which `15_energy_figure.py` and `16_binarization_figure.py` read.

`--fifty` renders the one-level arms at `V = 50 mV` instead, at register
codes 0 and 10 (Q0 100 uV, Q1 300 uV; v_n/V 0.002 and 0.006). The
temperature is v_n/V (Sec. IV C's 50 mV twin), and at 10 mV the register
floor is already 0.01 of the read, so an arm whose FID is still falling
at P1 has no physical point below it on the 10 mV grid. The 50 mV read
puts one there. Renders go to `gen/comparator-gray-one-level-M-{Q0,Q1}.npy`
and 07 scores them with the rest.

Cost: about 45 minutes; `--fifty` about 6. Requires 01-05.

    python experiments/head_to_head/34_grayscale_grid.py
    python experiments/head_to_head/34_grayscale_grid.py --fifty
"""
import argparse
import os
import pathlib
import sys
import time
from importlib import import_module

os.environ.setdefault("OMP_NUM_THREADS", "5")

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))

import dtm_fid as ev                                        # noqa: E402
from chill_zen import artifacts, codec, config                     # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import render                              # noqa: E402
from chill_zen.levels import BackboneLevel, backbone_cfg           # noqa: E402
from chill_zen.physical import (draw_backbone_physical,            # noqa: E402
                                read_patches_physical, register_level)

load_stack = import_module("07_verdict").load_stack

N_PER = 512
V_READ = 0.010
GRID = {"P0": 0.0, "P1": 160e-6, "P2": 500e-6, "P3": 1e-3, "P4": 2e-3,
        "P6": 3e-3}                                # v_n on the register
V_FIFTY = 0.050
GRID_FIFTY = {"Q0": 100e-6, "Q1": 300e-6}          # codes 0 and 10
SEED_BLOCK = ev.SEED_BLOCK + 200                   # as in 25
ONE_LEVEL = (16, 32, 64, 128)


def save(name, imgs, lab, log):
    a = imgs.clamp(0, 1).numpy().astype(np.float32)
    np.save(ev.GEN / f"{name}.npy", a)
    np.save(ev.GEN / f"{name}-lab.npy", lab.numpy().astype(np.int64))
    log(f"[gen] {name}: {a.shape} mean {a.mean():.4f}  on-fraction at 0.1 "
        f"{(a > 0.1).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fifty", action="store_true",
                    help="one-level arms at 50 mV, codes 0 and 10")
    args = ap.parse_args()
    v_read = V_FIFTY if args.fifty else V_READ
    grid = GRID_FIFTY if args.fifty else GRID
    torch.set_num_threads(5)
    log, fh = ev.make_log("34_grayscale_grid" + ("_fifty" if args.fifty else ""))
    log(f"\n=== grayscale register-grid render "
        f"({time.strftime('%Y-%m-%d %H:%M')}) — V = {v_read * 1e3:.0f} mV, "
        f"points {', '.join(f'{p} {v * 1e6:.0f}uV' for p, v in grid.items())} ===")
    for v_n in grid.values():
        if v_n:
            register_level(v_n)

    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    lab = torch.arange(10).repeat_interleave(N_PER)
    books = torch.load(artifacts.path("backbone_books"))
    codes = torch.load(artifacts.path("backbone_codes"))
    banks = torch.load(artifacts.path("backbone_banks"))
    S = load_stack(config.BACKBONE, config.PATCH)

    # No cache, for the reason 25 gives: a render carries no record of
    # the banks that drew it.
    for P, v_n in grid.items():
        for i, M in enumerate(ONE_LEVEL):
            name = f"comparator-gray-one-level-{M}-{P}"
            tag = f"{M}x16@p0.5"
            gc = codes[tag]
            level = BackboneLevel(tag, gc["tr"], gc["ev"], y_tr, y_ev)
            if banks[tag]["cfg"] != backbone_cfg(tag, level.M, level.NCH):
                raise RuntimeError(f"backbone bank configuration mismatch at {tag}")
            g1 = level.make_bank(SEED + 100)
            g1.load_state_dict(banks[tag]["g"])
            r1 = level.make_bank(SEED + 200)
            r1.load_state_dict(banks[tag]["r"])
            gen = torch.Generator().manual_seed(SEED_BLOCK + 1 + i)
            t0 = time.time()
            g_open, _ = draw_backbone_physical(level, g1, r1, lab, gen,
                                               v_read, v_n)
            log(f"[gen] {name}: {len(lab)} codes drawn ({time.time() - t0:.0f}s)")
            bk = books[tag]
            dec = codec.decode_global(g_open, bk["bias"], bk["atoms"],
                                      bk["cfg"]["p"]).reshape(-1, 28, 28)
            save(name, dec, lab, log)

        if args.fifty:
            continue
        name = f"comparator-gray-two-level-{P}"
        pl, jl = S["pl"], S["jl"]
        gen = torch.Generator().manual_seed(SEED_BLOCK)
        t0 = time.time()
        g_open, _ = draw_backbone_physical(S["bl"], S["g1"], S["r1"], lab, gen,
                                           V_READ, v_n)
        p_d, _ = read_patches_physical(pl, S["g2"], lab, g_open, gen,
                                       V_READ, v_n)
        pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
        log(f"[gen] {name}: drawn and swept ({time.time() - t0:.0f}s)")
        rend, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
        save(name, rend, lab, log)

    log("[gen] done")
    fh.close()


if __name__ == "__main__":
    main()
