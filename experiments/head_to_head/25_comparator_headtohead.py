"""Render CHILL ZEN's Table VII rows at the physical operating point.

`02_generate.py` rendered the two-level system and the one-level
frontier (16/32/64/128 books) at the emulator's `T = 0.1` draw, which
Sec. IV C states no physical operating point can produce. This script
redraws the same arms with `chill_zen.physical` at P4
(`V = 10 mV, v_n = 2 mV`, `v_n/V = 0.2`, register code 95), the
grayscale sweep's best physical point (`23_comparator_sweep.py`). Same recipe as `02_generate.py` in every other respect:
frozen banks, no teaching, no tuning. Renders go to
`gen/comparator-{two-level,one-level-M}.npy`; `07_comparator_score.py`
scores them against the same reference as `03_score.py`.

Cost: about 10 minutes. Requires 01-05.

    python experiments/head_to_head/25_comparator_headtohead.py
"""
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
                                read_patches_physical)

load_stack = import_module("07_verdict").load_stack

N_PER = 512
V_READ, V_N = 0.010, 2e-3     # P4: the grayscale sweep's best physical point
SEED_BLOCK = ev.SEED_BLOCK + 200
ONE_LEVEL = (16, 32, 64, 128)


def save(name, imgs, lab, log):
    a = imgs.clamp(0, 1).numpy().astype(np.float32)
    np.save(ev.GEN / f"{name}.npy", a)
    np.save(ev.GEN / f"{name}-lab.npy", lab.numpy().astype(np.int64))
    log(f"[gen] {name}: {a.shape} mean {a.mean():.4f}  on-fraction at 0.1 "
        f"{(a > 0.1).mean():.4f}")


def main():
    torch.set_num_threads(5)
    log, fh = ev.make_log("25_comparator_headtohead")
    log(f"\n=== physical-point head-to-head render "
        f"({time.strftime('%Y-%m-%d %H:%M')}) — P4 V={V_READ * 1e3:.0f}mV "
        f"v_n={V_N * 1e6:.0f}uV ===")

    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    lab = torch.arange(10).repeat_interleave(N_PER)
    books = torch.load(artifacts.path("backbone_books"))
    codes = torch.load(artifacts.path("backbone_codes"))
    banks = torch.load(artifacts.path("backbone_banks"))

    # No cache. These renders carry no record of the banks that drew them,
    # so an existence check hands the scorer the previous banks' images
    # under the current banks' name -- a stale FID with nothing to show
    # for it. Ten minutes is cheaper than that.
    for i, M in enumerate(ONE_LEVEL):
        name = f"comparator-one-level-{M}"
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
        g_open, _ = draw_backbone_physical(level, g1, r1, lab, gen, V_READ, V_N)
        log(f"[gen] {name}: {len(lab)} codes drawn "
            f"({time.time() - t0:.0f}s)")
        bk = books[tag]
        dec = codec.decode_global(g_open, bk["bias"], bk["atoms"],
                                  bk["cfg"]["p"]).reshape(-1, 28, 28)
        save(name, dec, lab, log)

    # Likewise no cache, for the same reason.
    S = load_stack(config.BACKBONE, config.PATCH)
    pl, jl = S["pl"], S["jl"]
    gen = torch.Generator().manual_seed(SEED_BLOCK)
    t0 = time.time()
    g_open, _ = draw_backbone_physical(S["bl"], S["g1"], S["r1"], lab, gen,
                                       V_READ, V_N)
    log(f"[gen] comparator-two-level: backbone codes drawn "
        f"({time.time() - t0:.0f}s)")
    p_d, _ = read_patches_physical(pl, S["g2"], lab, g_open, gen,
                                   V_READ, V_N)
    pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
    log(f"[gen] comparator-two-level: joint sweep done "
        f"({time.time() - t0:.0f}s)")
    rend, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
    save("comparator-two-level", rend, lab, log)

    log("[gen] done")
    fh.close()


if __name__ == "__main__":
    main()
