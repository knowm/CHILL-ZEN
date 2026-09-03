"""Free-label draw: the two-level arms as unconditional generators.

Every FID the paper reports is drawn with the label given (512 per
class). The DTM replication scores both modes every epoch — free
(unconditional) and clamped (label-conditional) — and its clamped best
beats its free best in all 44 shipped runs (median gap 3.0 FID at the
best epoch). To compare like to like, this script draws the two-level
arms unconditionally: the label is sampled from the uniform prior
(`torch.randint`, declared seed) and then the draw proceeds exactly as
the physical-point arms — same `chill_zen.physical` path, same
operating points (grayscale P3, binary P6), n = 5120. The label is a
sampled latent, so the sample set is a draw from the unconditional
model.

Renders to `head_to_head/gen/audit-free-{two-level,binary-two-level}
.npy`, scored by `head_to_head/32_audit_score.py`.

Cost: about 25 minutes. Requires 01-05 and 17-19.

    OMP_NUM_THREADS=8 nice -19 python experiments/31_free_draw.py
"""
import os
import pathlib
import sys
import time
from importlib import import_module

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "head_to_head"))
assert not any("drafts" in p for p in sys.path), "R1: no drafts imports"

from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import render                              # noqa: E402
from chill_zen.physical import (draw_backbone_physical,            # noqa: E402
                                read_patches_physical)

m24 = import_module("24_binary_comparator_sweep")
load_stack = import_module("07_verdict").load_stack

GEN = HERE / "head_to_head" / "gen"
N_GEN = 5120
SEED_LAB = SEED + 24100     # the label draw
SEED_BLOCK = SEED + 24000   # the image draw
P3 = (0.010, 1e-3)          # grayscale two-level operating point
P6 = (0.010, 3e-3)          # binary two-level operating point


def save(name, imgs, lab, log):
    GEN.mkdir(exist_ok=True)
    a = imgs.clamp(0, 1).numpy().astype(np.float32)
    np.save(GEN / f"{name}.npy", a)
    np.save(GEN / f"{name}-lab.npy", lab.numpy().astype(np.int64))
    log(f"[gen] {name}: {a.shape}  mean {a.mean():.4f}")


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("31_free_draw")
    log(f"\n=== free-label draw, n = {N_GEN} "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")

    glab = torch.Generator().manual_seed(SEED_LAB)
    lab = torch.randint(0, 10, (N_GEN,), generator=glab)
    counts = torch.bincount(lab, minlength=10).tolist()
    log(f"labels ~ uniform prior, seed {SEED_LAB}, class counts {counts}")

    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()

    # binary two-level at P6, the arm behind the 11.10
    S = m24.load_binary_stack(y_tr, y_ev)
    jl = S["jl"]
    gen = torch.Generator().manual_seed(SEED_BLOCK)
    t0 = time.time()
    pj, _ = m24.states_physical(S, lab, gen, *P6)
    img = m24.bin_render_two(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
    log(f"[draw] binary two-level at P6: {time.time() - t0:.0f}s")
    save("audit-free-binary-two-level", img, lab, log)

    # grayscale two-level at P3, the arm behind the 18.73
    S2 = load_stack(config.BACKBONE, config.PATCH)
    pl2, jl2 = S2["pl"], S2["jl"]
    gen = torch.Generator().manual_seed(SEED_BLOCK + 500)
    t0 = time.time()
    g_open, _ = draw_backbone_physical(S2["bl"], S2["g1"], S2["r1"], lab, gen,
                                       *P3)
    p_d, _ = read_patches_physical(pl2, S2["g2"], lab, g_open, gen, *P3)
    pj2 = jl2.read_joint(S2["jbank"], lab, g_open, p_d, 0.0, None)
    rend, _ = render(pj2[:, :jl2.NG], pj2[:, jl2.NG:], S2["gb"], S2["kb"])
    log(f"[draw] grayscale two-level at P3: {time.time() - t0:.0f}s")
    save("audit-free-two-level", rend, lab, log)

    log("done; score with head_to_head/32_audit_score.py")
    fh.close()


if __name__ == "__main__":
    main()
