"""Binary one-level Table VII rows at the physical operating point.

Table VII's "binary-trained one-level, 128/64 books, T = 0.2" rows were
drawn at the emulator's temperature dial. Sec. IV C states no physical
operating point produces `T = 0.2` before mid-chain either. This
script redraws both one-level binary arms with `chill_zen.physical` at
P6 (`V = 10 mV, v_n = 3 mV`, `v_n/V = 0.3`), the binary sweep's
optimum (`24_binary_comparator_sweep.py`). Declared rule (R3, before
running): P6 for both books counts, matching the two-level optimum --
not a fresh P4-vs-P6 search per book count, per the compute-economy
convention.

Renders to `head_to_head/gen/comparator-binary-one-level-{128,64}.npy`,
picked up by `head_to_head/07_comparator_score.py`'s `comparator-*`
glob alongside the grayscale physical-point arms.

Cost: about 5 minutes. Requires 17-18.

    python experiments/27_binary_comparator_headtohead.py
"""
import os
import pathlib
import sys
import time
from importlib import import_module

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts                                    # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.levels import BackboneLevel                         # noqa: E402
from chill_zen.physical import draw_backbone_physical               # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
bv = import_module("19_binary_verdict")
load_binary_stack, bin_render_backbone = bv.load_binary_stack, bv.bin_render_backbone

GEN = pathlib.Path(__file__).resolve().parent / "head_to_head" / "gen"
V_READ, V_N = 0.010, 3e-3     # P6: the binary sweep's optimum
N_PER_SCORE = 512
SEED_BLOCK = SEED + 23000


def save_npy(name, imgs, lab, log):
    GEN.mkdir(exist_ok=True)
    a = imgs.numpy().astype(np.float32)
    np.save(GEN / f"{name}.npy", a)
    np.save(GEN / f"{name}-lab.npy", lab.numpy().astype(np.int64))
    log(f"[render] {name}: {a.shape}  on-fraction {a.mean():.4f}")


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    log, fh = artifacts.make_log("27_binary_comparator_headtohead")
    log(f"binary one-level physical point, P6 V={V_READ * 1e3:.0f}mV "
        f"v_n={V_N * 1e3:.1f}mV ({time.strftime('%Y-%m-%d %H:%M')})")

    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    S = load_binary_stack(y_tr, y_ev)
    lab = torch.arange(10).repeat_interleave(N_PER_SCORE)

    # 128-book stack backbone (already loaded by load_binary_stack)
    gen = torch.Generator().manual_seed(SEED_BLOCK)
    t0 = time.time()
    g128, _ = draw_backbone_physical(S["bl"], S["g1"], S["r1"], lab, gen,
                                     V_READ, V_N)
    log(f"[draw] one-level-128: {len(lab)} codes ({time.time() - t0:.0f}s)")
    save_npy("comparator-binary-one-level-128",
             bin_render_backbone(g128, S["gb"]), lab, log)

    codes = torch.load(artifacts.path("binary_backbone_codes"))
    banks = torch.load(artifacts.path("binary_backbone_banks"))
    cs64 = codes["64x16@p0.5"]
    bl64 = BackboneLevel("64x16@p0.5", cs64["tr"].long(), cs64["ev"].long(),
                         y_tr, y_ev)
    g1_64 = bl64.make_bank(SEED + 100)
    g1_64.load_state_dict(banks["64x16@p0.5"]["g"])
    r1_64 = bl64.make_bank(SEED + 200)
    r1_64.load_state_dict(banks["64x16@p0.5"]["r"])
    gb64 = torch.load(artifacts.path("binary_backbone_books"))["64x16@p0.5"]
    gen = torch.Generator().manual_seed(SEED_BLOCK + 500)
    t0 = time.time()
    g64, _ = draw_backbone_physical(bl64, g1_64, r1_64, lab, gen, V_READ, V_N)
    log(f"[draw] one-level-64: {len(lab)} codes ({time.time() - t0:.0f}s)")
    save_npy("comparator-binary-one-level-64",
             bin_render_backbone(g64, gb64), lab, log)

    log("binary one-level physical render done")
    fh.close()


if __name__ == "__main__":
    main()
