"""Render 5120 images per arm from the frozen systems.

Runs in the emulator environment (torch), writes grayscale renders to
`gen/<arm>.npy` so `03_score.py`, which runs in the JAX environment,
can threshold and score them without either environment importing the
other's pins.

512 per class times 10 classes is their generated-side n. Nothing is
taught, nothing is tuned: every bank, book and recipe is loaded frozen
and driven exactly as Secs. III-IV deploy it. All arms render grayscale
in [0, 1]; the threshold is the only host-side step, and it is applied
in the scoring script, not here.

  two-level         the deployed system: 128 sampled autoregressive
                    reads at T = 0.1, one cold backbone sweep, one
                    parallel patch read at T = 0.1, one cold joint sweep
  one-level-{M}     the backbone-only series at 16, 32, 64, 128 books
  codec-ceiling     the codec's render of REAL train codes at 128 books.
                    Not a generator: the bound every arm through this
                    codec sits under
  real-train        real train images through the identical path

The ceiling and the control are taken on the TRAIN split because their
reference is the train split.

Cost: about 10 minutes. Requires experiments 01-05 of the main
sequence.

    python experiments/head_to_head/02_generate.py
"""
import argparse
import os
import pathlib
import sys
import time

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
from chill_zen.generate import T_GEN, draw_backbone, render        # noqa: E402
from chill_zen.levels import (BackboneLevel, backbone_cfg)         # noqa: E402

from importlib import import_module                           # noqa: E402
load_stack = import_module("07_verdict").load_stack

N_PER = 512                       # their fid_images_per_digit
SEED_BLOCK = ev.SEED_BLOCK + 100
ONE_LEVEL = (16, 32, 64, 128)


def save(name, imgs, lab, log):
    a = imgs.clamp(0, 1).numpy().astype(np.float32)
    np.save(ev.GEN / f"{name}.npy", a)
    np.save(ev.GEN / f"{name}-lab.npy", lab.numpy().astype(np.int64))
    log(f"[gen] {name}: {a.shape} mean {a.mean():.4f}  on-fraction at 0.1 "
        f"{(a > 0.1).mean():.4f}, at 0.5 {(a > 0.5).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    ap.add_argument("--threads", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = ev.make_log("02_generate")
    log(f"\n=== generate ({time.strftime('%Y-%m-%d %H:%M')}) — "
        f"{N_PER}/class = {N_PER * 10} per arm ===")

    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    lab = torch.arange(10).repeat_interleave(N_PER)
    books = torch.load(artifacts.path("backbone_books"))
    codes = torch.load(artifacts.path("backbone_codes"))
    banks = torch.load(artifacts.path("backbone_banks"))

    tr_idx = torch.cat([torch.where(y_tr == c)[0][:N_PER]
                        for c in range(10)])
    if not (ev.GEN / "real-train.npy").exists():
        save("real-train", X[:N_TRAIN, 0][tr_idx], lab, log)
    bb = books[args.backbone]
    if not (ev.GEN / "codec-ceiling.npy").exists():
        c = codes[args.backbone]["tr"].long()[tr_idx]
        dec = codec.decode_global(c, bb["bias"], bb["atoms"],
                                  bb["cfg"]["p"]).reshape(-1, 28, 28)
        save("codec-ceiling", dec, lab, log)

    for i, M in enumerate(ONE_LEVEL):
        name = f"one-level-{M}"
        if (ev.GEN / f"{name}.npy").exists():
            log(f"[gen] {name}: cached")
            continue
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
        g_open = draw_backbone(level, g1, r1, lab, gen)
        log(f"[gen] {name}: {len(lab)} codes drawn "
            f"({time.time() - t0:.0f}s)")
        bk = books[tag]
        dec = codec.decode_global(g_open, bk["bias"], bk["atoms"],
                                  bk["cfg"]["p"]).reshape(-1, 28, 28)
        save(name, dec, lab, log)

    if not (ev.GEN / "two-level.npy").exists():
        S = load_stack(args.backbone, args.patch)
        pl, jl = S["pl"], S["jl"]
        gen = torch.Generator().manual_seed(SEED_BLOCK)
        t0 = time.time()
        g_open = draw_backbone(S["bl"], S["g1"], S["r1"], lab, gen)
        log(f"[gen] two-level: backbone codes drawn "
            f"({time.time() - t0:.0f}s)")
        none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
        p_d0 = pl.read_all(S["g2"], lab, g_open, none, T_GEN, gen)
        pj = jl.read_joint(S["jbank"], lab, g_open, p_d0, 0.0, None)
        log(f"[gen] two-level: joint sweep done ({time.time() - t0:.0f}s)")
        rend, _ = render(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
        save("two-level", rend, lab, log)

    log("[gen] done")
    fh.close()


if __name__ == "__main__":
    main()
