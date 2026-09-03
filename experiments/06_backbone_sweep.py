"""Generate and judge from the backbone alone, across code shapes.

Paper Sec. IV E (a) and Fig. 5. The one-level system: label clamped,
M sampled autoregressive reads at T = 0.1 in book order, one cold
repair sweep, decode. n = 320 (32 per class).

Two readings come out of it. Generation improves with the number of
books at a fixed alphabet -- 64 to 128 books moves the critic from
0.806 to 0.834 -- and it degrades with alphabet depth at matched bits:
64 books of 32 symbols (320 bits) reaches 0.7969 where 128 books of 16
symbols (512 bits) reaches 0.8344. Deeper books are worse for
generation even where they are better for compression. The keep-p = 1
control is the plain additive quantization arm.

Seam ratios sit near real throughout, which is the point of a global
code: there are no tile boundaries to show.

The generated states are also the input to the per-size energy census
of experiment 12, so this must run before that one.

Cost: seconds to a couple of minutes per shape. Requires 01 and 03.

    python experiments/06_backbone_sweep.py --all
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, codec, config, judge              # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import T_GEN, draw_backbone                # noqa: E402
from chill_zen.levels import BackboneLevel, backbone_cfg           # noqa: E402

N_PER = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--points", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("06_backbone_sweep")

    points = args.points or (config.BACKBONE_POINTS if args.all
                             else config.FRONTIER)
    cfg = dict(seed=SEED, t_gen=T_GEN, n_per=N_PER, points=points)

    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN], y[N_TRAIN:]
    codes_store = torch.load(artifacts.path("backbone_codes"))
    books = torch.load(artifacts.path("backbone_books"))
    store = torch.load(artifacts.path("backbone_banks"))
    critic = judge.load_critic(artifacts.path("critic"))

    lab = torch.arange(10).repeat_interleave(N_PER)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER]
                          for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)

    spath = artifacts.path("backbone_sweep")
    saved = torch.load(spath) if spath.exists() else {}
    states = saved.get("states", {})
    res = saved.get("res", {})

    ra, rdv = judge.judge(critic, real, lab)
    log(f"\n{'point':>13}{'critic':>8}{'div':>8}{'tone':>7}{'seam':>7}")
    log(f"{'real':>13}{ra:>8.4f}{rdv:>8.4f}"
        f"{judge.tone_spread(real, lab):>7.3f}"
        f"{judge.seam_ratio(real):>7.3f}")

    for tag in points:
        if tag not in store:
            log(f"[sweep] {tag} has no banks — run 03 first; skipped")
            continue
        cs = codes_store[tag]
        level = BackboneLevel(tag, cs["tr"], cs["ev"], y_tr, y_ev)
        assert store[tag]["cfg"] == backbone_cfg(tag, level.M, level.NCH)
        if tag not in states:
            g1 = level.make_bank(SEED + 100)
            g1.load_state_dict(store[tag]["g"])
            r1 = level.make_bank(SEED + 200)
            r1.load_state_dict(store[tag]["r"])
            gen = torch.Generator().manual_seed(SEED + 700)
            t0 = time.time()
            states[tag] = draw_backbone(level, g1, r1, lab, gen,
                                        with_label=True).to(torch.int8)
            log(f"[sweep] {tag}: {len(lab)} states drawn "
                f"({time.time() - t0:.0f}s)")
            torch.save(dict(cfg=cfg, states=states, res=res), spath)
        b = books[tag]
        img = codec.decode_global(states[tag].long()[:, 1:], b["bias"],
                                  b["atoms"], b["cfg"]["p"]) \
            .reshape(-1, 28, 28).clamp(0, 1)
        a, dv = judge.judge(critic, img, lab)
        res[tag] = (a, dv, judge.tone_spread(img, lab), judge.seam_ratio(img))
        log(f"{tag:>13}{res[tag][0]:>8.4f}{res[tag][1]:>8.4f}"
            f"{res[tag][2]:>7.3f}{res[tag][3]:>7.3f}")
    torch.save(dict(cfg=cfg, states=states, res=res), spath)
    fh.close()


if __name__ == "__main__":
    main()
