"""The two-layer verdict at n = 1000 on a fresh seed.

Paper Sec. IV B, Table II, Fig. 3 (Table II carries the three-seed version). Two arms, 100 per class, on a seed
independent of everything used to build the stack:

* **end-to-end** -- label only. 128 sampled autoregressive reads at
  T = 0.1, one cold backbone sweep, one parallel patch read at T = 0.1,
  one cold joint sweep, render.
* **reconstruction** -- the same stack with exactly one component
  replaced: the draw gives way to a real evaluation image's encoded
  backbone code. Everything downstream runs together and exactly as
  deployed.

The difference between the two arms is the draw, and nothing else.

Expected (evaluation noise at this n is about +/-0.005):

    arm              critic     div    tone   seam   std-ratio
    real             0.8860  0.4866   2.362  1.067        1.00
    reconstruction   0.8860  0.5016   2.070  1.035        1.11
    end-to-end       0.8340  0.4621   2.390  1.064        1.24

The reconstruction arm reaching the real bar exactly is the
localization result: everything after the draw adds no error the judge
can see.

Cost: about a minute. Requires 01-05.

    python experiments/07_verdict.py
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config, judge                     # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import T_GEN, draw_backbone, render        # noqa: E402
from chill_zen.levels import (BackboneLevel, JointLevel,           # noqa: E402
                         backbone_cfg, load_patch_level)

N_PER = 100


def load_stack(backbone, patch):
    """Every frozen component of the deployed system, each checked
    against the configuration it was taught under."""
    pl = load_patch_level(backbone, patch)
    jl = JointLevel(pl)
    gcodes = torch.load(artifacts.path("backbone_codes"))[backbone]
    bl = BackboneLevel(backbone, gcodes["tr"], gcodes["ev"], pl.y_tr, pl.y_ev)
    bstore = torch.load(artifacts.path("backbone_banks"))
    assert bstore[backbone]["cfg"] == backbone_cfg(backbone, bl.M, bl.NCH)
    g1 = bl.make_bank(SEED + 100)
    g1.load_state_dict(bstore[backbone]["g"])
    r1 = bl.make_bank(SEED + 200)
    r1.load_state_dict(bstore[backbone]["r"])
    pstore = torch.load(artifacts.path("patch_banks"))
    assert pstore["cfg"] == pl.cfg()
    g2 = pl.make_bank(SEED + 100)
    g2.load_state_dict(pstore["g"])
    r2 = pl.make_bank(SEED + 200)
    r2.load_state_dict(pstore["r"])
    jstore = torch.load(artifacts.path("joint_bank"))
    assert jstore["cfg"] == jl.cfg()
    jbank = jl.make_bank(SEED + 600)
    jbank.load_state_dict(jstore["bank"])
    gb = torch.load(artifacts.path("backbone_books"))[backbone]
    kb = torch.load(artifacts.path("patch_books"))[patch]
    return dict(bl=bl, pl=pl, jl=jl, g1=g1, r1=r1, g2=g2, r2=r2,
                jbank=jbank, gb=gb, kb=kb, gcodes=gcodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    ap.add_argument("--n-per", type=int, default=N_PER)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("07_verdict")

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(args.backbone, args.patch)
    bl, pl, jl = S["bl"], S["pl"], S["jl"]
    critic = judge.load_critic(artifacts.path("critic"))

    n_per = args.n_per
    lab = torch.arange(10).repeat_interleave(n_per)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:n_per]
                          for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    g_real = S["gcodes"]["ev"].long()[real_idx]

    # The banks are part of the key. Without them a retaught stack leaves
    # these states looking valid, and the verdict below is drawn from
    # banks that no longer exist.
    cfg = dict(seed=SEED + 900, backbone=args.backbone, patch=args.patch,
               t_gen=T_GEN, n_per=n_per,
               banks=(backbone_cfg(args.backbone, bl.M, bl.NCH),
                      pl.cfg(), jl.cfg()))
    vpath = artifacts.path("verdict")
    if vpath.exists() and torch.load(vpath)["cfg"] == cfg:
        st = torch.load(vpath)["st"]
        log("[verdict] states cached")
    else:
        gen = torch.Generator().manual_seed(SEED + 900)
        t0 = time.time()
        g_open = draw_backbone(bl, S["g1"], S["r1"], lab, gen)
        log(f"[verdict] {len(lab)} backbone codes drawn "
            f"({time.time() - t0:.0f}s)")
        none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
        p_d0 = pl.read_all(S["g2"], lab, g_open, none, T_GEN, gen)
        p_r0 = pl.read_all(S["g2"], lab, g_real, none, T_GEN, gen)
        pj = jl.read_joint(S["jbank"], lab, g_open, p_d0, 0.0, None)
        pjr = jl.read_joint(S["jbank"], lab, g_real, p_r0, 0.0, None)
        st = dict(g_e2e=pj[:, :jl.NG].to(torch.int8),
                  p_e2e=pj[:, jl.NG:].to(torch.int8),
                  g_rec=pjr[:, :jl.NG].to(torch.int8),
                  p_rec=pjr[:, jl.NG:].to(torch.int8))
        torch.save(dict(cfg=cfg, st=st), vpath)

    e2e, _ = render(st["g_e2e"], st["p_e2e"], S["gb"], S["kb"])
    rec, _ = render(st["g_rec"], st["p_rec"], S["gb"], S["kb"])
    real_std = judge.fg_std(real)

    log(f"\nn = {len(lab)} on a fresh seed:")
    log(f"{'arm':<16}{'critic':>8}{'div':>8}{'tone':>7}{'seam':>7}{'stdr':>7}")
    ra, rdv = judge.judge(critic, real, lab)
    log(f"{'real':<16}{ra:>8.4f}{rdv:>8.4f}"
        f"{judge.tone_spread(real, lab):>7.3f}"
        f"{judge.seam_ratio(real):>7.3f}{1.0:>7.2f}")
    rows = {}
    for name, img in (("reconstruction", rec), ("end-to-end", e2e)):
        rows[name] = judge.verdict_row(critic, img, lab, real_std)
        log(f"{name:<16}{rows[name][0]:>8.4f}{rows[name][1]:>8.4f}"
            f"{rows[name][2]:>7.3f}{rows[name][3]:>7.3f}"
            f"{rows[name][4]:>7.2f}")
    log(f"\nthe gap between the two arms is the draw: "
        f"{rows['end-to-end'][0] - rows['reconstruction'][0]:+.4f} critic")
    torch.save(dict(cfg=cfg, st=st, real=(ra, rdv), rows=rows), vpath)
    fh.close()


if __name__ == "__main__":
    main()
