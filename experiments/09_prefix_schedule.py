"""The one targeted ablation of the draw: teach it the schedule it runs.

Paper Sec. IV D. The backbone fill bank is taught on iid Bernoulli
masks, but at generation time it only ever sees *prefix* contexts:
books below the current step present, everything above open. That is a
plausible reason for the draw's deficit, so it was tested directly.

This script does two things. First it reteaches the fill bank with one
change and one change only -- half the teaching samples keep the
Bernoulli mask, the other half get a prefix mask with the cut drawn
uniformly over the chain -- and selects on next-book accuracy, the
quantity the draw actually consumes. Then it reruns the n = 1000
verdict harness with that bank in place of the deployed one, every
other component frozen.

Expected: next-book accuracy 0.3720 against a class-majority prior of
0.2550 and a majority prior of 0.1290, with the Bernoulli fill probe at
0.2954 against the deployed bank's 0.3065 -- the reteach did what it
was asked to. And the end-to-end critic moves from 0.8340 to 0.8300:
minus 0.004, at the noise floor.

That is the result. The draw's deficit is distributional, not a
schedule mismatch, which is why the paper's next experiment is a
temperature schedule over the chain and not a better mask.

Cost: about 12 minutes to reteach plus a minute to rejudge.
Requires 01-05.

    python experiments/09_prefix_schedule.py
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config, judge, teach              # noqa: E402
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.generate import T_GEN, draw_backbone, render        # noqa: E402
from chill_zen.levels import backbone_cfg                          # noqa: E402
from chill_zen.lanes import snapshot                               # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from importlib import import_module                           # noqa: E402

load_stack = import_module("07_verdict").load_stack

P_PREFIX = 0.5      # half the teaching samples get a prefix mask
N_PER = 100
NBANDS = 8


def prefix_cfg(level):
    return dict(**backbone_cfg(level.tag, level.M, level.NCH),
                mask=f"bern{P_PREFIX}|prefix{P_PREFIX} m~U(0,{level.M})",
                sel="next-book")


def mixed_mask(level, n, gen):
    """Per sample: Bernoulli with probability 1-P_PREFIX, else a prefix."""
    _, mask = level.bernoulli_mask(n, gen)
    use_pref = torch.rand(n, generator=gen) < P_PREFIX
    m = torch.randint(0, level.M + 1, (n,), generator=gen)
    pref = torch.arange(level.NU)[None, :] < (1 + m[:, None])
    pref[:, 0] = True                                   # label held
    return torch.where(use_pref[:, None], pref, mask)


def reteach(level, y_ev, log, batch=teach.BATCH,
            max_epochs=teach.MAX_EPOCHS, patience=teach.PATIENCE):
    gen_ev = torch.Generator().manual_seed(SEED + 9)
    keep_ev, mask_ev = level.bernoulli_mask(len(level.u_ev), gen_ev)
    state_ev = level.masked(level.u_ev, mask_ev)
    holes = ~mask_ev

    # the next-book probe: a random prefix, score the book that follows
    m_ev = torch.randint(0, level.M, (len(level.u_ev),), generator=gen_ev)
    nb_mask = torch.arange(level.NU)[None, :] < (1 + m_ev[:, None])
    nb_mask[:, 0] = True
    nb_state = level.masked(level.u_ev, nb_mask)
    rows = torch.arange(len(level.u_ev))
    tgt_col = 1 + m_ev
    nb_truth = level.u_ev[rows, tgt_col]
    prior_maj = (level.maj[tgt_col] == nb_truth).float().mean().item()
    cls_maj = level.tab[y_ev.long()].argmax(-1)
    prior_cls = (cls_maj[rows, tgt_col] == nb_truth).float().mean().item()
    band_of = m_ev * NBANDS // level.M

    def probe(bank):
        pred = level.read_all(bank, state_ev, 0.0, None)
        fill = ((pred == level.u_ev) & holes).sum().item() / holes.sum().item()
        pnb = level.read_all(bank, nb_state, 0.0, None)
        hit = pnb[rows, tgt_col] == nb_truth
        bands = [hit[band_of == b].float().mean().item()
                 for b in range(NBANDS)]
        return hit.float().mean().item(), fill, bands

    log(f"[prefix] {level.LANES} lanes x {level.KSP} spaces; next-book "
        f"priors: majority {prior_maj:.4f}, class-majority {prior_cls:.4f}")
    gbank = level.make_bank(SEED + 100)
    gen = torch.Generator().manual_seed(SEED + 300)
    best, best_nb, best_ep, curve = None, -1.0, -1, []
    t0 = time.time()
    n_train = len(level.u_tr)
    for ep in range(max_epochs):
        mask = mixed_mask(level, n_train, gen)
        state_tr = level.masked(level.u_tr, mask)
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            gbank.adapt(level.build_aat(state_tr[b]), level.tgt_tr[b],
                        feedback="soft")
        nb, fill, bands = probe(gbank)
        curve.append((nb, fill, bands))
        star = ""
        if nb > best_nb:
            best, best_nb, best_ep, star = snapshot(gbank), nb, ep, " *"
        log(f"  [prefix] ep {ep + 1:2d}  next-book {nb:.4f}  fill "
            f"{fill:.4f}  ({time.time() - t0:.0f}s){star}")
        if ep - best_ep >= patience:
            break
    gbank.load_state_dict(best)
    nb, fill, bands = probe(gbank)
    log(f"[prefix] FROZEN next-book {nb:.4f} (majority prior "
        f"{prior_maj:.4f}, class-majority {prior_cls:.4f})  Bernoulli fill "
        f"{fill:.4f}")
    return snapshot(gbank), dict(nb=nb, fill=fill, bands=bands, curve=curve,
                                 prior_maj=prior_maj, prior_cls=prior_cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--patch", default=config.PATCH)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log, fh = artifacts.make_log("09_prefix_schedule")

    X, y = load_fashion(seed=0)
    y_ev = y[N_TRAIN:].long()
    S = load_stack(args.backbone, args.patch)
    bl, pl, jl = S["bl"], S["pl"], S["jl"]
    cfg = prefix_cfg(bl)

    bpath = artifacts.path("prefix_bank")
    if bpath.exists():
        store = torch.load(bpath)
        if store["cfg"] != cfg:
            raise RuntimeError(f"{bpath.name} was taught under a different "
                               "configuration — delete it to reteach")
        log("[prefix] bank cached")
    else:
        sd, res = reteach(bl, y_ev, log)
        store = dict(cfg=cfg, g=sd, res=res)
        torch.save(store, bpath)

    g1 = bl.make_bank(SEED + 100)
    g1.load_state_dict(store["g"])          # the retaught draw bank
    critic = judge.load_critic(artifacts.path("critic"))

    lab = torch.arange(10).repeat_interleave(N_PER)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:N_PER]
                          for c in range(10)])
    real = X[N_TRAIN:, 0][real_idx].clamp(0, 1)
    g_real = S["gcodes"]["ev"].long()[real_idx]

    vcfg = dict(seed=SEED + 900, backbone=args.backbone, patch=args.patch,
                t_gen=T_GEN, n_per=N_PER, draw_bank="prefix-schedule")
    vpath = artifacts.path("prefix_verdict")
    if vpath.exists() and torch.load(vpath)["cfg"] == vcfg:
        st = torch.load(vpath)["st"]
        log("[prefix] verdict states cached")
    else:
        gen = torch.Generator().manual_seed(SEED + 900)
        t0 = time.time()
        g_open = draw_backbone(bl, g1, S["r1"], lab, gen)
        log(f"[prefix] {len(lab)} backbone codes drawn "
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
        torch.save(dict(cfg=vcfg, st=st), vpath)

    real_std = judge.fg_std(real)
    rows = {}
    log(f"\nn = {len(lab)}, fresh seed, draw bank = prefix schedule:")
    log(f"{'arm':<16}{'critic':>8}{'div':>8}{'tone':>7}{'seam':>7}{'stdr':>7}")
    ra, rdv = judge.judge(critic, real, lab)
    log(f"{'real':<16}{ra:>8.4f}{rdv:>8.4f}"
        f"{judge.tone_spread(real, lab):>7.3f}"
        f"{judge.seam_ratio(real):>7.3f}{1.0:>7.2f}")
    for name, key in (("reconstruction", "rec"), ("end-to-end", "e2e")):
        img, _ = render(st[f"g_{key}"], st[f"p_{key}"], S["gb"], S["kb"])
        rows[name] = judge.verdict_row(critic, img, lab, real_std)
        log(f"{name:<16}{rows[name][0]:>8.4f}{rows[name][1]:>8.4f}"
            f"{rows[name][2]:>7.3f}{rows[name][3]:>7.3f}"
            f"{rows[name][4]:>7.2f}")
    torch.save(dict(cfg=vcfg, st=st, real=(ra, rdv), rows=rows), vpath)
    log("\ncompare with experiment 07: the deployed draw bank scores "
        "0.8340 end-to-end on this same seed.")
    fh.close()


if __name__ == "__main__":
    main()
