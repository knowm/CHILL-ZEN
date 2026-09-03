"""Teaching the banks by lane instructions (paper Sec. IV A).

No gradients are computed anywhere in this file. Teaching is the
emulator's own adapt instruction: present a tuple, name the target
addresses, let the substrate move its own weights.

Three recipes, one shape:

* **soft G banks** teach on Bernoulli-masked codes -- a random subset
  of target addresses is opened per sample, keep-p itself drawn
  U(0,1), the conditioning always held -- and are selected on how well
  they fill the holes.
* **sharp R banks** teach on *realistically* corrupted states, not on
  uniform noise: a Bernoulli mask, a fraction q ~ U(0, 0.5) of the
  surviving symbols replaced from that class's marginal, and the
  holes filled by the matching G bank read at T = 0.3 -- the same
  kind of wrongness the deployed system actually produces. They are
  selected on repair.
* the **joint bank** teaches on two-level corrupted states, with the
  patch level corrupted *under* the already-corrupted backbone, and
  its targets are all 226 addresses.

Every run keeps the best-probe snapshot rather than the last epoch,
with patience 3 and a cap of 15 epochs.
"""
import time

import torch

from .data import SEED
from .lanes import snapshot

BATCH = 256
MAX_EPOCHS, PATIENCE = 15, 3
BANDS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
T_HOT = 0.3          # the temperature the G fills are sampled at
Q_MAX = 0.5          # corruption rate ceiling in the R pools
Q_GAP = 0.25         # conditioning corruption of the patch gap probe


# ---------------------------------------------------------------------------
# The backbone level: G1 (fill) then R1 (repair).
# ---------------------------------------------------------------------------

def teach_backbone(level, y_tr, y_ev, log,
                   batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """-> (g_state, r_state, results) for one backbone code shape."""
    n_train = len(level.u_tr)
    res = {}
    gen_ev = torch.Generator().manual_seed(SEED + 9)
    keep_ev, mask_ev = level.bernoulli_mask(len(level.u_ev), gen_ev)
    state_ev = level.masked(level.u_ev, mask_ev)
    holes = ~mask_ev
    prior_hit = (level.maj[None, :] == level.u_ev) & holes
    prior = prior_hit.sum().item() / holes.sum().item()

    def probe_g(bank):
        pred = level.read_all(bank, state_ev, 0.0, None)
        hit = (pred == level.u_ev) & holes
        acc = hit.sum().item() / holes.sum().item()
        rows = []
        for lo, hi in BANDS:
            sel = (keep_ev >= lo) & (keep_ev < hi)
            nh = max(holes[sel].sum().item(), 1)
            rows.append((hit[sel].sum().item() / nh,
                         prior_hit[sel].sum().item() / nh))
        text = "  ".join(f"[{lo:.2f}-{hi:.2f}) {a:.3f}/{b:.3f}"
                         for (lo, hi), (a, b) in zip(BANDS, rows))
        return acc, rows, text

    gbank = level.make_bank(SEED + 100)
    gen = torch.Generator().manual_seed(SEED + 300)
    best, best_pr, best_ep, curve = None, -1.0, -1, []
    t0 = time.time()
    for ep in range(max_epochs):
        _, mask = level.bernoulli_mask(n_train, gen)
        state_tr = level.masked(level.u_tr, mask)
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            gbank.adapt(level.build_aat(state_tr[b]), level.tgt_tr[b],
                        feedback="soft")
        acc, rows, text = probe_g(gbank)
        curve.append((acc, rows))
        star = ""
        if acc > best_pr:
            best, best_pr, best_ep, star = snapshot(gbank), acc, ep, " *"
        log(f"  [{level.tag}] G ep {ep + 1:2d}  fill {acc:.4f}  {text}  "
            f"({time.time() - t0:.0f}s){star}")
        if ep - best_ep >= patience:
            break
    gbank.load_state_dict(best)
    acc, rows, _ = probe_g(gbank)
    log(f"[{level.tag}] G FROZEN fill {acc:.4f} (prior {prior:.4f}, lift "
        f"{acc - prior:+.4f})")
    res["g"] = dict(acc=acc, rows=rows, prior=prior, curve=curve)

    def build_pool(units, y, gen):
        _, mask = level.bernoulli_mask(len(units), gen)
        state = level.masked(units, mask)
        q = torch.rand(len(units), generator=gen) * Q_MAX
        hit = (torch.rand(units.shape, generator=gen) < q[:, None]) & mask
        hit[:, 0] = False                                 # label stays clean
        state = torch.where(hit, level.marginal_draw(y, gen), state)
        fills = level.read_all(gbank, state, T_HOT, gen)
        return torch.where(mask, state, fills)

    pool_tr = build_pool(level.u_tr, y_tr,
                         torch.Generator().manual_seed(SEED + 500))
    pool_ev = build_pool(level.u_ev, y_ev,
                         torch.Generator().manual_seed(SEED + 550))
    wrong_ev = pool_ev != level.u_ev
    maj_ev = level.u_ev == level.maj[None, :]
    log(f"  [{level.tag}] pool wrong fraction "
        f"{wrong_ev.float().mean().item():.3f}")

    def probe_r(bank):
        pred = level.read_all(bank, pool_ev, 0.0, None)
        hit = pred == level.u_ev
        rep = hit[wrong_ev].float().mean().item()
        pre = hit[~wrong_ev].float().mean().item()
        pre_mj = hit[~wrong_ev & maj_ev].float().mean().item()
        pre_mn = hit[~wrong_ev & ~maj_ev].float().mean().item()
        return rep, (pre, pre_mj, pre_mn)

    rbank = level.make_bank(SEED + 200)
    gen = torch.Generator().manual_seed(SEED + 400)
    best, best_rep, best_ep, curve = None, -1.0, -1, []
    t0 = time.time()
    for ep in range(max_epochs):
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            rbank.adapt(level.build_aat(pool_tr[b]), level.tgt_tr[b],
                        feedback="hard")
        rep, pres = probe_r(rbank)
        curve.append((rep, pres))
        star = ""
        if rep > best_rep:
            best, best_rep, best_ep, star = snapshot(rbank), rep, ep, " *"
        log(f"  [{level.tag}] R ep {ep + 1:2d}  repair {rep:.4f}  preserve "
            f"{pres[0]:.4f} (maj {pres[1]:.4f} / min {pres[2]:.4f})  "
            f"({time.time() - t0:.0f}s){star}")
        if ep - best_ep >= patience:
            break
    rbank.load_state_dict(best)
    rep, pres = probe_r(rbank)
    log(f"[{level.tag}] R FROZEN repair {rep:.4f}  preserve {pres[0]:.4f} "
        f"(maj {pres[1]:.4f} / min {pres[2]:.4f})")
    res["r"] = dict(rep=rep, pres=pres, curve=curve,
                    wrong=wrong_ev.float().mean().item())
    return snapshot(gbank), snapshot(rbank), res


# ---------------------------------------------------------------------------
# The patch level: G2 (fill under held conditioning) then R2 (repair).
# ---------------------------------------------------------------------------

def _patch_probe_state(pl):
    """The fixed held-out fill probe, and the gap probe beside it.

    Both are built from pinned generators, so the same states are used
    every time a bank is probed -- including when a cached bank is
    re-probed rather than retaught.
    """
    gen_ev = torch.Generator().manual_seed(SEED + 9)
    keep_ev, mask_ev = pl.bernoulli_mask(len(pl.p_ev), gen_ev)
    state_ev = pl.masked(pl.p_ev, mask_ev)
    holes = ~mask_ev
    prior_hit = (pl.maj[None, :] == pl.p_ev) & holes
    prior = prior_hit.sum().item() / holes.sum().item()
    # the gap probe: the same fill with the backbone conditioning itself
    # corrupted, which is what an imperfect draw hands this level
    gen_gap = torch.Generator().manual_seed(SEED + 11)
    hit_g = torch.rand(len(pl.g_ev), pl.NG, generator=gen_gap) < Q_GAP
    g_bad = torch.where(hit_g, pl.marginal_draw(pl.gtab, pl.y_ev, gen_gap),
                        pl.g_ev)
    return keep_ev, state_ev, holes, prior_hit, prior, g_bad


def probe_patch_fill(pl, bank, probe=None):
    """-> (fill accuracy, gap-probe accuracy, per-band rows, prior)."""
    keep_ev, state_ev, holes, prior_hit, prior, g_bad = \
        probe if probe is not None else _patch_probe_state(pl)
    pred = pl.read_all(bank, pl.y_ev, pl.g_ev, state_ev, 0.0, None)
    hit = (pred == pl.p_ev) & holes
    acc = hit.sum().item() / holes.sum().item()
    pred_b = pl.read_all(bank, pl.y_ev, g_bad, state_ev, 0.0, None)
    gap = ((pred_b == pl.p_ev) & holes).sum().item() / holes.sum().item()
    rows = []
    for lo, hi in BANDS:
        sel = (keep_ev >= lo) & (keep_ev < hi)
        nh = max(holes[sel].sum().item(), 1)
        rows.append((hit[sel].sum().item() / nh,
                     prior_hit[sel].sum().item() / nh))
    return acc, gap, rows, prior


def probe_patch_repair(pl, gbank, bank, pool_ev=None):
    """-> (repair rate, (preserve, majority, minority), pool wrong fraction)."""
    if pool_ev is None:
        pool_ev = patch_pool(pl, gbank, pl.y_ev, pl.g_ev, pl.p_ev,
                             torch.Generator().manual_seed(SEED + 550))
    wrong = pool_ev != pl.p_ev
    maj_ev = pl.p_ev == pl.maj[None, :]
    pred = pl.read_all(bank, pl.y_ev, pl.g_ev, pool_ev, 0.0, None)
    hit = pred == pl.p_ev
    return (hit[wrong].float().mean().item(),
            (hit[~wrong].float().mean().item(),
             hit[~wrong & maj_ev].float().mean().item(),
             hit[~wrong & ~maj_ev].float().mean().item()),
            wrong.float().mean().item())


def teach_patch_fill(pl, log, batch=BATCH, max_epochs=MAX_EPOCHS,
                     patience=PATIENCE):
    """-> (g_state, results, curve). The gap probe rides along."""
    probe_state = _patch_probe_state(pl)
    prior = probe_state[4]

    def probe_g(bank):
        acc, gap, rows, _ = probe_patch_fill(pl, bank, probe_state)
        text = "  ".join(f"[{lo:.2f}-{hi:.2f}) {a:.3f}/{b:.3f}"
                         for (lo, hi), (a, b) in zip(BANDS, rows))
        return acc, gap, rows, text

    n_train = len(pl.p_tr)
    log(f"[patch] G2: {pl.LANES} lanes x {pl.KSP} spaces x {pl.NCH} ch "
        f"(soft); prior {prior:.4f}, chance {1 / pl.S:.4f}")
    gbank = pl.make_bank(SEED + 100)
    gen = torch.Generator().manual_seed(SEED + 300)
    best, best_pr, best_ep, curve = None, -1.0, -1, []
    t0 = time.time()
    for ep in range(max_epochs):
        _, mask = pl.bernoulli_mask(n_train, gen)
        state_tr = pl.masked(pl.p_tr, mask)
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            gbank.adapt(pl.build_aat(pl.y_tr[b], pl.g_tr[b], state_tr[b]),
                        pl.tgt_tr[b], feedback="soft")
        acc, gap, rows, text = probe_g(gbank)
        curve.append((acc, gap, rows))
        star = ""
        if acc > best_pr:
            best, best_pr, best_ep, star = snapshot(gbank), acc, ep, "  *best"
        log(f"  G2 ep {ep + 1:2d}  fill {acc:.4f} (gap {gap:.4f})  {text}  "
            f"({time.time() - t0:.0f}s){star}")
        if ep - best_ep >= patience:
            log(f"  patience — freezing epoch {best_ep + 1}")
            break
    gbank.load_state_dict(best)
    acc, gap, rows, text = probe_g(gbank)
    log(f"[patch] G2 FROZEN  fill {acc:.4f} (prior {prior:.4f}, lift "
        f"{acc - prior:+.4f}; gap probe {gap:.4f}, drop {acc - gap:+.4f})")
    return best, dict(acc=acc, gap=gap, rows=rows, prior=prior), curve


def patch_pool(pl, gbank, y, g, patches, gen):
    """One realistically corrupted patch state per sample."""
    _, mask = pl.bernoulli_mask(len(patches), gen)
    state = pl.masked(patches, mask)
    q = torch.rand(len(patches), generator=gen) * Q_MAX
    hit = (torch.rand(patches.shape, generator=gen) < q[:, None]) & mask
    state = torch.where(hit, pl.marginal_draw(pl.ptab, y, gen), state)
    fills = pl.read_all(gbank, y, g, state, T_HOT, gen)
    return torch.where(mask, state, fills)


def teach_patch_repair(pl, gbank, log, batch=BATCH, max_epochs=MAX_EPOCHS,
                       patience=PATIENCE):
    """-> (r_state, results, curve)."""
    pool_tr = patch_pool(pl, gbank, pl.y_tr, pl.g_tr, pl.p_tr,
                         torch.Generator().manual_seed(SEED + 500))
    pool_ev = patch_pool(pl, gbank, pl.y_ev, pl.g_ev, pl.p_ev,
                         torch.Generator().manual_seed(SEED + 550))
    wrong_ev = pool_ev != pl.p_ev
    log(f"[patch] pool wrong fraction {wrong_ev.float().mean().item():.3f}")

    n_train = len(pl.p_tr)
    rbank = pl.make_bank(SEED + 200)
    gen = torch.Generator().manual_seed(SEED + 400)
    best, best_rep, best_ep, curve = None, -1.0, -1, []
    t0 = time.time()
    for ep in range(max_epochs):
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            rbank.adapt(pl.build_aat(pl.y_tr[b], pl.g_tr[b], pool_tr[b]),
                        pl.tgt_tr[b], feedback="hard")
        rep, pres, _ = probe_patch_repair(pl, gbank, rbank, pool_ev)
        curve.append((rep, pres))
        star = ""
        if rep > best_rep:
            best, best_rep, best_ep, star = snapshot(rbank), rep, ep, "  *best"
        log(f"  R2 ep {ep + 1:2d}  repair {rep:.4f}  preserve {pres[0]:.4f} "
            f"(maj {pres[1]:.4f} / min {pres[2]:.4f})  "
            f"({time.time() - t0:.0f}s){star}")
        if ep - best_ep >= patience:
            log(f"  patience — freezing epoch {best_ep + 1}")
            break
    rbank.load_state_dict(best)
    rep, pres, wrong = probe_patch_repair(pl, gbank, rbank, pool_ev)
    log(f"[patch] R2 FROZEN  repair {rep:.4f}  preserve {pres[0]:.4f} "
        f"(maj {pres[1]:.4f} / min {pres[2]:.4f})")
    return best, dict(rep=rep, pres=pres, wrong=wrong), curve


# ---------------------------------------------------------------------------
# The joint bank.
# ---------------------------------------------------------------------------

def joint_pool(jl, backbone_level, g1, g2, y, g_true, p_true, gen, log, tag):
    """Corrupt both levels the way the deployed system fails.

    Backbone: Bernoulli mask, a fraction of the survivors replaced from
    the class marginal, the holes filled by G1 at T = 0.3. Patch: the
    same recipe through G2, but conditioned on the *already corrupted*
    backbone -- which is the state a real draw hands the joint sweep.
    """
    t0 = time.time()
    pl = jl.pl
    B = len(y)
    keep_g = torch.rand(B, generator=gen)
    mask_g = torch.rand(B, jl.NG, generator=gen) < keep_g[:, None]
    q_g = torch.rand(B, generator=gen) * Q_MAX
    hit_g = (torch.rand(B, jl.NG, generator=gen) < q_g[:, None]) & mask_g
    g_state = torch.where(hit_g, pl.marginal_draw(pl.gtab, y, gen), g_true)
    g_state = torch.where(mask_g, g_state,
                          torch.full((1,), -1, dtype=torch.long))
    units = torch.cat([y[:, None], g_state], 1)
    fills = backbone_level.read_all(g1, units, T_HOT, gen)[:, 1:]
    g_pool = torch.where(mask_g, g_state, fills)

    keep_p = torch.rand(B, generator=gen)
    mask_p = torch.rand(B, jl.NPU, generator=gen) < keep_p[:, None]
    q_p = torch.rand(B, generator=gen) * Q_MAX
    hit_p = (torch.rand(B, jl.NPU, generator=gen) < q_p[:, None]) & mask_p
    p_state = torch.where(hit_p, pl.marginal_draw(pl.ptab, y, gen), p_true)
    p_state = torch.where(mask_p, p_state,
                          torch.full((1,), -1, dtype=torch.long))
    fills = pl.read_all(g2, y, g_pool, p_state, T_HOT, gen)
    p_pool = torch.where(mask_p, p_state, fills)
    wg = (g_pool != g_true).float().mean().item()
    wp = (p_pool != p_true).float().mean().item()
    log(f"  [{tag}] pool: {B} states, wrong globals {wg:.3f} / patches "
        f"{wp:.3f} ({time.time() - t0:.0f}s)")
    return g_pool.to(torch.int8), p_pool.to(torch.int8)


def probe_joint(jl, y_ev, truth_ev, pool_ev, maj):
    """-> (repair, (backbone, patch), (preserve, maj, min, backbone), text)."""
    gp_ev, pp_ev = pool_ev
    pool = torch.cat([gp_ev.long(), pp_ev.long()], 1)
    wrong = pool != truth_ev
    maj_ev = truth_ev == maj[None, :]
    is_g = torch.zeros(jl.NU, dtype=torch.bool)
    is_g[:jl.NG] = True

    def run(bank):
        pred = jl.read_joint(bank, y_ev, gp_ev.long(), pp_ev.long(), 0.0,
                             None)
        hit = pred == truth_ev
        rep = hit[wrong].float().mean().item()
        rep_g = hit[wrong & is_g[None, :]].float().mean().item()
        rep_p = hit[wrong & ~is_g[None, :]].float().mean().item()
        pre = hit[~wrong].float().mean().item()
        pre_mj = hit[~wrong & maj_ev].float().mean().item()
        pre_mn = hit[~wrong & ~maj_ev].float().mean().item()
        pre_g = hit[~wrong & is_g[None, :]].float().mean().item()
        return rep, (rep_g, rep_p), (pre, pre_mj, pre_mn, pre_g), \
            (f"repair {rep:.4f} (backbone {rep_g:.4f} / patch {rep_p:.4f})"
             f"  preserve {pre:.4f} (maj {pre_mj:.4f} / min {pre_mn:.4f} / "
             f"backbone {pre_g:.4f})")
    run.wrong = wrong.float().mean().item()
    return run


def teach_joint(jl, y_tr, y_ev, truth_tr, truth_ev, pool_tr, pool_ev, maj,
                log, batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """-> (bank_state, results, curve). Targets are all 226 addresses."""
    pl = jl.pl
    gp_tr, pp_tr = pool_tr
    tgt_tr = jl.targets(truth_tr[:, :jl.NG], truth_tr[:, jl.NG:])
    probe = probe_joint(jl, y_ev, truth_ev, pool_ev, maj)

    n_train = len(truth_tr)
    log(f"[joint] {jl.LANES} lanes x {jl.KSP} spaces x {jl.NCH} ch (sharp); "
        f"wrong in the eval pool {probe.wrong:.3f}")
    bank = jl.make_bank(SEED + 600)
    gen = torch.Generator().manual_seed(SEED + 400)
    best, best_rep, best_ep, curve = None, -1.0, -1, []
    t0 = time.time()
    gtr, ptr = gp_tr.long(), pp_tr.long()
    for ep in range(max_epochs):
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            bank.adapt(pl.build_aat(y_tr[b], gtr[b], ptr[b]), tgt_tr[b],
                       feedback="hard")
        rep, reps, pres, text = probe(bank)
        curve.append((rep, reps, pres))
        star = ""
        if rep > best_rep:
            best, best_rep, best_ep, star = snapshot(bank), rep, ep, "  *best"
        log(f"  ep {ep + 1:2d}  {text}  ({time.time() - t0:.0f}s){star}")
        if ep - best_ep >= patience:
            log(f"  patience — freezing epoch {best_ep + 1}")
            break
    bank.load_state_dict(best)
    rep, reps, pres, text = probe(bank)
    log(f"[joint] FROZEN  {text}")
    return best, dict(rep=rep, reps=reps, pres=pres, wrong=probe.wrong), curve
