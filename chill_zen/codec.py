"""The dropout additive-quantization codec (paper Sec. III A).

A code is M books of S atoms; the decode is the bias plus the sum of
the M chosen atoms, scaled by the keep probability p:

    x_hat = b + p * sum_m a_{m, z_m}

and the fit minimizes the expected error when each atom is present
only with probability p,

    E_z ||x - b - sum_j z_j a_j||^2
        = ||x - b - p * sum_j a_j||^2 + p(1-p) * sum_j ||a_j||^2 ,

which is what makes a partial code decode to something sensible and
what puts the cross-address redundancy the banks feed on into the
code. p = 1 is plain additive quantization.

Two geometries, one algorithm:

* the **backbone** codec fits M books over the whole 784-pixel image
  (`fit_global`), and
* the **patch** codec fits M books per slot over a 7x7 grid of
  4x4-pixel slots (`fit_slots`), on the residual left by the backbone
  decode.

Fitting alternates greedy+ICM encoding with a joint fp64 ridge solve
of the dropout normal equations, followed by a per-book gauge
re-centering that moves each book's usage-weighted mean into the bias
(so the books carry only deviations). Initialization is an
independent per-book Lloyd pass on the residual, one generator per
book at seed SEED + 101*m.
"""
import math

import torch
import torch.nn.functional as F

from .data import SEED

ITERS = 5          # alternations of (encode, refit)
PASSES = 3         # ICM sweeps per encode, after the greedy pass
KM_ITERS = 15      # Lloyd iterations in the initializer
CHUNK = 8192
RIDGE = 1e-6       # scaled by N inside the refit


def bits_of(M, S):
    return int(round(M * math.log2(S)))


# ---------------------------------------------------------------------------
# Backbone: M books over the whole image (one position).
# ---------------------------------------------------------------------------

def _kmeans_global(x, S, gen, iters=KM_ITERS):
    D = x.shape[1]
    C = x[torch.randint(0, len(x), (S,), generator=gen)].clone()
    for _ in range(iters):
        a = torch.empty(len(x), dtype=torch.int64)
        for lo in range(0, len(x), CHUNK):
            sc = 2 * x[lo:lo + CHUNK] @ C.T - (C ** 2).sum(-1)[None]
            a[lo:lo + CHUNK] = sc.argmax(-1)
        cnt = torch.bincount(a, minlength=S).float()
        s = torch.zeros(S, D).index_add_(0, a, x)
        C = torch.where((cnt > 0)[:, None], s / cnt.clamp(min=1)[:, None], C)
    return C


@torch.no_grad()
def encode_global(x, bias, atoms, p, passes=PASSES):
    """[N, D] -> codes [N, M]. Greedy pass then `passes` ICM sweeps."""
    M = atoms.shape[0]
    codes = torch.zeros(len(x), M, dtype=torch.int64)
    for lo in range(0, len(x), CHUNK):
        r = x[lo:lo + CHUNK] - bias
        for pas in range(passes + 1):
            for m in range(M):
                if pas > 0:
                    r = r + p * atoms[m][codes[lo:lo + CHUNK, m]]
                sc = 2 * r @ atoms[m].T - (atoms[m] ** 2).sum(-1)[None]
                pick = sc.argmax(-1)
                codes[lo:lo + CHUNK, m] = pick
                r = r - p * atoms[m][pick]
    return codes


def refit_global(x, codes, p, S):
    """Joint fp64 solve of the dropout normal equations, plus gauge."""
    N, M = len(x), codes.shape[1]
    D = x.shape[1]
    Dm = M * S + 1
    Fk = F.one_hot(codes, S).reshape(N, M * S).float()
    FtF = (Fk.T @ Fk).double()
    cnt = FtF.diagonal().clone()
    AtA = torch.zeros(Dm, Dm, dtype=torch.float64)
    AtA[:-1, :-1] = p * p * FtF
    AtA[:-1, :-1].diagonal().copy_(p * cnt)          # p^2 c + p(1-p) c
    AtA[:-1, -1] = p * cnt
    AtA[-1, :-1] = p * cnt
    AtA[-1, -1] = N
    AtA += (RIDGE * N) * torch.eye(Dm, dtype=torch.float64)
    Atb = torch.zeros(Dm, D, dtype=torch.float64)
    Atb[:-1] = (p * (Fk.T @ x)).double()
    Atb[-1] = x.sum(0).double()
    sol = torch.linalg.solve(AtA, Atb).float()
    atoms, bias = sol[:-1].reshape(M, S, D), sol[-1]
    usage = F.one_hot(codes, S).float().mean(0)
    for m in range(M):
        mu = (usage[m][:, None] * atoms[m]).sum(0)
        atoms[m] -= mu
        bias += p * mu
    return bias, atoms


def decode_global(codes, bias, atoms, p):
    at = torch.zeros(len(codes), atoms.shape[2])
    for m in range(atoms.shape[0]):
        at += atoms[m][codes[:, m]]
    return bias + p * at


def fit_global(x_tr, M, S, p, log=print, iters=ITERS, seed=SEED):
    """Fit the backbone codec on [N, D] images. -> (bias, atoms)."""
    import time
    from .data import rel_mse
    t0 = time.time()
    bias = x_tr.mean(0)
    r = x_tr - bias
    atoms = torch.stack([
        _kmeans_global(r, S, torch.Generator().manual_seed(seed + 101 * m))
        for m in range(M)])
    for it in range(iters):
        codes = encode_global(x_tr, bias, atoms, p)
        bias, atoms = refit_global(x_tr, codes, p, S)
        sub = decode_global(encode_global(x_tr[:4096], bias, atoms, p),
                            bias, atoms, p)
        log(f"  [{M}x{S}@p{p}] iter {it + 1}  train relMSE(4k) "
            f"{rel_mse(sub, x_tr[:4096]):.4f}  ({time.time() - t0:.0f}s)")
    return bias, atoms


def coherence_global(codes_tr, codes_ev, M, S):
    """Cross-book predictability, and the majority bar it must beat.

    For each ordered pair of books, predict book m's symbol from book
    j's via the argmax of the train co-occurrence table, and score on
    eval. The bar is the per-book train-majority symbol: what a
    predictor that ignores its input would get.
    """
    accs = []
    for j in range(M):
        for m in range(M):
            if j == m:
                continue
            idx = codes_tr[:, j] * S + codes_tr[:, m]
            tab = torch.bincount(idx, minlength=S * S).reshape(S, S)
            pred = tab.argmax(-1)
            accs.append((pred[codes_ev[:, j]] == codes_ev[:, m])
                        .float().mean().item())
    coh = sum(accs) / len(accs)
    hits = []
    for m in range(M):
        maj = torch.bincount(codes_tr[:, m], minlength=S).argmax()
        hits.append((codes_ev[:, m] == maj).float().mean().item())
    return coh, sum(hits) / len(hits)


# ---------------------------------------------------------------------------
# Patch level: M books per slot, over a 7x7 grid of 4x4-pixel slots.
# ---------------------------------------------------------------------------

P_SLOTS, PX, S_PATCH = 49, 16, 16


def to_patches(img, P=P_SLOTS, PX_=PX):
    """[N, 784] -> [N, 49, 16], row-major over a 7x7 grid of 4x4 blocks."""
    return img.reshape(-1, 7, 4, 7, 4).permute(0, 1, 3, 2, 4) \
        .reshape(-1, P, PX_)


def from_patches(pt):
    """[N, 49, 16] -> [N, 784]. The inverse of `to_patches`."""
    return pt.reshape(-1, 7, 7, 4, 4).permute(0, 1, 3, 2, 4) \
        .reshape(-1, 784)


def _kmeans_slots(x, S, gen, iters=KM_ITERS):
    """Vectorized per-slot Lloyd. x [N, P, px] -> C [P, S, px]."""
    N, P, PX_ = x.shape
    idx = torch.randint(0, N, (P, S), generator=gen)
    C = x.transpose(0, 1)[torch.arange(P)[:, None], idx].clone()
    xt = x.reshape(N * P, PX_)
    posg = torch.arange(P)[None, :].expand(N, P)
    for _ in range(iters):
        a = torch.empty(N, P, dtype=torch.int64)
        for lo in range(0, N, CHUNK):
            xc = x[lo:lo + CHUNK]
            sc = 2 * torch.einsum("bpx,psx->bps", xc, C) \
                - (C ** 2).sum(-1)[None]
            a[lo:lo + CHUNK] = sc.argmax(-1)
        g = (posg * S + a).reshape(-1)
        cnt = torch.zeros(P * S).index_add_(0, g, torch.ones(N * P))
        s = torch.zeros(P * S, PX_).index_add_(0, g, xt)
        C = torch.where((cnt > 0)[:, None], s / cnt.clamp(min=1)[:, None],
                        C.reshape(P * S, PX_)).reshape(P, S, PX_)
    return C


@torch.no_grad()
def encode_slots(pt, bias, W, p, passes=PASSES):
    """[N, P, px] -> codes [N, P, M]."""
    N, P = pt.shape[0], pt.shape[1]
    M = W.shape[1]
    codes = torch.zeros(N, P, M, dtype=torch.int64)
    pos = torch.arange(P)[None, :]
    for lo in range(0, N, CHUNK):
        r = pt[lo:lo + CHUNK] - bias
        for pas in range(passes + 1):
            for m in range(M):
                Wm = W[:, m]
                if pas > 0:
                    r = r + p * Wm[pos, codes[lo:lo + CHUNK, :, m]]
                sc = 2 * torch.einsum("bpx,psx->bps", r, Wm) \
                    - (Wm ** 2).sum(-1)[None]
                pick = sc.argmax(-1)
                codes[lo:lo + CHUNK, :, m] = pick
                r = r - p * Wm[pos, pick]
    return codes


def refit_slots(pt, codes, p, S):
    N, P, PX_ = pt.shape
    M = codes.shape[2]
    D = M * S + 1
    W = torch.zeros(P, M, S, PX_)
    bias = torch.zeros(P, PX_)
    eye = (RIDGE * N) * torch.eye(D, dtype=torch.float64)
    for k in range(P):
        Fk = F.one_hot(codes[:, k], S).reshape(N, M * S).float()
        FtF = (Fk.T @ Fk).double()
        cnt = FtF.diagonal().clone()
        AtA = torch.zeros(D, D, dtype=torch.float64)
        AtA[:-1, :-1] = p * p * FtF
        AtA[:-1, :-1].diagonal().copy_(p * cnt)
        AtA[:-1, -1] = p * cnt
        AtA[-1, :-1] = p * cnt
        AtA[-1, -1] = N
        AtA += eye
        Atb = torch.zeros(D, PX_, dtype=torch.float64)
        Atb[:-1] = (p * (Fk.T @ pt[:, k])).double()
        Atb[-1] = pt[:, k].sum(0).double()
        sol = torch.linalg.solve(AtA, Atb).float()
        W[k] = sol[:-1].reshape(M, S, PX_)
        bias[k] = sol[-1]
        usage = F.one_hot(codes[:, k], S).float().mean(0)
        for m in range(M):
            mu = (usage[m][:, None] * W[k, m]).sum(0)
            W[k, m] -= mu
            bias[k] += p * mu
    return bias, W


@torch.no_grad()
def decode_slots(codes, bias, W, p):
    P, M = codes.shape[1], W.shape[1]
    at = W[torch.arange(P)[None, :, None],
           torch.arange(M)[None, None, :], codes]
    return bias + p * at.sum(2)


def fit_slots(res_tr, M, S, p, log=print, iters=ITERS, seed=SEED):
    """Fit the patch codec on residual patches [N, P, px]. -> (bias, W)."""
    import time
    from .data import rel_mse
    t0 = time.time()
    P, PX_ = res_tr.shape[1], res_tr.shape[2]
    bias = res_tr.mean(0)
    W = torch.zeros(P, M, S, PX_)
    for m in range(M):
        gen = torch.Generator().manual_seed(seed + 101 * m)
        W[:, m] = _kmeans_slots(res_tr - bias, S, gen)
    for it in range(iters):
        codes = encode_slots(res_tr, bias, W, p)
        bias, W = refit_slots(res_tr, codes, p, S)
        sub = decode_slots(encode_slots(res_tr[:4096], bias, W, p),
                           bias, W, p)
        log(f"  [{M}x{S}@p{p}] iter {it + 1}  residual relMSE(4k) "
            f"{rel_mse(sub, res_tr[:4096]):.4f}  ({time.time() - t0:.0f}s)")
    return bias, W


def coherence_slots(codes_tr, codes_ev, M, S):
    """Within-slot cross-book predictability, and the majority bar."""
    P = codes_tr.shape[1]
    accs = []
    for j in range(M):
        for m in range(M):
            if j == m:
                continue
            tab = torch.zeros(P, S, S)
            idx = codes_tr[:, :, j] * S + codes_tr[:, :, m]
            for k in range(P):
                tab[k] = torch.bincount(idx[:, k], minlength=S * S) \
                    .reshape(S, S).float()
            pred = tab.argmax(-1)
            hit = (pred[torch.arange(P)[None, :], codes_ev[:, :, j]]
                   == codes_ev[:, :, m]).float().mean().item()
            accs.append(hit)
    coh = sum(accs) / len(accs) if accs else float("nan")
    hits = []
    for m in range(M):
        maj = torch.zeros(P, dtype=torch.int64)
        for k in range(P):
            maj[k] = torch.bincount(codes_tr[:, k, m], minlength=S).argmax()
        hits.append((codes_ev[:, :, m] == maj[None, :]).float()
                    .mean().item())
    return coh, sum(hits) / len(hits)
