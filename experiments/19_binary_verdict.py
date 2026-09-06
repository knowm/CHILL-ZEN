"""Screen the binary arms over temperature, and render for scoring.

The binary-trained arms of Sec. V C and Table VII, in three stages
(`--stage screen|window|render|all`):

* **screen** -- generate at n = 1000 with the backbone draw
  temperature swept over the pre-declared grid {0.1, 0.2, 0.3, 0.5}
  (the patch read stays at the deployed 0.1). An arm carries forward
  to scoring only if its within-class diversity is at least 0.75x the
  real bar's; the rule was declared before any result was seen. Tone
  spread and the fg-std ratio are degenerate on {0,1} renders (every
  foreground pixel is exactly 1), so the binary screen reports critic,
  div, and seam only.
* **window** -- the window draw: G banks only, random book order, each
  read at the highest temperature the read's own physics supplies
  (T_max from the sense floor and settling time), no free-choice
  temperature argument. This is the only draw policy the modeled
  hardware can run without an external temperature knob. It is
  screened by the same diversity rule and reported either way.
* **render** -- for each carried temperature, the two-level stack and
  the one-level 128 and 64 arms at n = 5120 (512 per class), plus the
  non-generator bounds: the binary codec ceiling (real train codes
  through the binary 128 backbone codec), the binary stack ceiling
  (real codes through both levels), and the real 0.1-binarized train
  control. All renders round at the 0.5 midpoint and are saved as
  {0,1} float32 arrays to `head_to_head/gen/binary-*.npy`, where
  `head_to_head/06_binary_arm.py` scores them.

Expected at the screen (real bar critic 0.8680, div 0.1811):
one-level T = 0.1 fails the diversity screen (0.1296 -- the collapse
signature); diversity and critic both rise with T; every temperature
is carried for the two-level arm. The window draw screens out (div
0.1025, critic 0.7890) with T_max medians 0.014-0.043, far below the
~0.3 the binary draw needs.

Cost: minutes for screen and window; about an hour for render.
Requires 17 and 18.

    python experiments/19_binary_verdict.py
"""
import argparse
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, judge                             # noqa: E402
from chill_zen.codec import (decode_global, decode_slots,          # noqa: E402
                             from_patches)
from chill_zen.data import N_TRAIN, SEED, load_fashion             # noqa: E402
from chill_zen.energy import (BANDS, KBT, V_FLOOR,                 # noqa: E402
                              g_sum_from_bytes, t_read)
from chill_zen.generate import draw_backbone                       # noqa: E402
from chill_zen.lanes import LREP                                   # noqa: E402
from chill_zen.levels import (BackboneLevel, JointLevel,           # noqa: E402
                              PatchLevel)
from ktram_neural_core.torch import _lane                          # noqa: E402

SRC_THRESH, RENDER_THRESH = 0.1, 0.5
T_GRID = [0.1, 0.2, 0.3, 0.5]
T_PATCH = 0.1                     # deployed patch-read temperature
DIV_FRAC = 0.75                   # the pre-declared screen rule
N_PER_SCREEN, N_PER_SCORE = 100, 512
STACK_BACKBONE = "128x16@p0.5"
PATCH_TAG = "2x16@p0.5"
CTX_SIZES = [0, 16, 64, 127]      # window-table context sizes
N_REACH = 256
GEN = pathlib.Path(__file__).resolve().parent / "head_to_head" / "gen"


def load_binary_stack(y_tr, y_ev):
    codes = torch.load(artifacts.path("binary_backbone_codes"))
    cs = codes[STACK_BACKBONE]
    bl = BackboneLevel(STACK_BACKBONE, cs["tr"].long(), cs["ev"].long(),
                       y_tr, y_ev)
    bstore = torch.load(artifacts.path("binary_backbone_banks"))
    g1 = bl.make_bank(SEED + 100)
    g1.load_state_dict(bstore[STACK_BACKBONE]["g"])
    r1 = bl.make_bank(SEED + 200)
    r1.load_state_dict(bstore[STACK_BACKBONE]["r"])
    kd = torch.load(artifacts.path("binary_patch_codes"))[PATCH_TAG]
    pl = PatchLevel(STACK_BACKBONE, PATCH_TAG, y_tr, y_ev,
                    cs["tr"].long(), cs["ev"].long(),
                    kd["tr"].long().reshape(len(y_tr), -1),
                    kd["ev"].long().reshape(len(y_ev), -1))
    pstore = torch.load(artifacts.path("binary_patch_banks"))
    g2 = pl.make_bank(SEED + 100)
    g2.load_state_dict(pstore["g"])
    jl = JointLevel(pl)
    jstore = torch.load(artifacts.path("binary_joint_bank"))
    jbank = jl.make_bank(SEED + 600)
    jbank.load_state_dict(jstore["bank"])
    gb = torch.load(artifacts.path("binary_backbone_books"))[STACK_BACKBONE]
    kb = torch.load(artifacts.path("binary_patch_books"))[PATCH_TAG]
    return dict(bl=bl, pl=pl, jl=jl, g1=g1, r1=r1, g2=g2, jbank=jbank,
                gb=gb, kb=kb)


def bin_render_backbone(g_codes, gb):
    dec = decode_global(g_codes.long(), gb["bias"], gb["atoms"],
                        gb["cfg"]["p"])
    return (dec > RENDER_THRESH).float().reshape(-1, 28, 28)


def bin_render_two(g_codes, p_codes, gb, kb):
    dec = decode_global(g_codes.long(), gb["bias"], gb["atoms"],
                        gb["cfg"]["p"])
    slots = p_codes.long().reshape(len(p_codes), 49, -1)
    rend = dec + from_patches(decode_slots(slots, kb["bias"], kb["W"],
                                           kb["cfg"]["p"]))
    return (rend > RENDER_THRESH).float().reshape(-1, 28, 28)


def real_bar(critic, X, y_ev, n_per):
    lab = torch.arange(10).repeat_interleave(n_per)
    real_idx = torch.cat([torch.where(y_ev == c)[0][:n_per]
                          for c in range(10)])
    real = (X[N_TRAIN:, 0][real_idx].clamp(0, 1) > SRC_THRESH).float()
    ra, rdv = judge.judge(critic, real, lab)
    return lab, real, ra, rdv


def stage_screen(S, critic, X, y_ev, store, log):
    lab, real, ra, rdv = real_bar(critic, X, y_ev, N_PER_SCREEN)
    log(f"\nreal bar (0.1-binarized eval, n = {len(lab)}):")
    log(f"{'arm':<22}{'critic':>8}{'div':>8}{'seam':>7}")
    log(f"{'real':<22}{ra:>8.4f}{rdv:>8.4f}"
        f"{judge.seam_ratio(real):>7.3f}")
    results = {"real": (ra, rdv)}
    carried = []
    pl, jl = S["pl"], S["jl"]
    for i, T in enumerate(T_GRID):
        gen = torch.Generator().manual_seed(SEED + 900 + 7 * i)
        t0 = time.time()
        g_open = draw_backbone(S["bl"], S["g1"], S["r1"], lab, gen, t_gen=T)
        none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
        p_d = pl.read_all(S["g2"], lab, g_open, none, T_PATCH, gen)
        pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
        one = bin_render_backbone(g_open, S["gb"])
        two = bin_render_two(pj[:, :jl.NG], pj[:, jl.NG:], S["gb"], S["kb"])
        for name, img in ((f"one-level T={T}", one),
                          (f"two-level T={T}", two)):
            a, dv = judge.judge(critic, img, lab)
            row = (a, dv, judge.seam_ratio(img))
            results[name] = row
            passed = dv >= DIV_FRAC * rdv
            if passed and name.startswith("two-level"):
                carried.append(T)
            log(f"{name:<22}{a:>8.4f}{dv:>8.4f}{row[2]:>7.3f}"
                f"   div screen {'PASS' if passed else 'FAIL'} "
                f"({time.time() - t0:.0f}s)")
    store["screen"] = dict(results=results, t_grid=T_GRID, carried=carried,
                           div_rule=f">= {DIV_FRAC} x real")
    if not carried:
        log("\nevery temperature fell below the diversity screen; nothing "
            "to render.")
    else:
        log(f"\ncarried to render: T in {carried} (rule: two-level div >= "
            f"{DIV_FRAC} x real)")


@torch.no_grad()
def t_max_per_book(bl, g1, codes, lab, ctx_size, gen, log):
    M, Sv = bl.M, bl.S
    out = {b: torch.zeros(M) for b in BANDS}
    y_med, m_med, s_unit = torch.zeros(M), torch.zeros(M), torch.zeros(M)
    keep_p = ctx_size / (M - 1)
    for k in range(M):
        units = torch.full((N_REACH, bl.NU), -1, dtype=torch.long)
        units[:, 0] = lab
        if ctx_size > 0:
            mask = torch.rand(N_REACH, M, generator=gen) < keep_p
            mask[:, k] = False
            units[:, 1:] = torch.where(mask, codes, units[:, 1:])
        yv, mm = g1._y_m(bl.build_aat(units))
        yk = yv.reshape(N_REACH, bl.NU, Sv)[:, 1 + k]
        mk = mm.reshape(N_REACH, bl.NU, Sv)[:, 1 + k]
        y_med[k] = yk.abs().median()
        m_med[k] = mk.median()
        n_sel = LREP + (units[:, 1:] >= 0).sum(1).float().median().item()
        s_unit[k] = g1.noise.sigma_unit(
            torch.tensor(float(y_med[k])), torch.tensor(float(m_med[k])))
        for band in BANDS:
            gs = g_sum_from_bytes(n_sel, float(m_med[k]), band)
            ts = t_read(gs)
            sigma_max = math.sqrt(2 * KBT / (ts * gs * V_FLOOR ** 2))
            out[band][k] = sigma_max / float(s_unit[k])
    log(f"  window ctx {ctx_size:>3}: |y| med {y_med.median():.3f}; m med "
        f"{m_med.median():.0f}; sigma_unit med {s_unit.median():.4f}; "
        f"T_max med " + ", ".join(f"{b} {out[b].median():.3f}"
                                  for b in BANDS))
    return out


def interp_ctx(table, t):
    xs = sorted(table)
    if t <= xs[0]:
        return table[xs[0]]
    if t >= xs[-1]:
        return table[xs[-1]]
    for a, b in zip(xs[:-1], xs[1:]):
        if a <= t <= b:
            w = (t - a) / (b - a)
            return (1 - w) * table[a] + w * table[b]


@torch.no_grad()
def draw_window(bl, g1, lab, gen, tmat):
    """Random order, label clamped, per-(step, book) T from tmat. G-only."""
    B = len(lab)
    units = torch.full((B, bl.NU), -1, dtype=torch.long)
    units[:, 0] = lab
    order = torch.stack([torch.randperm(bl.M, generator=gen)
                         for _ in range(B)])
    rows = torch.arange(B)
    for t in range(bl.M):
        yv, mm = g1._y_m(bl.build_aat(units))
        yv = yv.reshape(B, bl.NU, bl.S)
        mm = mm.reshape(B, bl.NU, bl.S)
        book = order[:, t]
        yb = yv[rows, 1 + book]
        mb = mm[rows, 1 + book]
        for k in torch.unique(book).tolist():
            idx = torch.where(book == k)[0]
            T = float(tmat[k, t])
            if T > 0:
                # device-only by design; see chill_zen.levels._sample
                yb[idx] = _lane.sample_read(yb[idx], mb[idx], T,
                                            g1.noise, gen, comparator=False)
        units[rows, 1 + book] = yb.argmax(-1)
    return units[:, 1:]


def stage_window(S, critic, X, y_ev, store, log):
    bl, g1 = S["bl"], S["g1"]
    M = bl.M
    gen = torch.Generator().manual_seed(SEED + 940)
    idx = torch.randperm(len(bl.u_tr), generator=gen)[:N_REACH]
    codes, lab_r = bl.u_tr[idx, 1:], bl.u_tr[idx, 0]
    ctx = {n: t_max_per_book(bl, g1, codes, lab_r, n, gen, log)
           for n in CTX_SIZES}
    tmat = torch.zeros(M, M)
    for t in range(M):
        per_band = [interp_ctx({n: ctx[n][b] for n in CTX_SIZES}, t)
                    for b in BANDS]
        tmat[:, t] = torch.stack(per_band).min(0).values
    log(f"[window] tmax_step median at steps 0/64/127: "
        f"{tmat[:, 0].median():.3f} / {tmat[:, 64].median():.3f} / "
        f"{tmat[:, 127].median():.3f}")

    lab, real, ra, rdv = real_bar(critic, X, y_ev, N_PER_SCREEN)
    gen = torch.Generator().manual_seed(SEED + 941)
    t0 = time.time()
    g_codes = draw_window(bl, g1, lab, gen, tmat)
    img = bin_render_backbone(g_codes, S["gb"])
    a, dv = judge.judge(critic, img, lab)
    seam = judge.seam_ratio(img)
    passed = dv >= DIV_FRAC * rdv
    log(f"\n{'arm':<22}{'critic':>8}{'div':>8}{'seam':>7}")
    log(f"{'real':<22}{ra:>8.4f}{rdv:>8.4f}")
    log(f"{'window (G-only)':<22}{a:>8.4f}{dv:>8.4f}{seam:>7.3f}   "
        f"div screen {'PASS' if passed else 'FAIL'} "
        f"({time.time() - t0:.0f}s)")
    store["window"] = dict(tmat=tmat, ctx_sizes=CTX_SIZES,
                           row=(a, dv, seam), passed=passed)
    if passed:
        lab5 = torch.arange(10).repeat_interleave(N_PER_SCORE)
        gen = torch.Generator().manual_seed(SEED + 942)
        g5 = draw_window(bl, g1, lab5, gen, tmat)
        save_npy("binary-window", bin_render_backbone(g5, S["gb"]), lab5, log)
    else:
        log("window arm screened out; reported, not scored (it is a row, "
            "not a gate)")


def save_npy(name, imgs, lab, log):
    GEN.mkdir(exist_ok=True)
    a = imgs.numpy().astype(np.float32)
    np.save(GEN / f"{name}.npy", a)
    np.save(GEN / f"{name}-lab.npy", lab.numpy().astype(np.int64))
    log(f"[render] {name}: {a.shape}  on-fraction {a.mean():.4f}")


def stage_render(S, X, y_tr, y_ev, carried, log):
    pl, jl = S["pl"], S["jl"]
    lab = torch.arange(10).repeat_interleave(N_PER_SCORE)
    for i, T in enumerate(carried):
        gen = torch.Generator().manual_seed(SEED + 21000 + 31 * i)
        t0 = time.time()
        g_open = draw_backbone(S["bl"], S["g1"], S["r1"], lab, gen, t_gen=T)
        none = torch.full((len(lab), pl.NPU), -1, dtype=torch.long)
        p_d = pl.read_all(S["g2"], lab, g_open, none, T_PATCH, gen)
        pj = jl.read_joint(S["jbank"], lab, g_open, p_d, 0.0, None)
        log(f"[draw] T={T}: {len(lab)} states ({time.time() - t0:.0f}s)")
        save_npy(f"binary-two-level-T{T}",
                 bin_render_two(pj[:, :jl.NG], pj[:, jl.NG:],
                                S["gb"], S["kb"]), lab, log)
        save_npy(f"binary-one-level-128-T{T}",
                 bin_render_backbone(g_open, S["gb"]), lab, log)

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
    for i, T in enumerate(carried):
        gen = torch.Generator().manual_seed(SEED + 21000 + 500 + 31 * i)
        g64 = draw_backbone(bl64, g1_64, r1_64, lab, gen, t_gen=T)
        save_npy(f"binary-one-level-64-T{T}",
                 bin_render_backbone(g64, gb64), lab, log)

    # the bounds: real train codes, 512 per class
    tr_idx = torch.cat([torch.where(y_tr == c)[0][:N_PER_SCORE]
                        for c in range(10)])
    lab_tr = y_tr[tr_idx]
    g_real = S["pl"].g_tr[tr_idx]
    p_real = S["pl"].p_tr[tr_idx]
    save_npy("binary-codec-ceiling-128",
             bin_render_backbone(g_real, S["gb"]), lab_tr, log)
    save_npy("binary-stack-ceiling",
             bin_render_two(g_real, p_real, S["gb"], S["kb"]), lab_tr, log)
    real = (X[:N_TRAIN, 0][tr_idx].clamp(0, 1) > SRC_THRESH).float()
    save_npy("binary-real-train", real, lab_tr, log)


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["screen", "window", "render", "all"])
    args = ap.parse_args()
    log, fh = artifacts.make_log("19_binary_verdict")
    log(f"binary verdict, stage {args.stage} "
        f"({time.strftime('%Y-%m-%d %H:%M')})")

    X, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    S = load_binary_stack(y_tr, y_ev)
    critic = judge.load_critic(artifacts.path("critic"))

    rpath = artifacts.path("binary_verdict")
    store = torch.load(rpath) if rpath.exists() else {}
    if args.stage in ("screen", "all"):
        stage_screen(S, critic, X, y_ev, store, log)
        torch.save(store, rpath)
    if args.stage in ("window", "all"):
        stage_window(S, critic, X, y_ev, store, log)
        torch.save(store, rpath)
    if args.stage in ("render", "all"):
        carried = store.get("screen", {}).get("carried")
        if not carried:
            log("no carried temperatures on record -- run --stage screen "
                "first")
            fh.close()
            sys.exit(1)
        stage_render(S, X, y_tr, y_ev, carried, log)
    log("binary verdict done")
    fh.close()


if __name__ == "__main__":
    main()
