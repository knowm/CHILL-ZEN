"""The physical hot read (paper Sec. II A's comparator term, Sec. IV C).

One Gaussian draw per read, clipped to [-1, 1]:

    sigma_y^2 = (0.02 * sigma_unit(y, m; V, t_b))^2 + (v_n / V)^2

`0.02` is the device gain (Core `READ_NOISE`); it is never varied.
`sigma_unit` is the emulator's own law (thermal + flicker in
quadrature), evaluated through `ktram_neural_core.torch._lane.NoiseParams`
at the read voltage `V` and the settling-limited pulse `t_b`. `t_b =
max(2 ns, 5 C_LANE / G_sum)` per read, `G_sum` from the committing
lanes' median pooled magnitude and `n_sel`, on the SnCr band -- the
emulator's thermal constant reads as a SnCr-band Johnson source; the
W pairing is not physical. `v_n / V` is the comparator's
input-referred noise, flat in `y` and `m`, added in quadrature. Cold
reads are unaffected: this module only replaces the hot autoregressive
backbone draw and the parallel patch read.

Written once here and generalized over any `BackboneLevel`/`PatchLevel`
pair, so the same read drives the one-level frontier and the binary
stack as well as the deployed two-level system. No change to
`kt-ram-neural-core`; this module calls its public API (`NoiseParams`,
`sigma_unit`).
"""
import torch

from ktram_neural_core.torch import _lane

from .energy import BANDS, g_sum_from_bytes, t_read
from .lanes import LREP
from .levels import S as SYM

GAIN_DEV = 0.02      # Core READ_NOISE: the device, never varied
BAND = "SnCr"        # the band the emulator's thermal constant reads as


def params_at(base, v_read, pw):
    """The bank's own NoiseParams re-pointed at a read (V, t)."""
    return _lane.NoiseParams(a_thermal_unit=base.a_thermal_unit,
                             a_flicker_unit=base.a_flicker_unit,
                             sqrt_ref_m=base.sqrt_ref_m,
                             flicker_ln_ref=base.flicker_ln_ref,
                             ref_pw=base.ref_pw, read_pw=pw, v_read=v_read)


def physical_read(yv, mm, sel, n_sel, base, v_read, v_n, gen):
    """One hot read at (V, t_b, v_n). `sel` picks the committing lanes
    for the pool load; the same (V, t_b) applies to every lane in the
    read. Returns the noisy y and the variance shares at the committing
    lanes (thermal, flicker, comparator)."""
    m_med = mm[sel].median().item()
    gs = g_sum_from_bytes(n_sel, m_med, BAND)
    t_b = t_read(gs)
    p = params_at(base, v_read, t_b)
    s_lane = GAIN_DEV * p.sigma_unit(yv, mm)
    s_cmp = v_n / v_read
    sigma = torch.sqrt(s_lane * s_lane + s_cmp * s_cmp)
    r = torch.randn(yv.shape, generator=gen, dtype=yv.dtype)
    out = (yv + sigma * r).clamp(-1.0, 1.0)
    # shares, for the composition table
    bw_th = (p.ref_pw / p.read_pw) ** 0.5
    ln_band = p.flicker_ln_ref + torch.log(torch.tensor(p.ref_pw / p.read_pw)).item()
    bw_fl = (ln_band / p.flicker_ln_ref) ** 0.5 if ln_band > 0 else 0.0
    f_m = p.sqrt_ref_m / torch.sqrt(mm[sel].clamp(min=1e-30))
    v_th = (GAIN_DEV * p.a_thermal_unit * f_m / v_read * bw_th) ** 2
    v_fl = (GAIN_DEV * p.a_flicker_unit * (1 - yv[sel] ** 2) * f_m * bw_fl) ** 2
    v_cm = torch.full_like(v_th, s_cmp ** 2)
    tot = v_th + v_fl + v_cm
    shares = ((v_th / tot).median().item(), (v_fl / tot).median().item(),
              (v_cm / tot).median().item(), sigma[sel].median().item(),
              m_med, gs, t_b)
    return out, shares


@torch.no_grad()
def draw_backbone_physical(bl, g1, r1, lab, gen, v_read, v_n, log_ctx=()):
    """draw_backbone with the physical read in place of _sample.
    Works for any BackboneLevel (the deployed 128-book stack, the
    one-level frontier, or the binary banks)."""
    B = len(lab)
    units = torch.full((B, bl.NU), -1, dtype=torch.long)
    units[:, 0] = lab
    comp = {}
    for m in range(bl.M):
        yv, mm = g1._y_m(bl.build_aat(units))
        sel = torch.zeros(B, bl.NU, bl.S, dtype=torch.bool)
        sel[:, 1 + m] = True
        sel = sel.reshape(yv.shape)
        yv, sh = physical_read(yv, mm, sel, LREP + m, g1.noise, v_read, v_n, gen)
        if m in log_ctx:
            comp[m] = sh
        units[:, 1 + m] = yv.reshape(B, bl.NU, bl.S)[:, 1 + m].argmax(-1)
    out = bl.read_all(r1, units, 0.0, None)
    out[:, 0] = lab
    return out[:, 1:], comp


@torch.no_grad()
def read_patches_physical(pl, g2, lab, g, gen, v_read, v_n, chunk=512):
    B = len(lab)
    none = torch.full((B, pl.NPU), -1, dtype=torch.long)
    out = torch.empty(B, pl.NPU, dtype=torch.long)
    sh = None
    for i in range(0, B, chunk):
        yv, mm = g2._y_m(pl.build_aat(lab[i:i + chunk], g[i:i + chunk],
                                      none[i:i + chunk]))
        sel = torch.ones(yv.shape, dtype=torch.bool)
        yv, s = physical_read(yv, mm, sel, LREP + pl.NG, g2.noise, v_read, v_n, gen)
        sh = sh or s
        out[i:i + chunk] = yv.reshape(-1, pl.NPU, SYM).argmax(-1)
    return out, sh


def log_comp(comp, log):
    log(f"    {'read':<8}{'m_med':>7}{'G_sum':>9}{'t_b':>9}{'sigma':>8}"
        f"{'th%':>5}{'fl%':>5}{'cm%':>5}")
    for k, sh in comp.items():
        th, fl, cm, sg, m_med, gs, t_b = sh
        name = f"ctx{k}" if k != "patch" else "patch"
        log(f"    {name:<8}{m_med:>7.0f}{gs * 1e6:>8.2f}uS{t_b * 1e9:>7.1f}ns"
            f"{sg:>8.4f}{100 * th:>5.0f}{100 * fl:>5.0f}{100 * cm:>5.0f}")
