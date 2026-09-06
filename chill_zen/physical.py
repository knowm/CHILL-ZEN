"""The physical hot read (paper Sec. II A's comparator term, Sec. IV C).

One Gaussian draw per read, clipped to [-1, 1]:

    sigma_y^2 = (0.02 * sigma_unit(y, m; V, t_b))^2 + (v_cmp / V)^2

That law is the emulator's, not this module's: it is
`ktram_neural_core.torch._lane.sample_read`, whose `NoiseParams` carries
the comparator term as `v_cmp`. This module only points the bank's own
params at a read -- the read voltage `V`, the settling-limited pulse
`t_b`, and a comparator level -- and takes the draw.

`0.02` is the device gain (Core `READ_NOISE`); it is never varied.
`t_b = max(2 ns, 5 C_LANE / G_sum)` per read, `G_sum` from the
committing lanes' median pooled magnitude and `n_sel`, on the SnCr band
-- the emulator's thermal constant corresponds to a SnCr-band Johnson
source. The band decides only whether the 2 ns timing floor binds: on
the W band it does, the read is then no longer settling-limited, and
the thermal noise is below kT/C_LANE rather than above it. At the
settling-limited pulse the thermal term is kT/C of the node and the
device conductance cancels.

The comparator level is an 8-bit trim register (`COMPARATOR_V_MIN`
100 uV, `COMPARATOR_V_STEP` 20 uV, codes 0-255, ceiling 5.20 mV), so a
level that no register could hold raises here rather than producing a
number. `v_n = 0` is the modeling switch -- the comparator is not
modeled at all -- and is the only way to get a read with no comparator
term; code 0 is the quietest comparator modeled, not the absence of one.

Cold reads are unaffected: this module only replaces the hot
autoregressive backbone draw and the parallel patch read.

Written once here and generalized over any `BackboneLevel`/`PatchLevel`
pair, so the same read drives the one-level frontier and the binary
stack as well as the deployed two-level system. No change to
`kt-ram-neural-core`; this module calls its public API.
"""
import torch

from ktram_neural_core.core import comparator_code_for, comparator_volts
from ktram_neural_core.torch import _lane

from .energy import g_sum_from_bytes, t_read
from .lanes import LREP
from .levels import S as SYM

GAIN_DEV = 0.02      # Core READ_NOISE: the device, never varied
BAND = "SnCr"        # the band the emulator's thermal constant reads as


def register_level(v_n):
    """(code, volts) for a requested comparator level, on the register.

    `v_n = 0` is the modeling switch: the comparator is not modeled, and
    the read is the device-only law. Any other level must be one the
    8-bit register reaches exactly -- below the floor, above the ceiling
    or between two codes raises, because a swept level no hardware could
    be programmed to is not an operating point.
    """
    if v_n == 0.0:
        return None, 0.0
    code = comparator_code_for(v_n)          # raises off-register
    return code, comparator_volts(code)


def params_at(base, v_read, pw, v_cmp=0.0):
    """The bank's own NoiseParams re-pointed at a read (V, t).

    `v_cmp` is named explicitly and defaults to zero. NoiseParams
    defaults it to 300 uV, and this function rebuilds the params field by
    field, so a field left unnamed is a comparator level nobody chose.
    The caller states the level or gets none."""
    return _lane.NoiseParams(a_thermal_unit=base.a_thermal_unit,
                             a_flicker_unit=base.a_flicker_unit,
                             sqrt_ref_m=base.sqrt_ref_m,
                             flicker_ln_ref=base.flicker_ln_ref,
                             ref_pw=base.ref_pw, read_pw=pw, v_read=v_read,
                             v_cmp=v_cmp)


def physical_read(yv, mm, sel, n_sel, base, v_read, v_n, gen):
    """One hot read at (V, t_b, v_n). `sel` picks the committing lanes
    for the pool load; the same (V, t_b) applies to every lane in the
    read. Returns the noisy y and the variance shares at the committing
    lanes (thermal, flicker, comparator)."""
    m_med = mm[sel].median().item()
    gs = g_sum_from_bytes(n_sel, m_med, BAND)
    t_b = t_read(gs)
    _, v_cmp = register_level(v_n)
    p = params_at(base, v_read, t_b, v_cmp=v_cmp)
    out = _lane.sample_read(yv, mm, GAIN_DEV, p, gen)
    # shares, for the composition table
    s_lane = GAIN_DEV * p.sigma_unit(yv, mm)
    s_cmp = v_cmp / v_read
    sigma = torch.sqrt(s_lane * s_lane + s_cmp * s_cmp)
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
def read_all_backbone_physical(bl, bank, units, v_read, v_n, gen, chunk=512):
    """`BackboneLevel.read_all` on the physical read.

    The parallel counterpart of `draw_backbone_physical`: one hot read of
    all 1+M addresses at (V, t_b, v_n), used for the G fills the repair
    banks teach against. `n_sel` follows the same convention as the
    autoregressive draw -- the label plus the set context units -- taken
    per chunk at the median, since a pool's states are masked to
    different depths.
    """
    out = torch.empty(len(units), bl.NU, dtype=torch.long)
    for i in range(0, len(units), chunk):
        u = units[i:i + chunk]
        yv, mm = bank._y_m(bl.build_aat(u))
        sel = torch.ones(yv.shape, dtype=torch.bool)
        n_set = int((u != -1).sum(1).median().item())
        yv, _ = physical_read(yv, mm, sel, LREP + max(n_set - 1, 0),
                              bank.noise, v_read, v_n, gen)
        out[i:i + chunk] = yv.reshape(-1, bl.NU, bl.S).argmax(-1)
    return out


@torch.no_grad()
def read_all_patch_physical(pl, bank, y, g, patches, v_read, v_n, gen,
                            chunk=512):
    """`PatchLevel.read_all` on the physical read; see above. `n_sel`
    matches `read_patches_physical`, the deployed parallel patch read."""
    out = torch.empty(len(patches), pl.NPU, dtype=torch.long)
    for i in range(0, len(patches), chunk):
        yv, mm = bank._y_m(pl.build_aat(y[i:i + chunk], g[i:i + chunk],
                                        patches[i:i + chunk]))
        sel = torch.ones(yv.shape, dtype=torch.bool)
        yv, _ = physical_read(yv, mm, sel, LREP + pl.NG, bank.noise,
                              v_read, v_n, gen)
        out[i:i + chunk] = yv.reshape(-1, pl.NPU, SYM).argmax(-1)
    return out


@torch.no_grad()
def read_joint_physical(jl, bank, y, g, patches, v_read, v_n, gen,
                        chunk=512):
    """`JointLevel.read_joint` on the physical read: one parallel read of
    all 226 addresses at (V, t_b, v_n). Used by the cold-read control of
    experiment 35, which runs the cold sweeps at the physical floor
    (50 mV, the register's code 0, device terms on) instead of as a
    noiseless argmax. `n_sel` is the full tuple, label plus both levels."""
    out = torch.empty(len(patches), jl.NU, dtype=torch.long)
    for i in range(0, len(patches), chunk):
        yv, mm = bank._y_m(jl.pl.build_aat(y[i:i + chunk], g[i:i + chunk],
                                           patches[i:i + chunk]))
        sel = torch.ones(yv.shape, dtype=torch.bool)
        yv, _ = physical_read(yv, mm, sel, jl.KSP, bank.noise, v_read, v_n,
                              gen)
        out[i:i + chunk] = yv.reshape(-1, jl.NU, SYM).argmax(-1)
    return out


@torch.no_grad()
def draw_backbone_hot_only(bl, g1, lab, gen, v_read, v_n):
    """The autoregressive hot pass of `draw_backbone_physical` without its
    closing R1 sweep, so a caller can run that sweep under more than one
    cold-read condition on the same drawn states. Returns the [B, 1+M]
    units with the label in column 0."""
    B = len(lab)
    units = torch.full((B, bl.NU), -1, dtype=torch.long)
    units[:, 0] = lab
    for m in range(bl.M):
        yv, mm = g1._y_m(bl.build_aat(units))
        sel = torch.zeros(B, bl.NU, bl.S, dtype=torch.bool)
        sel[:, 1 + m] = True
        sel = sel.reshape(yv.shape)
        yv, _ = physical_read(yv, mm, sel, LREP + m, g1.noise, v_read, v_n,
                              gen)
        units[:, 1 + m] = yv.reshape(B, bl.NU, bl.S)[:, 1 + m].argmax(-1)
    return units


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
