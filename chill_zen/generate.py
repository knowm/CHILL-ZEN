"""The generation recipe (paper Sec. III B).

From a label alone:

1. **draw** -- 128 sequential sampled reads of the backbone bank at
   T = 0.1, each conditioned on the label and the symbols already
   committed. This is the only stochastic step: the sample is the
   substrate's own read noise, taken through the weight it belongs to.
2. **cold sweep** -- one parallel read of the backbone repair bank at
   T = 0, full commit.
3. **patch level** -- one parallel sampled read of the patch fill bank
   at T = 0.1 under the drawn backbone held as conditioning. Ninety-
   eight addresses in one instruction, not ninety-eight instructions.
4. **joint sweep** -- one cold parallel read of the joint bank over all
   226 addresses; the levels reconcile each other.
5. **render** -- decode the backbone code, add the decoded residual
   patches, clamp to [0, 1]. Host-side floating point.

131 sequential steps in all: 128 draw reads plus three parallel
sweeps. The reconstruction arm replaces step 1 with a real image's
encoded backbone code and runs every other step exactly as deployed.
"""
import torch

from ktram_neural_core.torch import _lane

from .codec import decode_global, decode_slots, from_patches

T_GEN = 0.1          # the draw temperature
LEVELS = 16
FG_THR = 1.0 / LEVELS


@torch.no_grad()
def draw_backbone(level, g1, r1, lab, gen, t_gen=T_GEN, with_label=False,
                  sigma_flat=None, t_sched=None):
    """Label -> backbone code. M sampled AR reads, then one cold sweep.

    `with_label=True` keeps the label column, which is what the
    backbone-only sweep stores so a later energy census can replay the
    exact contexts each read saw.

    Two ablation arguments, both defaulting to the deployed behaviour:
    `sigma_flat` replaces the device's state-dependent sigma(y, m) with
    a constant (E2a), and `t_sched`, a sequence of M temperatures,
    replaces the fixed `t_gen` with a per-step schedule along the AR
    chain (E2b).
    """
    from .levels import _sample
    B = len(lab)
    units = torch.full((B, level.NU), -1, dtype=torch.long)
    units[:, 0] = lab
    for m in range(level.M):
        t_m = t_gen if t_sched is None else float(t_sched[m])
        yv, mm = g1._y_m(level.build_aat(units))
        yv = _sample(yv, mm, t_m, g1.noise, gen, sigma_flat)
        units[:, 1 + m] = yv.reshape(B, level.NU, level.S)[:, 1 + m].argmax(-1)
    out = level.read_all(r1, units, 0.0, None)
    out[:, 0] = lab
    return out if with_label else out[:, 1:]


@torch.no_grad()
def draw_patches(pl, g2, r2, y, g, gen, mode="parallel", dose=1,
                 t_gen=T_GEN, sigma_flat=None):
    """Backbone code -> patch symbols.

    `mode="parallel"` is the deployed step: one sampled read of all 98
    addresses at once. `mode="ar"` reads them one at a time, which is
    the control that measures what the parallel read costs. `dose=1`
    adds the patch repair bank's cold sweep; the deployed system uses
    dose 0 here and lets the joint bank do the reconciliation.
    """
    B = len(y)
    none = torch.full((B, pl.NPU), -1, dtype=torch.long)
    if mode == "parallel":
        p = pl.read_all(g2, y, g, none, t_gen, gen, sigma_flat=sigma_flat)
    else:
        p = none.clone()
        for u in range(pl.NPU):
            p[:, u] = pl.read_all(g2, y, g, p, t_gen, gen)[:, u]
    if dose:
        p = pl.read_all(r2, y, g, p, 0.0, None)
    return p


def render(g_codes, p_codes, backbone_book, patch_book):
    """Decode both levels into images [N, 28, 28] in [0, 1].

    Returns (render, backbone-only decode).
    """
    dec = decode_global(g_codes.long(), backbone_book["bias"],
                        backbone_book["atoms"], backbone_book["cfg"]["p"])
    slots = p_codes.long().reshape(len(p_codes), 49, -1)
    rend = dec + from_patches(decode_slots(slots, patch_book["bias"],
                                           patch_book["W"],
                                           patch_book["cfg"]["p"]))
    return (rend.reshape(-1, 28, 28).clamp(0, 1),
            dec.reshape(-1, 28, 28).clamp(0, 1))


def render_backbone(g_codes, backbone_book):
    """Backbone-only render, the one-level system's output."""
    dec = decode_global(g_codes.long(), backbone_book["bias"],
                        backbone_book["atoms"], backbone_book["cfg"]["p"])
    return dec.reshape(-1, 28, 28).clamp(0, 1)
