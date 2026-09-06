"""The AAT layouts of the two levels and of the joint bank (paper Sec. III B).

Every bank in this work reads one activation address tuple and writes
back symbols at target addresses. The three layouts differ only in
which addresses are inputs, which are targets, and how many there are.

`BackboneLevel` -- the global level. The tuple is
`[label x6 | M backbone addresses]`; targets are the label address and
all M backbone addresses; lanes are `(1 + M) x 16`. The label is
replicated six times on the input side because one address out of
1 + M has vanishing influence otherwise; it is never dropped during
teaching.

`PatchLevel` -- the residual level over a 7x7 grid of 4x4-pixel slots
at two books per slot. The tuple is
`[label x6 | M backbone addresses | 98 patch addresses]`; the backbone
addresses are held conditioning, never dropped and never targets; only
the 98 patch addresses are targets, so lanes are `98 x 16`.

`JointLevel` -- one bank over the same 232-space tuple whose targets
are *all* 226 addresses, both levels at once. It is what lets the
patch level's evidence repair a backbone address: a single cold
parallel sweep in which the levels correct each other.

Group sizes: every address is a WTA Group of 16 lanes (the label
address uses 10 of its 16 symbols).
"""
import os

import torch

from ktram_neural_core.torch import _lane

from .lanes import INIT, LREP, LaneBank
from .data import SEED

S = 16                      # symbols per address (the WTA Group width)
NCH = 16                    # channels per input space


def parse_tag(tag):
    """'128x16@p0.5' -> (128, 16, 0.5)."""
    m, rest = tag.split("x")
    s, p = rest.split("@p")
    return int(m), int(s), float(p)



def _sample(yv, mm, T, noise, gen, sigma_flat=None):
    """The hot read. sigma_flat=None is the deployed device read
    (sample_read's state-dependent sigma(y, m)); a float replaces that
    sigma with a constant at the same T -- the E2a ablation. Same RNG
    consumption either way. comparator=False: this is a device-only
    draw by design -- the comparator term is applied separately and
    deliberately in `chill_zen.physical`, so taking it here too would
    count it twice."""
    if sigma_flat is None:
        return _lane.sample_read(yv, mm, T, noise, gen, comparator=False)
    if T <= 0.0:
        return yv
    r = torch.randn(yv.shape, generator=gen, dtype=yv.dtype,
                    device=yv.device)
    return (yv + T * sigma_flat * r).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# The backbone level.
# ---------------------------------------------------------------------------

# The read the repair banks' G fills are drawn at (see chill_zen.teach).
# It goes into every taught bank's config, so a bank taught at a
# different fill read does not match and is refit rather than reused.
FILL_V_READ = 0.010     # the sense floor, as deployed
# Code 245, sigma 0.500 at the sense floor. Chosen by the teaching-level
# sweep, not by argument: the fills' level is a knob of its own, and
# teaching at code 45 (the operating point at the time) is markedly worse
# than teaching hot. The register's ceiling here is 0.520, so there is
# little left above this. CHILLZEN_FILL_VCMP overrides it to re-run the
# sweep; the level in force is recorded in every bank's config by
# fill_read_tag, so banks taught at different levels do not match.
FILL_V_CMP = float(os.environ.get("CHILLZEN_FILL_VCMP", 5e-3))


def fill_read_tag(v_read=None, v_cmp=None):
    """How the G fills were drawn, as one config-comparable string."""
    v_read = FILL_V_READ if v_read is None else v_read
    v_cmp = FILL_V_CMP if v_cmp is None else v_cmp
    return (f"V={v_read * 1e3:g}mV,v_cmp={v_cmp * 1e6:g}uV,"
            f"read_noise=0.02")


def backbone_cfg(tag, M, nch, batch=256, epochs=15, fill_read=None,
                 q_max=0.5):
    """The teaching configuration recorded inside a frozen bank file."""
    return dict(seed=SEED, init=INIT, tag=tag, m=M, lrep=LREP, nch=nch,
                batch=batch, epochs=epochs,
                fill_read=fill_read or fill_read_tag(),
                q=f"U(0,{q_max})")


class BackboneLevel:
    """Sizes, tuples and reads for one backbone code (books x symbols @ keep-p)."""

    def __init__(self, tag, codes_tr, codes_ev, y_tr, y_ev):
        self.tag = tag
        self.M, self.S, _ = parse_tag(tag)
        assert self.M == codes_tr.shape[1]
        self.NCH = max(self.S, 10)
        self.NU = 1 + self.M                     # target addresses
        self.LANES = self.NU * self.S
        self.KSP = LREP + self.M                 # input spaces
        self.u_tr = torch.cat([y_tr[:, None].long(), codes_tr.long()], 1)
        self.u_ev = torch.cat([y_ev[:, None].long(), codes_ev.long()], 1)
        self.maj = torch.cat([
            torch.bincount(y_tr, minlength=10).argmax()[None],
            torch.stack([torch.bincount(codes_tr[:, m].long(),
                                        minlength=self.S).argmax()
                         for m in range(self.M)])])
        self.tgt_tr = torch.arange(self.NU)[None, :] * self.S + self.u_tr
        # per-(class, address) symbol marginals -- the corruption tables
        tab = torch.zeros(10, self.NU, self.S)
        tab[:, 0] = torch.eye(10, self.S)
        for m in range(self.M):
            for c in range(10):
                h = torch.bincount(codes_tr[y_tr == c, m].long(),
                                   minlength=self.S).float()
                tab[c, 1 + m] = h / h.sum()
        self.tab = tab

    def build_aat(self, units):
        """[B, 1+M] symbols -> [B, 6+M] input tuple (label replicated)."""
        return torch.cat([units[:, :1].expand(-1, LREP), units[:, 1:]], 1)

    def make_bank(self, seed_base):
        return LaneBank(self.LANES, self.KSP, self.NCH, init=INIT,
                        seed=seed_base)

    @torch.no_grad()
    def read_all(self, bank, units, T, gen, chunk=512, sigma_flat=None):
        """One parallel read of all 1+M addresses -> winning symbols.

        `sigma_flat` replaces the device's state-dependent sigma(y, m)
        with a constant (the E2a ablation). None, the default, is the
        deployed behaviour.
        """
        out = torch.empty(len(units), self.NU, dtype=torch.long)
        for i in range(0, len(units), chunk):
            yv, mm = bank._y_m(self.build_aat(units[i:i + chunk]))
            if T > 0:
                yv = _sample(yv, mm, T, bank.noise, gen, sigma_flat)
            out[i:i + chunk] = yv.reshape(-1, self.NU, self.S).argmax(-1)
        return out

    def bernoulli_mask(self, n, gen):
        """Per-sample keep-p ~ U(0,1); the label address is always held."""
        keep_p = torch.rand(n, generator=gen)
        mask = torch.rand(n, self.NU, generator=gen) < keep_p[:, None]
        mask[:, 0] = True
        return keep_p, mask

    def masked(self, units, mask):
        return torch.where(mask, units,
                           torch.full((1,), -1, dtype=torch.long))

    def marginal_draw(self, y, gen):
        """Draw a symbol per address from that class's train marginal."""
        pr = self.tab[y.long()]
        cdf = pr.cumsum(-1)
        u = torch.rand(*pr.shape[:-1], 1, generator=gen)
        return (u > cdf).long().sum(-1).clamp(max=self.S - 1)


# ---------------------------------------------------------------------------
# The patch level and the joint bank.
# ---------------------------------------------------------------------------

def patch_cfg(backbone, patch, ksp, nch, batch=256, epochs=15,
              fill_read=None, q_max=0.5, q_gap=0.25):
    return dict(seed=SEED, init=INIT, backbone=backbone, patch=patch,
                ksp=ksp, nch=nch, lrep=LREP, batch=batch, epochs=epochs,
                fill_read=fill_read or fill_read_tag(),
                q=f"U(0,{q_max})", q_gap=q_gap)


def joint_cfg(backbone, patch, ksp, nch, batch=256, epochs=15,
              fill_read=None, q_max=0.5):
    return dict(seed=SEED, init=INIT, backbone=backbone, patch=patch,
                ksp=ksp, nch=nch, fill_read=fill_read or fill_read_tag(),
                q=f"U(0,{q_max})",
                corrupt="marginal+G-fills-both-levels", batch=batch,
                epochs=epochs, sharp=True)


class PatchLevel:
    """The residual level: backbone addresses held, patch addresses taught.

    `g_tr`/`g_ev` are backbone symbols [N, NG]; `p_tr`/`p_ev` are patch
    symbols [N, NPU] with NPU = slots x books (49 x 2 = 98 deployed).
    """

    def __init__(self, backbone, patch, y_tr, y_ev, g_tr, g_ev, p_tr, p_ev):
        self.backbone, self.patch = backbone, patch
        self.NG = g_tr.shape[1]
        self.NPU = p_tr.shape[1]
        self.S = S
        self.NCH = NCH
        self.KSP = LREP + self.NG + self.NPU
        self.LANES = self.NPU * S
        self.y_tr, self.y_ev = y_tr, y_ev
        self.g_tr, self.g_ev = g_tr, g_ev
        self.p_tr, self.p_ev = p_tr, p_ev
        self._pu = torch.arange(self.NPU)
        self.tgt_tr = self._pu[None, :] * S + p_tr
        # per-class marginals at both levels, and the patch majority prior
        self.gtab = torch.zeros(10, self.NG, S)
        self.ptab = torch.zeros(10, self.NPU, S)
        for c in range(10):
            for u in range(self.NG):
                h = torch.bincount(g_tr[y_tr == c, u], minlength=S).float()
                self.gtab[c, u] = h / h.sum()
            for u in range(self.NPU):
                h = torch.bincount(p_tr[y_tr == c, u], minlength=S).float()
                self.ptab[c, u] = h / h.sum()
        self.maj = torch.stack([
            torch.bincount(p_tr[:, u], minlength=S).argmax()
            for u in range(self.NPU)])

    def cfg(self, **kw):
        return patch_cfg(self.backbone, self.patch, self.KSP, self.NCH, **kw)

    def build_aat(self, y, g, patches):
        return torch.cat([y[:, None].expand(-1, LREP), g, patches], 1)

    def make_bank(self, seed_base):
        return LaneBank(self.LANES, self.KSP, self.NCH, init=INIT,
                        seed=seed_base)

    @torch.no_grad()
    def read_all(self, bank, y, g, patches, T, gen, chunk=512,
                 sigma_flat=None):
        out = torch.empty(len(patches), self.NPU, dtype=torch.long)
        for i in range(0, len(patches), chunk):
            yv, mm = bank._y_m(self.build_aat(y[i:i + chunk], g[i:i + chunk],
                                              patches[i:i + chunk]))
            if T > 0:
                yv = _sample(yv, mm, T, bank.noise, gen, sigma_flat)
            out[i:i + chunk] = yv.reshape(-1, self.NPU, S).argmax(-1)
        return out

    def bernoulli_mask(self, n, gen):
        keep_p = torch.rand(n, generator=gen)
        mask = torch.rand(n, self.NPU, generator=gen) < keep_p[:, None]
        return keep_p, mask

    @staticmethod
    def masked(patches, mask):
        return torch.where(mask, patches,
                           torch.full((1,), -1, dtype=torch.long))

    @staticmethod
    def marginal_draw(tab, y, gen):
        pr = tab[y]
        cdf = pr.cumsum(-1)
        u = torch.rand(*pr.shape[:-1], 1, generator=gen)
        return (u > cdf).long().sum(-1).clamp(max=S - 1)


class JointLevel:
    """One bank over the patch level's tuple whose targets are both levels."""

    def __init__(self, patch_level):
        self.pl = patch_level
        self.NG, self.NPU = patch_level.NG, patch_level.NPU
        self.NU = self.NG + self.NPU            # 226 deployed
        self.KSP = patch_level.KSP              # 232 deployed
        self.LANES = self.NU * S                # 3616 deployed
        self.NCH = NCH
        self._u = torch.arange(self.NU)

    def cfg(self, **kw):
        return joint_cfg(self.pl.backbone, self.pl.patch, self.KSP,
                         self.NCH, **kw)

    def make_bank(self, seed_base):
        return LaneBank(self.LANES, self.KSP, self.NCH, init=INIT,
                        seed=seed_base)

    def targets(self, g, p):
        return self._u[None, :] * S + torch.cat([g, p], 1)

    @torch.no_grad()
    def read_joint(self, bank, y, g, patches, T, gen, chunk=512):
        """One parallel read of all 226 addresses -> [B, 226]."""
        out = torch.empty(len(patches), self.NU, dtype=torch.long)
        for i in range(0, len(patches), chunk):
            yv, mm = bank._y_m(self.pl.build_aat(
                y[i:i + chunk], g[i:i + chunk], patches[i:i + chunk]))
            if T > 0:
                # device-only by design; see _sample
                yv = _lane.sample_read(yv, mm, T, bank.noise, gen,
                                       comparator=False)
            out[i:i + chunk] = yv.reshape(-1, self.NU, S).argmax(-1)
        return out


def load_patch_level(backbone, patch):
    """Assemble the patch level from the frozen codes on disk."""
    import torch as _t
    from . import artifacts
    from .data import N_TRAIN, load_fashion
    _, y = load_fashion(seed=0)
    y_tr, y_ev = y[:N_TRAIN].long(), y[N_TRAIN:].long()
    g = _t.load(artifacts.path("backbone_codes"))[backbone]
    k = _t.load(artifacts.path("patch_codes"))[patch]
    npu = k["tr"].shape[1] * k["tr"].shape[2]
    return PatchLevel(backbone, patch, y_tr, y_ev,
                      g["tr"].long(), g["ev"].long(),
                      k["tr"].long().reshape(-1, npu),
                      k["ev"].long().reshape(-1, npu))
