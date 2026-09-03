"""The lane bank: reads and weight updates on the emulated substrate.

`LaneBank` is `ktram_neural_core.torch.Classifier` with one override.
The emulator's reference `adapt` walks a batch synapse by synapse;
`LaneBank` runs the identical arithmetic as two dense matrix products
over the one-hot activity matrix of the batch, which is what makes a
68,000-image teaching epoch over a few thousand lanes finish on a CPU.

Nothing about the substrate changes. A whole adapt batch is
accumulate-then-clamp in the emulator's own semantics, so each phase's
weight update is exactly `D^T A` over the activity matrix A (one
column per (space, symbol)) and each phase's fresh read is exactly
`A W` over the current (difference, magnitude) table. Same float64
lane walk, same rounding, same clamp order; only the summation is
dense. Every product is an integer and every partial sum stays far
below 2^24, where float32 is exact -- the bound is asserted rather
than assumed. CPU only: a CUDA matmul may use TF32, which would break
exactness, so a non-CPU bank falls back to the reference path.

The two feedback modes used in this work are the emulator's:
`feedback="soft"` drives only the instructed lanes (the G banks, which
must fill an open address without being told what is wrong), and
`feedback="hard"` also punishes lanes that fired without instruction
(the R banks, whose job is selection).
"""
import torch

from ktram_neural_core.torch import Classifier, _lane
from ktram_neural_core.torch._lane import GMAX, GMIN, java_round, rank_cut

INIT = "low"        # the weight initialization used by every bank here
LREP = 6            # label replication: the label occupies 6 input spaces


def snapshot(bank):
    """A detached copy of a bank's weights, for best-probe selection."""
    return {k: v.clone() for k, v in bank.state_dict().items()}


class LaneBank(Classifier):
    """A bank of neural lanes whose adapt runs as dense GEMMs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dense = {}

    def _load_from_state_dict(self, *args, **kwargs):
        self._dense = {}
        super()._load_from_state_dict(*args, **kwargs)

    def _check_bound(self):
        if self.num_spaces * 2 * GMAX >= 2 ** 24:
            raise ValueError("K * max|w| >= 2**24; fp32 GEMM sums would round")

    def adapt(self, aat, target, feedback="hard", per_group=False):
        self._dense = {}
        grouped = self.num_groups is not None
        if (grouped and not per_group) or aat.device.type != "cpu":
            self._bag = {}
            return super().adapt(aat, target, feedback=feedback,
                                 per_group=per_group)
        if self.frozen:
            raise RuntimeError("frozen pack weights cannot adapt; lift live "
                               "state via from_core")
        self._check_per_group(per_group)
        self._check_bound()
        self._bag = {}
        K, L, S = self.num_spaces, self.num_lanes, self.num_channels
        G = self.num_groups
        gshape = () if G is None else (G,)
        k_axes = 2 if per_group else 1
        lead = aat.shape[:-k_axes]
        a = (aat.reshape(-1, G, K) if per_group else aat.reshape(-1, K)) \
            .to(torch.int64)
        B = a.shape[0]
        tgt = target.reshape(B, *gshape, -1).to(torch.int64)

        # One-hot activity over (space, symbol) columns; an open address
        # (-1) contributes nothing.
        cols = torch.arange(K) * S + a.clamp(min=0)
        on = (a >= 0).to(torch.float32)
        if per_group:
            act = torch.zeros(G, B, K * S)
            act.scatter_(2, cols.transpose(0, 1), on.transpose(0, 1))
        else:
            act = torch.zeros(B, K * S)
            act.scatter_(1, cols, on)

        def read():
            d = (self.ga - self.gb).to(torch.float32)
            m = (self.ga + self.gb).to(torch.float32)
            if per_group:
                w = torch.cat([d.reshape(G, L, K * S),
                               m.reshape(G, L, K * S)], 1).transpose(1, 2)
                out = torch.bmm(act, w).transpose(0, 1)
            else:
                w = torch.cat([d.reshape(L, K * S), m.reshape(L, K * S)], 0)
                out = act @ w.T
            return out[..., :L].to(torch.int32), out[..., L:].to(torch.int32)

        def apply(dga, dgb, mask):
            D = torch.cat([dga * mask, dgb * mask], -1).to(torch.float32)
            if per_group:
                dW = torch.bmm(act.transpose(1, 2),
                               D.permute(1, 0, 2)).transpose(1, 2)
                self.ga += dW[:, :L].to(torch.int32).reshape(G, L, K, S)
                self.gb += dW[:, L:].to(torch.int32).reshape(G, L, K, S)
            else:
                dW = D.T @ act
                self.ga += dW[:L].to(torch.int32).reshape(L, K, S)
                self.gb += dW[L:].to(torch.int32).reshape(L, K, S)
            self.ga.clamp_(GMIN, GMAX)
            self.gb.clamp_(GMIN, GMAX)

        # Feed-forward drive at full voltage; the read itself drives every
        # active synapse.
        top, bot = read()
        dt = _lane.y_dtype(top.device)
        y = _lane.divide(top, bot, dt)
        vy = self.vf * y
        dga = java_round(self.vf - vy).to(torch.int32)
        dgb = java_round(vy + self.vf).to(torch.int32)
        active_read = bot != 0
        apply(dga, dgb, active_read)

        # Instruction select from the feed-forward read; an uninstructed
        # target parks in a spare column.
        tmask = torch.zeros(B, *gshape, L + 1, dtype=torch.bool)
        tmask.scatter_(-1, torch.where(tgt >= 0, tgt, L), True)
        tmask = tmask[..., :L]
        fired = y > 0
        is_rl = (fired & ~tmask) if feedback == "hard" \
            else torch.zeros_like(tmask)

        # The feedback phase is itself a read: its update uses the fresh
        # post-drive y of its own lane.
        top2, bot2 = read()
        y2 = _lane.divide(top2, bot2, dt)
        vy_rf = self.vr * y2
        dga_rf = java_round(self.vr - vy_rf).to(torch.int32)
        dgb_rf = java_round(vy_rf + self.vr).to(torch.int32)
        vy_rh = 1.0 * self.vr
        dga_rh = int(java_round(torch.tensor(self.vr - vy_rh)).item())
        dgb_rh = int(java_round(torch.tensor(vy_rh + self.vr)).item())
        vy_rl = -1.0 * self.vr
        dga_rl = int(java_round(torch.tensor(self.vr - vy_rl)).item())
        dgb_rl = int(java_round(torch.tensor(vy_rl + self.vr)).item())
        dga_fb = torch.where(tmask, dga_rh, torch.where(is_rl, dga_rl, dga_rf))
        dgb_fb = torch.where(tmask, dgb_rh, torch.where(is_rl, dgb_rl, dgb_rf))
        apply(dga_fb, dgb_fb, active_read)

        width = L if self.N is None else self.N
        return rank_cut(y, self.Vt, self.N).reshape(*lead, *gshape, width)
