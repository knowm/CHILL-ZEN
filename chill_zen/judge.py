"""The judge (paper Sec. IV A).

Four numbers, all on grayscale renders against a real-image reference
measured the same way. No FID appears here; the comparison with
Ref. [DTM] is a separate protocol, in `experiments/head_to_head/`.

* **critic** -- label agreement of a frozen pixel-space MLP
  (784 -> 256 -> ReLU -> 10) trained once on the real train split. The
  real-reference row is the bar; parity with it is the target, not 1.0.
* **div** -- within-class sample diversity: the fraction of pixel pairs
  between two samples of a class differing by more than 1/16.
* **tone** -- the spread of per-image foreground mean level within a
  class, in 16ths.
* **std-ratio** -- foreground pixel standard deviation relative to
  real. It is the contrast alarm: a system that over-drives its decode
  looks sharp and scores well on the critic.

`seam_ratio` is reported alongside: the mean pixel step across the tile
boundaries of a 4x4 patch grid relative to the step elsewhere. A global
code has no tile boundaries, so it should sit near the real value by
construction; a per-patch code does not.
"""
import torch
from torch import nn

LEVELS = 16
FG_THR = 1.0 / LEVELS
_SEAMS = torch.tensor([6, 13, 20])         # 7x7-pixel tile boundaries


def make_critic():
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def load_critic(path):
    critic = make_critic()
    critic.load_state_dict(torch.load(path)["state"])
    critic.eval()
    return critic


def train_critic(X_tr, y_tr, X_ev, y_ev, seed=0, epochs=8, batch=256,
                 lr=1e-3, log=print):
    """The frozen judge, trained once. Deterministic at a fixed seed."""
    import time
    import torch.nn.functional as F
    torch.manual_seed(seed)
    critic = make_critic()
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = len(X_tr)
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(critic(X_tr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = (critic(X_ev).argmax(1) == y_ev).float().mean().item()
    log(f"[critic] trained ({time.time() - t0:.0f}s)  eval {acc:.4f} raw")
    critic.eval()
    return critic, acc


@torch.no_grad()
def judge(critic, img, y):
    """-> (critic agreement, within-class diversity), both class-averaged."""
    pred = critic(img.reshape(len(img), -1)).argmax(1)
    agree = torch.tensor([(pred[y == c] == c).float().mean()
                          for c in range(10)])
    div = torch.zeros(10)
    for c in range(10):
        s = img[y == c]
        diff = (s[:, None] - s[None, :]).abs() > FG_THR
        iu = torch.triu_indices(len(s), len(s), offset=1)
        div[c] = diff[iu[0], iu[1]].float().mean()
    return agree.mean().item(), div.mean().item()


@torch.no_grad()
def tone_spread(img, y):
    out = torch.zeros(10)
    for c in range(10):
        s = img[y == c]
        means = torch.stack([(im[im >= FG_THR] * LEVELS).mean()
                             if (im >= FG_THR).any() else torch.tensor(0.0)
                             for im in s])
        out[c] = means.std()
    return out.mean().item()


def fg_std(img):
    """Foreground pixel standard deviation -- the contrast alarm."""
    fg = img >= FG_THR
    return img[fg].std().item()


def seam_ratio(img01):
    dv = (img01[:, :, 1:] - img01[:, :, :-1]).abs()
    dh = (img01[:, 1:, :] - img01[:, :-1, :]).abs()
    seam_cols = torch.zeros(27, dtype=torch.bool)
    seam_cols[_SEAMS] = True
    seam = torch.cat([dv[:, :, seam_cols].reshape(-1),
                      dh[:, seam_cols, :].reshape(-1)])
    rest = torch.cat([dv[:, :, ~seam_cols].reshape(-1),
                      dh[:, ~seam_cols, :].reshape(-1)])
    return (seam.mean() / rest.mean()).item()


def verdict_row(critic, img, y, real_std):
    """The five-column judge row for one arm."""
    a, dv = judge(critic, img, y)
    return (a, dv, tone_spread(img, y), seam_ratio(img),
            fg_std(img) / real_std)
