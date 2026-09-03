"""Their FID, run unmodified.

The evaluation code of the DTM replication code base is fetched
read-only into `vendor/dtm-replication` by `fetch_vendor.sh` at a
recorded commit. Nothing here edits it: it is imported as a package
and called. The Inception weights their `utils.download` fetches are
cached to the path that function looks in, so their code takes its own
path with no patch.

Their protocol, read off their source and then confirmed against their
own shipped statistics by `01_gate.py`:

  fid/fid.py       256x256 bilinear resize with antialiasing, grayscale
                    tiled to three channels, scaled to [-1, 1],
                    2048-dimensional pool features, Frechet distance via
                    scipy's matrix square root
  DTM_config.py     512 images per digit, so 5120 generated
  utils.py          binarize at 0.1 (`images > threshold`)
  fid/precomputed_stats/bw_fashion_mnist_train.npz
                    mu and sigma over the 60,000-image train split

Fashion-MNIST comes from the canonical idx files in `data/`, not from
the shuffled split the rest of this repository uses. Their reference is
the train split in canonical order and the self-consistency check only
passes if the data matches.
"""
import gzip
import hashlib
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = HERE / "vendor" / "dtm-replication"
DATA = ROOT / "data"
GEN = HERE / "gen"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

REF_BW_FASHION = (VENDOR / "thrmlDenoising" / "fid" / "precomputed_stats"
                  / "bw_fashion_mnist_train.npz")

# Their fixed numbers.
THEIR_THRESH = 0.1          # utils.py load_dataset(threshold=0.1)
THEIR_N_GEN = 5120          # 512 per digit x 10
PRIMARY_BAR = 24.9          # their 8-step DTM: the number to beat
DTM_CHAIN = ((1, 1.961e-9, 78.4), (2, 3.921e-9, 32.6), (4, 7.843e-9, 28.5),
             (6, 1.176e-8, 26.5), (8, 1.568e-8, 24.9))
OUR_THRESH = 0.5            # the midpoint, kept only for contrast
SEED_BLOCK = 20000


def make_log(name):
    fh = open(HERE / f"{name}.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        print(msg, file=fh, flush=True)
    return log, fh


# ---- data ---------------------------------------------------------------

IDX_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _read_idx(path):
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    ndim = int.from_bytes(raw[:4], "big") & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big")
            for i in range(ndim)]
    return np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8).reshape(dims)


def fashion(split):
    """Canonical Fashion-MNIST: uint8 [N, 28, 28] and labels [N].

    Read here rather than through `chill_zen.data` so that this environment
    needs nothing from the emulator's, and so that the two data orders
    cannot be mixed by an import.
    """
    img, lab = IDX_FILES[split]
    return _read_idx(DATA / img), _read_idx(DATA / lab)


def binarize(x_uint8, thresh):
    """Their pipeline: divide by 255, then compare. float32 in {0, 1}."""
    return (np.asarray(x_uint8, np.float32) / 255.0 > thresh).astype(np.float32)


# ---- their FID ----------------------------------------------------------
_APPLY = None


def _apply():
    global _APPLY
    if _APPLY is None:
        from thrmlDenoising.fid.fid import get_apply_fn
        t0 = time.time()
        _APPLY = get_apply_fn()
        print(f"[fid] inception loaded ({time.time() - t0:.0f}s)", flush=True)
    return _APPLY


def their_features(images, batch_size=100, tag=None):
    """Pool features through their compute_statistics path.

    This reproduces the jitted body of their `compute_statistics`
    exactly -- same resize, tile, scaling and apply function -- and
    keeps the activations so several statistics can be built from one
    pass. `01_gate.py` is the check that it matches their own function
    to floating-point noise.
    """
    import math

    import jax
    import jax.numpy as jnp

    if tag is not None:
        cf = CACHE / f"feat-{tag}.npy"
        if cf.exists():
            return np.load(cf)

    params, apply_fn = _apply()
    images = np.asarray(images, np.float32)
    if images.ndim == 3:
        images = images[..., None]
    assert images.ndim == 4 and images.shape[1] == images.shape[2]
    n_channels = images.shape[3]

    @jax.jit
    def compute_act(_x):
        _x = jnp.array(_x)
        if _x.shape[1] != 256:
            _x = jax.image.resize(_x, (_x.shape[0], 256, 256, 1),
                                  method="bilinear", antialias=True)
        if n_channels == 1:
            _x = jnp.tile(_x, (1, 1, 1, 3))
        elif n_channels != 3:
            raise ValueError("Images must have 1 or 3 channels.")
        _x = 2 * _x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(_x))
        return pred.squeeze(axis=1).squeeze(axis=1)

    nb = math.ceil(len(images) / batch_size)
    act, t0 = [], time.time()
    for i in range(nb):
        act.append(np.array(compute_act(
            images[i * batch_size:(i + 1) * batch_size])))
        if i % 20 == 0 or i == nb - 1:
            done = min((i + 1) * batch_size, len(images))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  [feat] {done}/{len(images)}  {rate:.1f} img/s",
                  flush=True)
    act = np.concatenate(act, 0)
    if tag is not None:
        np.save(CACHE / f"feat-{tag}.npy", act)
    return act


def stats_of(act):
    return np.mean(act, axis=0), np.cov(act, rowvar=False)


def frechet(mu1, sigma1, mu2, sigma2):
    """Their compute_frechet_distance, imported rather than retyped."""
    from thrmlDenoising.fid.fid import compute_frechet_distance
    fid, t1, t2 = compute_frechet_distance(mu1, mu2, sigma1, sigma2)
    return float(fid), float(t1), float(t2)


def ref_stats(path=REF_BW_FASHION):
    d = np.load(path)
    return d["mu"], d["sigma"]        # dtypes exactly as they ship them


def their_fid(images, tag=None, ref=None):
    mu_r, sig_r = ref_stats() if ref is None else ref
    act = their_features(images, tag=tag)
    mu, sig = stats_of(act)
    return frechet(mu, sig, mu_r, sig_r)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
