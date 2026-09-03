# The head-to-head

Paper Sec. V C. This directory reproduces the only comparison in the
paper that is scored by somebody else's code.

The DTM paper reports quality as FID on binarized Fashion-MNIST and
states nothing else about the measurement: no sample size, no feature
extractor, no binarization threshold, no evaluation appendix. Its
released library contains no FID code at all. The protocol is
nevertheless fully determined by the DTM replication code base linked
from that library, which ships both the evaluation code and precomputed
reference statistics.

So the comparison is made by driving *their* code, unmodified, against
*their* shipped reference, at *their* threshold and *their* sample
count. The vendored tree is read-only; the harness imports it as a
package and calls it.

## What their code does

| element | value | where it is set |
|---|---|---|
| extractor | InceptionV3, jax port of the pytorch-fid weights, 2048-d pool | `thrmlDenoising/fid/inception.py` |
| resize | 256x256 bilinear, antialiased (not the conventional 299) | `thrmlDenoising/fid/fid.py` |
| channels | grayscale tiled to 3, then scaled to [-1, 1] | same |
| distance | Frechet distance via `scipy.linalg.sqrtm` | same |
| generated n | 5120 = 512 per class | `DTM_config.py` |
| reference | shipped mu and sigma over the 60,000-image train split | `fid/precomputed_stats/bw_fashion_mnist_train.npz` |
| binarization | **0.1** | `thrmlDenoising/utils.py` |

Two of those are unconventional and neither is recoverable from the
paper: the 256-pixel resize and the 0.1 threshold.

## Two environments

The emulator is pinned to torch 2.2.2 with numpy below 2; their FID
code needs JAX with numpy 2. They cannot share an interpreter. The
interface between them is `gen/*.npy`: generation writes grayscale
renders, scoring thresholds and scores them.

```
python -m venv .venv           # emulator: pip install -r requirements.txt
python -m venv .venv-dtm      # scoring:  pip install -r experiments/head_to_head/requirements-dtm.txt
```

Neither environment ever imports the other's pins.

## Data order matters

Fashion-MNIST here comes from the canonical idx files, **not** from the
shuffled split the rest of the repository uses. Their reference is the
train split in canonical order, and the self-consistency check below
only passes if the data matches.

## Run order

```bash
./experiments/head_to_head/fetch_vendor.sh          # clone + checksums
.venv-dtm/bin/python experiments/head_to_head/01_gate.py       # ~2.5 h
.venv/bin/python      experiments/head_to_head/02_generate.py   # ~10 min
.venv-dtm/bin/python experiments/head_to_head/03_score.py      # ~1 h
.venv-dtm/bin/python experiments/head_to_head/04_strips.py     # seconds
.venv/bin/python      experiments/head_to_head/05_census.py     # seconds
.venv-dtm/bin/python experiments/head_to_head/06_binary_arm.py # ~1 h / minutes cached
```

`06_binary_arm.py` scores the binary-trained arms of Table VII; it
reads what `experiments/19_binary_verdict.py` rendered into
`gen/binary-*.npy`.

That sequence reproduces the deployed-temperature numbers, which the
paper reports as superseded. The rows Table VII actually carries are
drawn at the physical operating point, and they need a second pass:

```bash
# render the physical-point arms (emulator environment)
.venv/bin/python      experiments/head_to_head/25_comparator_headtohead.py  # ~10 min
.venv/bin/python      experiments/27_binary_comparator_headtohead.py        # ~5 min
.venv/bin/python      experiments/24_binary_comparator_sweep.py             # ~20 min

# score every one of them in a single pass
.venv-dtm/bin/python experiments/head_to_head/07_comparator_score.py       # ~1 h / cached
```

`07_comparator_score.py` globs `gen/comparator-*.npy`, so it scores the
grayscale rows, the binary one-level rows and the binary sweep
together — which is why there is no separate scorer for each.

The Sec. V C audit is a third pass, and independent of the above:

```bash
.venv-dtm/bin/python experiments/head_to_head/30_bernoulli_baseline.py     # < 1 min
.venv/bin/python      experiments/31_free_draw.py                           # ~25 min
.venv-dtm/bin/python experiments/head_to_head/32_audit_score.py            # minutes
.venv-dtm/bin/python experiments/head_to_head/33_binary_memorization.py    # minutes
```

Inception features are cached in `cache/feat-*.npy`, keyed by arm and
threshold, so re-scoring is free.

## The gate

`01_gate.py` is the acceptance test for everything downstream. If a run
of it does not reproduce these, nothing after it should be believed:

| check | expected |
|---|---|
| threshold probe picks | 0.1, by a factor of 8.1 in relative L2 over the next candidate |
| FID(60,000 train at 0.1, their shipped statistics) | 0.000039 |
| mu / sigma maximum absolute difference | 6.0e-04 / 3.0e-04 |
| held-out floor, test at n = 5120 | 2.142 |
| held-out floor, test at n = 10,000 | 1.194 |
| the same 60,000 train images at the midpoint | 29.75 |

Agreement at 4e-05 across a 2048x2048 covariance pins down the
reference identity, the threshold, the data order, the scaling, the
resize, the extractor and the statistics all at once. The last row is
the measurement-convention result the paper reports: real data at the
wrong threshold scores worse than their generator.

## Checksums

Nothing below is redistributed here; `fetch_vendor.sh` fetches each
file and verifies it.

| item | identity |
|---|---|
| `pschilliOrange/dtm-replication` | commit `7c22d19c218ab353a770f7c2b1504570b7cbe3ba` |
| `bw_fashion_mnist_train.npz` | `66003004dc99115b20c146bd3c2a7d9d85fb85a3c0c9e991f11951933f97c5d8` |
| `inception_v3_weights_fid.pickle` | `4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77` |
| `train-images-idx3-ubyte.gz` | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `train-labels-idx1-ubyte.gz` | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |
| `t10k-images-idx3-ubyte.gz` | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` |
| `t10k-labels-idx1-ubyte.gz` | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` |

The Inception weights are fetched by their own code on first use, into
`$TMPDIR/jax_fid/`, from the URL in their `inception.py`.

## If the upstream repository goes away

`fetch_vendor.sh` depends on one third-party repository staying
reachable. The pinned commit is archived in Software Heritage, which
holds the identical bytes — the archived
`thrmlDenoising/fid/precomputed_stats/bw_fashion_mnist_train.npz`
carries the same sha256 as the table above.

| archive | identity |
|---|---|
| Software Heritage revision | `swh:1:rev:7c22d19c218ab353a770f7c2b1504570b7cbe3ba` |

Browse it at
`https://archive.softwareheritage.org/browse/revision/7c22d19c218ab353a770f7c2b1504570b7cbe3ba/`.

That is an archive reference, not a copy hosted here: nothing in this
repository redistributes their code, and the checksums above remain
the test of whether you have the right files.

## What is scored, and what is not

The scored arms are the frozen grayscale systems of Secs. III-IV: the
same books, the same banks, the same recipe, rendering grayscale in
[0, 1]. The threshold is the only host-side step. Nothing was trained,
tuned or refit for this measurement, and no binarized data was used
anywhere in fitting.

`real-train` and `codec-ceiling` are not generators. The first is real
data through the identical path; the second is the codec's render of
real codes, the bound every arm through that codec sits under. Neither
is eligible for the bar, and `03_score.py` will not name one as a
winner.
