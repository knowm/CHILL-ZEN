# CHILL ZEN

Companion code for **CHILL ZEN** — a generative model whose every read
and every weight update is an instruction on an emulated
thermodynamic-computing substrate, and the measurements the paper
reports about it.

The paper itself is here too, as `main.tex`; `make paper` builds it.

The measurements the paper reports run from here: the codecs, the bank
teaching, the generation recipe, the judge, the scaling and
rate-distortion studies, the energy model, and the head-to-head against
the DTM/DTCA scored by that work's own evaluation code. The rough edges
are listed under [Known wrinkles](#known-wrinkles).

The lane emulator itself is not in this repository. It is
[`knowm/ktram-neural-core`](https://github.com/knowm/ktram-neural-core),
and every read and weight update below goes through it. That includes
the read-noise law of the paper's Eq. (2) in full: the two device terms
and the comparator's input-referred noise are all the emulator's, and
the comparator level is its 8-bit trim register (100 uV floor, 20 uV
step, 5.20 mV ceiling), not a number this repository invents. Where a
draw here is device-only by design, the call site says so with
`comparator=False` rather than relying on a library default.

The teaching path runs on the same read. The G fills that the R banks
and the joint bank teach against are drawn through `chill_zen.physical`
at the emulator's device noise, the 10 mV read, and the
comparator register at code 245 (`levels.FILL_V_CMP`). That setting was chosen during development. The intermediate fill-sweep
logs were not retained and generation-noise settings differed between
runs; the archived summary does not isolate the effect of fill noise. Every bank's config records its fill read, so banks taught at
another level are refitted rather than reused.

---

## Install

Two environments, because they cannot share one. The emulator is
pinned to torch 2.2.2 with numpy below 2; the evaluation code of the
head-to-head needs JAX with numpy 2. Only the head-to-head needs the
second one.

```bash
git clone https://github.com/knowm/CHILL-ZEN
cd CHILL-ZEN

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install "ktram-neural-core @ git+https://github.com/knowm/ktram-neural-core@6899ca6717377a952782724c33c85b0c4e580e56#subdirectory=python"

# only for experiments/head_to_head
python3 -m venv .venv-dtm
.venv-dtm/bin/pip install -r experiments/head_to_head/requirements-dtm.txt
```

`ktram-neural-core` is not on PyPI; install it from the repository as
above, or clone it and `pip install -e python/`.

Everything runs on a CPU. The reported runs were done on an Intel iMac,
eight threads; the per-step costs quoted below are from that machine.

## Data

Fashion-MNIST, in two orders, and they must not be mixed.

* `experiments/` uses `chill_zen.data.load_fashion`: the 70,000 images as
  OpenML serves them, permuted once by numpy's PCG64 at seed 0, then
  cut 68,000 train / 2,000 evaluation. scikit-learn downloads and
  caches it on first call. Every codec and bank in the paper was fitted
  on those 68,000 images and every probe is on the last 2,000.
* `experiments/head_to_head/` uses the canonical idx files, train and
  test kept apart, because the reference statistics it is scored
  against were computed over the canonical 60,000-image train split.
  `experiments/head_to_head/fetch_vendor.sh` downloads them into
  `data/` and checks their sha256.

## Scope of the evidence

The 2,000-image partition is used for bank-epoch and hyperparameter
selection. It is a validation partition, not an untouched test set.
Generation seeds measure sampling variation of selected banks. FID noise
levels were chosen using the same scores that are reported.

The replication gate verifies our implementation of that replication's
metric. It does not establish the scoring path used for the published
DTM numbers. Our fitting split also differs from DTM's canonical 60k.
Reconstruction FIDs are references, not bounds, and the 9.57 binary
backbone reference must not be substituted for the full two-level 2.83.

`evidence/` preserves existing logs and the corrected split audit.
Its manifest records the current local artifacts and saved renders;
these hashes were taken after the experiments and cannot establish
which historical bank produced a render. A clean rebuild is the way to
establish that dependency afresh. Cached filenames or matching recipe
fields alone are not proof of identical upstream contents.

## Artifacts

Fitted codebooks, taught bank weights, generated states and results
tables are written to `artifacts/`. Only the judge's critic is
committed; everything else is an output of a script here, and shipping
an output would let the script that recomputes it skip the step
instead.

The codecs and banks are what take hours, and they are large (the joint
bank alone is 3616 lanes over 232 spaces). `make artifacts` runs
experiments 00-05 and produces every one of them from scratch: about an
hour and a half for the deployed point, and about four hours in all once
`make sweep` adds the backbone codes (book count, alphabet size, keep-p)
the design studies compare. Once they
exist, `make all` finishes experiments 06 through 14 in about
twenty-five minutes; the binary arms, the comparator sweeps and the
head-to-head are hours more on top of that.

Every artifact file is config-keyed: a script that finds one matching
the configuration it asked for uses it, and refits otherwise. Delete a
file to force it to be rebuilt.

## Run order

Costs are wall-clock on eight CPU threads.

| step | what | cost |
|---|---|---|
| `00_train_critic.py` | the judge's frozen critic | 1 min |
| `01_fit_backbone_codec.py` | the backbone codec (`--all` for the 15-point sweep over books, alphabet size and keep-p) | 14 min / 70 min |
| `02_fit_patch_codec.py` | the residual patch codec, six (books per slot, keep-p) points | 5 min |
| `03_teach_backbone_banks.py` | G1 and R1 (`--all` for the sweep) | 18 min / 65 min |
| `04_teach_patch_banks.py` | G2 and R2 | 23 min |
| `05_teach_joint_bank.py` | the joint repair bank, pool included | 25 min |
| `06_backbone_sweep.py` | one-level generation across book count, alphabet size and keep-p | minutes |
| `07_verdict.py` | the two-level verdict at n = 1000 (single seed; Table II is the three-seed version from 21) | 1 min |
| `08_stack_arms.py` | what each component of the stack contributes | 1 min |
| `09_prefix_schedule.py` | the draw's one targeted ablation | 14 min |
| `10_draw_analysis.py` | where the draw deviates from real codes | 2 min |
| `11_operating_point.py` | the emulator's reads in physical units | 1 min |
| `12_energy_model.py` | joules per image, three accountings | 1 min |
| `13_energy_frontier.py` | the same per code size | 2 min |
| `14_figures.py` | Figs. 1, 3, 5, 6 | 1 min |
| `15_energy_figure.py` | Fig. 8 (needs the head-to-head) | seconds |
| `16_binarization_figure.py` | Fig. 7 (needs the head-to-head) | seconds |
| `17_fit_binary_codec.py` | the binary codec at threshold 0.1 | hours / seconds cached |
| `18_teach_binary_banks.py` | the binary banks, same recipe | hours / seconds cached |
| `19_binary_verdict.py` | binary screen, window draw, renders for scoring | minutes + 1 h render |
| `20_draw_ablations.py` | flat sigma and the temperature schedules, three seeds | 10 min |
| `21_reporting.py` | Table II's seed spread; memorization; the sweep's flip rate | 10 min |
| `22_binary_energy.py` | the energy model on the binary census | minutes |
| `23_comparator_sweep.py` | the comparator-noise sweep, grayscale (Fig. 4a; separate seeds from Table II) | 15 min |
| `24_binary_comparator_sweep.py` | the same sweep on the binary-trained stack (`--extend` for the last two points) | 20 min + scoring |
| `27_binary_comparator_headtohead.py` | binary one-level rows of Table VII at the operating point | 5 min |
| `28_physical_energy.py` | the energy model restated at the operating point (nothing moves) | 1 min |
| `29_comparator_figure.py` | Fig. 4 (`make comparator-figure`; reads 23's artifact and the scoring pass, so run it after `07_comparator_score.py`) | seconds |
| `31_free_draw.py` | the free-label draw for the audit of Sec. V C | 25 min |
| `35_cold_noise_control.py` | the cold sweeps at the physical floor against the noiseless argmax (Sec. IV C's cold-read control) | 6 min |
| `36_sneak_paths.py` | the unit crossbar's sneak network and the energy of each way out (Limitation 2) | seconds |
| `38_readout_options.py` | the readout priced five ways (continuous-time, clocked ramp, tournament) and the comparator's noise-energy bound (Appendix B) | seconds |
| `37_dose_chain.py` | the joint sweep chained to eight doses, with and without sweep read noise (`--quench`: noisy doses then one cold dose); Sec. III B's one-dose statement | 5 min each |

The binary arms are scored by `experiments/head_to_head/06_binary_arm.py`
and the operating-point rows by `07_comparator_score.py`, both in the
head-to-head's own environment. `head_to_head/25_comparator_headtohead.py`
renders the grayscale arms at the critic's operating point and
`head_to_head/34_grayscale_grid.py` renders them at every level of the
register grid (about an hour); `07_comparator_score.py` scores both and
prints each arm's best level, which Table VII, Fig. 7 and Fig. 8 use.

The comparator sweep is three steps in two environments, in this order:

```bash
make comparator                 # 23, 24, and 24 --extend
.venv-dtm/bin/python experiments/head_to_head/07_comparator_score.py
make comparator-figure          # 29, from the two results files
```

Fig. 4 is a separate target because it reads that scoring pass; run it
before the scoring and it draws the previous scoring's binary curve.

The head-to-head has its own sequence and its own environment; see
[`experiments/head_to_head/README.md`](experiments/head_to_head/README.md).

`make all` runs the grayscale main sequence, experiments 00–14, in
order. The rest are separate commands, because each costs hours or
needs artifacts `all` does not build: `make binary` (17–19, 22),
`make comparator` (23, 24), `make comparator-figure` (29),
`make physical` (25, 34, 27, 28),
`make audit` (31), `make cold-control` (35), `make ablations` (20),
`make reporting` (21), and
`make head-to-head-figures` (15, 16). The Makefile header lists them
in the order they need to run.

The numbering is one sequence across both directories: 25, 30, 32 and
33 sit in `experiments/head_to_head/` because they run in its
environment. There is no 26 — the slot was held for a second scorer
that proved unnecessary, since `07_comparator_score.py` scores every
physical-operating-point arm in a single pass.

## What the numbers should be

[`NUMBERS.md`](NUMBERS.md) lists every quantity the paper states, the
script that produces it, and the value to expect. It is the file to
check a run against.

The headline (Table II), three seeds at n = 1000, both arms with their
hot reads at the operating point (10 mV, comparator 2 mV), from
`21_reporting.py`:

```
arm               critic            div             tone   stdr
real              0.8860            0.4866          2.362  1.00
reconstruction    0.8963 (0.0110)   0.5184 (0.0015) 2.004  1.13
end-to-end        0.9007 (0.0310)   0.5487 (0.0052) 1.590  1.14
```

and from the head-to-head (Table VII), scored by the DTM replication
code against its own shipped reference at its own 0.1 threshold and
n = 5120, each arm at its best level on the register grid of Sec. IV C:

```
arm                        FID     div   J (nominal)
two-level                18.09  0.1558    2.00e-9
binary-trained two-level  9.69  0.1472    2.00e-9
their DTM, 8 steps       24.90       —    1.57e-08
```

## Determinism

Every stochastic step draws from an explicit seeded `torch.Generator`,
and the substrate's adapt is integer arithmetic. Reruns on the same
machine reproduce byte-identical bank weights, codes and generated
states; the verdict table above was checked that way against the run
the paper reports.

Across machines, floating-point codec fitting, encoding, judging and
FID can differ with the library build and hardware. Small score changes
can change an argmax and propagate through later computation. The seed
ranges in the paper do not establish cross-machine tolerances. Use the
saved evidence to inspect differences; byte identity is expected only
where it has actually been checked under the recorded environment.

## Layout

```
chill_zen/           the package: data, codecs, lane banks, AAT layouts,
                     teaching, generation, judge, energy model
experiments/         one script per measurement, in run order
experiments/head_to_head/
                     the comparison scored by the DTM replication code
reference/           the DTM paper's own Fig. 1 points, decoded from its
                     arXiv source
artifacts/           frozen codebooks, banks, states (see above)
figures/             output
logs/                output
data/                canonical Fashion-MNIST, fetched
```

## Known wrinkles

Everything here that is less than seamless, in one place.

**1. Two design choices were settled before this pipeline existed.**
Holding the label rather than masking it, and committing a single cold
repair dose, were fixed by ablation on a per-patch predecessor: a code
over a 4x4 grid of 7x7-pixel patches rather than the global-plus-residual
code this repository builds. Neither ablation's address tuple exists on
the deployed pipeline, so neither is re-runnable here. The paper states
both choices and their rationale and quotes no number from those runs.

**2. Codebooks do not rebuild byte-identically across machines.** See
*Determinism* above. Metrics agree to three or four decimals; the
acceptance test for a rebuild is [`NUMBERS.md`](NUMBERS.md), not a
checksum.

**3. `ktram-neural-core` is not on PyPI.** Install it from GitHub, as
the install section shows.

**4. The head-to-head is slow the first time and needs its own
environment.** About 3.5 hours of feature extraction across the gate
and the scoring pass, in a second virtualenv, because their FID code
needs JAX with numpy 2 while the emulator is pinned below it. Features
are cached, so a second run is minutes. Their evaluation code, their
reference statistics and the canonical Fashion-MNIST files are fetched
and checksum-verified rather than redistributed.

**5. Sec. IV D diagnoses the earlier fixed-gain control.** Experiment
10 prints that control beside the prefix-retaught draw. Neither column
is a measurement of the selected comparator-driven generator.

## What is not here

* The DTM replication code and the reference statistics it ships. They
  are fetched read-only and never edited or redistributed;
  `fetch_vendor.sh` records the commit and checks every checksum.
* The scripts of runs that were set aside. The one that matters is a
  binary-trained series invalidated by a soft-teach that silently
  applied hard feedback. `experiments/18_teach_binary_banks.py` states
  that lesson in its docstring, and its `feedback_semantics_check()`
  refuses to teach at all if the defect recurs — a live check rather
  than a record of an old one.

## Citing

If you use this, cite the paper. The evaluation pipeline is from `dtm-replication`; the published DTM
reference is Jelinčić et al. (arXiv:2510.23972v2); cite them for anything scored with it.

## License

MIT (see [`LICENSE`](LICENSE)) grants copyright. A separate [`PATENTS`](PATENTS) file reserves
Knowm's US hardware-patent rights: **software emulation is permitted with no patent license;
hardware realization in the US requires a separate license.** This matches `ktram-neural-core`.
