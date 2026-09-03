# Every number the paper states, and where it comes from

One row per claim: the value to expect, the script that prints it, and
what to run first. Values are from the runs the paper reports.

Reruns on one machine reproduce these exactly. Across machines, expect
codec-derived numbers (the `floor` and `combined` columns, and anything
downstream of a refitted codebook) to agree to three or four decimals
rather than exactly — see *Determinism* in the README.

---

## Setup

| claim | value | source |
|---|---|---|
| split | 68,000 train / 2,000 evaluation of the seed-0 permutation of all 70,000 images | `chill_zen/data.py` |
| deployed code | 128 x 16 backbone at keep-p 0.5 (512 bits) + 2 x 16 residual on 49 slots (392 bits) = 904 bits, plus the clamped label | `chill_zen/config.py` |
| judge critic | 784 -> 256 -> ReLU -> 10, raw held-out accuracy 0.8820 | `00_train_critic.py` |
| evaluation noise at n = 1000, measured over three seeds | critic spread (max − min) 0.0040 reconstruction / 0.0130 end-to-end | `21_reporting.py` |

## Sec. IV B — the two-level result (Table II, Fig. 3)

Table II carries the three-seed mean ± spread from `21_reporting.py`:

| arm | critic (spread) | div (spread) | tone | std-ratio |
|---|---|---|---|---|
| real references | 0.8860 | 0.4866 | 2.362 | 1.00 |
| reconstruction | **0.8877** (0.0040) | 0.5019 (0.0007) | 2.071 | 1.11 |
| end-to-end (physical, P3) | **0.9043** (0.0070) | 0.5179 (0.0039) | 1.929 | 1.15 |
| end-to-end (deployed T=0.1, superseded) | 0.8317 (0.0130) | 0.4715 (0.0067) | 2.505 | 1.25 |

Table II's end-to-end row is the physical row: `23_comparator_sweep.py`
P3 (`V = 10 mV, v_n = 1 mV`), three seeds. The fixed-gain `T = 0.1`
row is kept here as the control it superseded; no `T`-drawn number is
presented as a result in the paper.

The single-seed verdict, `07_verdict.py` after 01-05 (within the
spreads above; the recorded run behind Fig. 3 and the draw analysis):

| arm | critic | div | tone | seam | std-ratio |
|---|---|---|---|---|---|
| real references | 0.8860 | 0.4866 | 2.362 | 1.067 | 1.00 |
| reconstruction | **0.8860** | 0.5016 | 2.070 | 1.035 | 1.11 |
| end-to-end | 0.8340 | 0.4621 | 2.390 | 1.064 | 1.24 |

Memorization and the sweep's flip rate, `21_reporting.py`:

| claim | value | source |
|---|---|---|
| no memorization signature | 5120/5120 distinct joint AND backbone codes; exact copies 0; NN-to-train L2 q01/q50 generated 1.558/2.880 vs held-out real 1.465/3.433 | `21_reporting.py` |
| one dose is a reconciliation, not a fixed point | dose 2 flips 40.8/226 addresses per image (100% of images), dose 3 21.7, dose 4 13.5 | `21_reporting.py` |

`08_stack_arms.py`, n = 320, same batch across arms:

| arm | critic | div | tone | seam | std-ratio |
|---|---|---|---|---|---|
| real | 0.9031 | 0.4823 | 2.360 | 1.100 | 1.00 |
| backbone | 0.8344 | 0.5156 | 2.582 | 1.053 | 1.21 |
| patch level | 0.8375 | 0.5184 | 2.609 | 1.061 | 1.21 |
| joint | 0.8344 | **0.4771** | 2.733 | 1.051 | 1.24 |
| joint-recon | 0.9031 | 0.5005 | 2.095 | 1.062 | 1.11 |

| claim | value | source |
|---|---|---|
| the joint sweep moves diversity onto the real bar | 0.4771 against 0.4823 | `08_stack_arms.py` |
| doubling backbone capacity moves the end-to-end critic | 0.8062 -> 0.8344 | `08_stack_arms.py --backbone 64x16@p0.5`, then the default |

## Sec. IV D — where the remaining error is (localization)

`07_verdict.py` and `10_draw_analysis.py`.

| claim | value | source |
|---|---|---|
| reconstruction critic matches the real bar within seed spread | 0.8877 ± 0.0040 against 0.8860 | `21_reporting.py` |
| the deployed T=0.1 end-to-end gap (superseded) | 0.8317 against 0.8860 | `21_reporting.py` |
| the physical end-to-end draw exceeds both | 0.9043 ± 0.0070 against reconstruction 0.8877 and real 0.8860 | `23_comparator_sweep.py` (P3) |
| foreground contrast, end-to-end (deployed) / reconstruction | 1.25 / 1.11 | `21_reporting.py` |
| foreground contrast, end-to-end (physical, P3) | 1.15 | `23_comparator_sweep.py` |
| backbone decode, drawn codes / real codes | 1.198 to 1.226 / 1.039 | `10_draw_analysis.py`, probe 5 |
| the patch level adds on top | about 0.01 | `10_draw_analysis.py`, probe 5 |

## Sec. IV D — where the remaining error is (the draw's signature)

`10_draw_analysis.py` (deployed draw from 07, prefix-schedule draw from
09) and `09_prefix_schedule.py`. The paper quotes the
prefix-schedule column for the distributional probes.

| claim | deployed | prefix-schedule | bar |
|---|---|---|---|
| per-book marginal TV against train | 0.2138 | **0.2032** | 0.0454 |
| worst book | 0.3409 at book 2 | **0.4029 at book 5** | — |
| pair-joint TV, adjacent | 0.4015 | **0.3693** | 0.1622 |
| pair-joint TV, offset-7 | 0.4021 | 0.3690 | 0.1641 |
| re-encode self-consistency, overall | 0.8070 | **0.7794** | real codes 0.6888 |
| re-encode, first half of the chain | 0.8347 | **0.8060** | real codes 0.7143 |
| correlation of over-pick with atom contrast | -0.004 | -0.002 | — |

| claim | value | source |
|---|---|---|
| the over-picked symbols are near-zero-contrast atoms in the first books | top over-picks at contrast quantile 0.00 in books 0, 1, 2 | `10_draw_analysis.py`, probe 2 |
| reteaching the draw bank on its run-time schedule moves the critic | 0.8340 -> 0.8300, i.e. -0.004 | `09_prefix_schedule.py` |
| that reteach did work | next-book accuracy 0.3720 against a class-majority prior of 0.2550; Bernoulli fill holds at 0.2954 against 0.3065 | `09_prefix_schedule.py` |

**The draw ablations** — `20_draw_ablations.py`, three seeds per arm at
n = 1000, real bar critic 0.8860 / div 0.4866. Mean over seeds; the
parenthesis is the critic's seed spread (max − min). The flat-sigma
row is quoted in Sec. V A (the calibration scope); the schedule rows
close Sec. IV D.

| arm | critic (spread) | div | tone | std-ratio |
|---|---|---|---|---|
| deployed | 0.8350 (0.013) | 0.4671 | 2.498 | 1.25 |
| flat sigma, matched median | 0.8417 (0.009) | 0.4384 | 2.403 | 1.28 |
| schedule S-lin, T 0.20 -> 0.05 | 0.8253 (0.022) | 0.5000 | 1.983 | 1.15 |
| schedule S-eff, constant effective noise | 0.8330 (0.023) | 0.4197 | 2.209 | 1.32 |

## Sec. IV E — design studies

The held label and the single repair dose were settled by ablation on a
per-patch predecessor of this pipeline. Those runs are not reproducible
from `experiments/` and no number from them appears in the paper; the
paper states the two choices and their rationale (Sec. III B, Sec. IV A)
without quoting a value. Nothing below depends on them.

**(a) Capacity and alphabet scaling** — `06_backbone_sweep.py --all`,
after `01 --all` and `03 --all`.

| code shape | bits | critic | div | tone | seam |
|---|---|---|---|---|---|
| 16 x 16 @ p0.5 | 64 | 0.7406 | 0.4970 | 1.639 | 0.914 |
| 32 x 16 @ p0.5 | 128 | 0.7969 | 0.5165 | 1.748 | 0.969 |
| 64 x 16 @ p0.5 | 256 | 0.7969 | 0.5392 | 2.144 | 1.039 |
| 128 x 16 @ p0.5 | 512 | **0.8344** | 0.5156 | 2.582 | 1.053 |
| 64 x 32 @ p0.5 | 320 | **0.7969** | 0.5415 | 1.897 | 1.034 |
| 64 x 16 @ p1.0 | 256 | 0.7406 | 0.6233 | 1.570 | 1.072 |

| claim | value | source |
|---|---|---|
| deepening the alphabet at matched bits measures worse | 64x32 at 320 bits reaches 0.7969; 128x16 at 512 bits reaches 0.8344 | `06_backbone_sweep.py` |
| a global code removes tile seams by construction | 0.91 to 1.07 across the sweep, real 1.100 | `06_backbone_sweep.py` |
| the per-patch comparison bar | 1.768 | predecessor pipeline, quoted as an anchor; not reproducible here |

**(b) Codec rate-distortion** — `01_fit_backbone_codec.py --all` and
`02_fit_patch_codec.py`.

| claim | value | source |
|---|---|---|
| the global code at 512 bits, keep-p 0.75 | relMSE 0.0500 | `01`, `128x16@p0.75` |
| a per-patch code at 512 bits | relMSE 0.0576 | predecessor pipeline, quoted as an anchor; not reproducible here |
| the deployed backbone alone (keep-p 0.5) | relMSE 0.0612 | `01`, `128x16@p0.5` |
| the two-level deployed point at 904 bits | relMSE 0.0382 | `02`, `2x16@p0.5` |
| plain additive quantization degrades with scale | 128x16@p1.0 floor 0.0902 with 56 dead atoms, against 0.0612 and none at p0.5 | `01` |
| keep-p 0.5 concedes floor for coherence over the bar | 128 books: 0.0612 against 0.0500 (+22%), +0.057 against +0.016 (3.6x). 64 books: +17%, 2.9x. 32 books: +14%, 2.3x. 16 books: +12%, 2.2x. The paper's "roughly 20% for three to five times" is the 128-book row. | `01` |

## Sec. III B — teaching (no figure; the numbers behind the banks)

| claim | value | source |
|---|---|---|
| G1 fill on held-out holes | 0.3065 against a 0.1358 prior, lift +0.1708 | `03_teach_backbone_banks.py` |
| R1 repair / preserve | 0.2648 / 0.6198, pool wrong 0.542 | `03_teach_backbone_banks.py` |
| G2 fill | 0.3959 against a 0.3121 prior, gap probe 0.3562 | `04_teach_patch_banks.py` |
| R2 repair / preserve | 0.3469 / 0.5829, pool wrong 0.467 | `04_teach_patch_banks.py` |
| joint repair, overall / backbone / patch | 0.2985 / 0.3139 / 0.2768, eval pool wrong 0.528 | `05_teach_joint_bank.py` |
| the joint bank beats the bank it replaces on backbone addresses, at a matched 64-book configuration | 0.372 against 0.287 | `05 --backbone 64x16@p0.5` and `03 --points 64x16@p0.5` |

## Sec. V B — the energy model (Table V)

`11_operating_point.py`, `12_energy_model.py`, `13_energy_frontier.py`.
All of it is a physical model, never a measurement.

| claim | value | source |
|---|---|---|
| the emulator's default noise gain implies a Johnson source of | 1.2 MOhm at the reference read | `11` |
| draw noise, median over deployed contexts | 0.0294 backbone (0.147 at step 0 to 0.016 at step 127), 0.0165 patch | `11` |
| cold verdict margin over the physical flicker floor | 0.0967 against 2.09e-03, a factor of 46 | `11` |
| settling, 5RC across the band | 0.093 to 1.87 ns at 100 fF | `11` |
| the thermodynamic floor at the draw noise | 2.3e3 kBT backbone, 7.4e3 kBT patch | `11` |
| census per image | 2048 + 1568 + 2048 + 3616 + 784 (decode) = 10,064 lane reads over 226 addresses | `12` |
| the decode's share | 784 cold reads, one parallel group read; +1.8e-12 J with its comparators, moving the bounded figure 1.69e-11 -> 1.87e-11 and the headline ratio 926 -> 838 | `12` |
| device-level bounded total | 8.24e-12 J (SnCr) / 5.80e-11 J (W) | `12` |
| of the device total, cold reads at 5 C V^2 | 8.06e-12 J (6448 cold reads) | `12` |
| periphery, comparator-dominated | 1.05e-11 J, of which 1.01e-11 is comparators | `12` |
| with periphery, per image | 1.87e-11 J bounded to 1.58e-09 J at the SnCr anchor | `12` |
| the same on the tungsten band | bounded 6.85e-11 J (229x below the DTM chain), anchor 3.37e-08 J | `12` |
| per drawn bit | 4.52e-16 J bounded (1.1e5 kBT), 8.93e-17 J at the floor (2.2e4 kBT) | `12` |
| their RNG cell, for comparison | 2e-15 J (4.9e5 kBT) per Bernoulli sample | their Fig. 12b |

Per-image energy across the frontier, `13`:

| system | bounded (SnCr) | bounded (W) | SnCr anchor | W anchor |
|---|---|---|---|---|
| 16 books | 8.65e-13 | 8.65e-13 | 5.81e-12 | 1.20e-10 |
| 32 books | 1.73e-12 | 1.73e-12 | 2.12e-11 | 4.49e-10 |
| 64 books | 3.47e-12 | 4.68e-12 | 7.97e-11 | 1.72e-09 |
| 128 books | 6.97e-12 | 1.47e-11 | 3.28e-10 | 7.08e-09 |
| deployed two-level | 1.87e-11 | 6.85e-11 | 1.58e-09 | 3.37e-08 |

The one-level rows are device-level bounded totals with no decode row;
the deployed row carries the full census (decode and periphery
included).

**Physical-operating-point reconciliation, `28_physical_energy.py`.**
Confirms every bounded figure above is unchanged: the earlier code
already clamped the operating voltage to `V_FLOOR` whenever the
emulator-`T`-derived target wanted a sub-floor read, which is every
hot-read class reported here, so `e_bound = V_FLOOR^2 G_sum t_b`
either way (ratio to the old value 1.0000 at every class checked).

| claim | value | source |
|---|---|---|
| grayscale composite sigma at the physical point (P3, v_n=1mV) | 0.1063 (ctx 0) to 0.1014 (patch), matching the sweep's own composition to three digits | `28_physical_energy.py` |
| lane-only sigma (no comparator) | 0.017-0.036 across the chain | `28_physical_energy.py` |
| binary composite sigma at the physical point (P6, v_n=3mV) | 0.3014 (ctx 0) to 0.3005 (ctx 127), matching the binary sweep's own composition exactly | `28_physical_energy.py` |
| reference-read row (P5 twin, V=50mV, v_n=5mV) | lane 0.0050, comparator 0.1000 -- comparator-dominated, matches P5's measured 97-100% share | `28_physical_energy.py` |

## Sec. IV C — the comparator sets the draw temperature (Fig. 4)

The grid, the per-point noise composition (thermal / flicker /
comparator variance shares) and both curves are printed by
`23_comparator_sweep.py` and `24_binary_comparator_sweep.py`
(`--extend` adds the last three binary points); the argument is
Sec. IV C of the paper and the read model is `chill_zen/physical.py`.

| $v_n/V$ | grayscale end-to-end critic | binary-trained two-level FID |
|---|---|---|
| 0 (P0) | 0.8170 | 22.65 |
| 0.015 (P1) | 0.8480 | 21.67 |
| 0.05 (P2) | 0.8847 | 16.39 |
| **0.10 (P3)** | **0.9043** | 12.99 |
| 0.20 (P4) | 0.8977 | 11.23 |
| **0.30 (P6)** | -- | **11.10** |
| 0.50 (P7) | -- | 11.57 |
| 0.75 (P8) | -- | 12.27 |
| deployed (emulator $T$) | 0.8350 | 9.82 (recorded 9.82, reproduced 9.88 on a different seed block) |

| claim | value | source |
|---|---|---|
| P3 = P5 (same v_n/V, 5x different V) | grayscale critic 0.9043 vs 0.9030; binary FID 12.99 vs 12.39 -- the comparator's share sets the temperature, not V | `23_comparator_sweep.py`, `24_binary_comparator_sweep.py` |
| the binary turnover basin | P4-P6 differ by 0.13 FID, about the seed-block scatter on the deployed arm (9.88 vs recorded 9.82) | `24_binary_comparator_sweep.py --extend` |
| comparator classes modeled against | raw dynamic latch 0.5-2mV rms / 5-20mV offset; auto-zeroed 50-200uV / 0.1-1mV; digitally trimmed tens of uV / tens of uV | Table III of the paper; sources Razavi 2015, Pelgrom 1989 |
| T_max, lanes alone at the 10mV floor | ~0.025 empty context, ~0.05 full context | `23_comparator_sweep.py` (the `v_n = 0` point's composition) |
| the median-convention note | grayscale backbone read noise 0.023 (median over steps) vs 0.030 in the energy log (mean of per-step medians) | `11_operating_point.py` against `23_comparator_sweep.py` |

## Sec. V C — the head-to-head (Tables VI and VII)

`experiments/head_to_head/`, its own environment. The gate is
`01_gate.py` and nothing downstream counts unless it passes.

| claim | value | source |
|---|---|---|
| threshold probe picks 0.1 | relative L2 0.01407 against 0.11370 (0.3), 0.22605 (0.5), 0.24048 (0.0) — 8.1x | `01_gate.py` |
| FID(60,000 train at 0.1, their shipped statistics) | 0.000039 | `01_gate.py` |
| mu / sigma maximum absolute difference | 6.0e-04 / 3.0e-04 | `01_gate.py` |
| held-out floor at n = 5120 / 10,000 | 2.142 / 1.194 | `01_gate.py` |
| the same 60,000 train images at the midpoint | 29.75 | `01_gate.py` |

`03_score.py` and `05_census.py`, their code, their reference, their
0.1, n = 5120:

| system | FID | div | J/image bounded |
|---|---|---|---|
| real train images (control) | 1.907 | 0.1806 | — |
| 128-book codec ceiling | 14.205 | 0.1634 | — |
| **two-level (deployed)** | **18.931** | 0.1540 | 1.87e-11 |
| one-level, 128 books | 22.320 | 0.1583 | 6.97e-12 |
| one-level, 64 books | 23.867 | 0.1512 | 3.47e-12 |
| one-level, 32 books | 28.752 | 0.1420 | 1.73e-12 |
| one-level, 16 books | 46.218 | 0.1262 | 8.65e-13 |
| DTM, 8 steps | 24.90 | — | 1.568e-08 |
| DTM, 2 steps | 32.60 | — | 3.921e-09 |

Table above is superseded (no T-drawn number stands as a
result). Table VII carries the physical-point row
(`25_comparator_headtohead.py`, P3 = 10 mV/1 mV; scored by
`07_comparator_score.py`, in the DTM environment):

| system | FID | div | J/image bounded |
|---|---|---|---|
| **two-level (physical, P3)** | **18.73** | 0.1519 | 1.87e-11 |
| one-level, 128 books | 25.68 | 0.1390 | 6.97e-12 |
| one-level, 64 books | 26.61 | 0.1286 | 3.47e-12 |
| one-level, 32 books | 32.81 | 0.1256 | 1.73e-12 |
| one-level, 16 books | 44.89 | 0.1156 | 8.65e-13 |

**Flagged, not softened:** at the physical point only the two-level
arm clears the DTM bar (24.90); every one-level row now sits above
it. P3 was found by sweeping the
two-level stack; applied to the one-level frontier (a different
architecture, no patch decoration) it costs FID rather than saving it.
No per-architecture sweep was run (compute economy, out of the
declared scope of that run). Energies are unchanged: the bounded
figures do not depend on draw quality.

| claim | value | source |
|---|---|---|
| energy ratio, bounded | 838x less | `12_energy_model.py` |
| energy ratio, tungsten bounded | 229x less | `12_energy_model.py` |
| energy ratio, SnCr anchor | 9.9x less | `12_energy_model.py` |
| energy ratio, tungsten anchor | **2.2x more** | `12_energy_model.py` |
| the deployed draw sits above its own codec ceiling by | 4.7 FID | `03_score.py` |
| diversity against real | 0.154 against 0.181, 85%; the codec's own render of real codes is 0.163 | `03_score.py` |
| their GPU baselines | VAE best 17.9, GAN best 26.3, MEBM best 33.5; DDPM best decoded 12.2 (clipped by their y-axis; best visible 16.4) | `reference/dtm_figure1_data.py` |

**The binary-trained arms** — fit by `17_fit_binary_codec.py`, taught
by `18_teach_binary_banks.py`, screened and rendered by
`19_binary_verdict.py`, scored by `head_to_head/06_binary_arm.py`,
energy census by `22_binary_energy.py`. The temperature sweep
{0.1, 0.2, 0.3, 0.5} was declared before the run; Table VII carries the
best-T rows.

| system | FID | div | J/image bounded |
|---|---|---|---|
| binary codec ceiling, 128 books | 9.566 | 0.1675 | — |
| binary stack ceiling (real codes, both levels) | 2.832 | 0.1757 | — |
| binary real-train control | 1.907 | 0.1806 | — |
| binary two-level, T = 0.3 (superseded) | 9.824 | 0.1577 | 1.87e-11 (with periphery) |
| binary two-level, T = 0.5 / 0.2 / 0.1 (superseded) | 9.930 / 10.878 / 16.807 | 0.158 / 0.148 / 0.130 | — |
| binary one-level 128, T = 0.2 (superseded) | 12.469 | 0.1407 | 3.64e-12 (device, decode incl.) |
| binary one-level 64, T = 0.2 (superseded) | 13.622 | 0.1395 | 2.31e-12 (device, decode incl.) |

Physical-point rows (P6 for two-level and for one-level; both
10 mV/3 mV, `v_n/V = 0.30`), the ones Table VII carries:

| system | FID | div | J/image bounded |
|---|---|---|---|
| **binary two-level (physical, P6)** | **11.10** | 0.1423 | 1.87e-11 (with periphery) |
| binary one-level 128 (physical, P6) | 16.53 | 0.1241 | 3.64e-12 (device, decode incl.) |
| binary one-level 64 (physical, P6) | 17.21 | 0.1155 | 2.31e-12 (device, decode incl.) |

**Flagged, not softened:** the one-level rows get worse at the
physical point too (12.47 -> 16.53, 13.62 -> 17.21), the same
direction as the grayscale frontier and for the same reason: P6
was found by sweeping the two-level stack. Both one-level rows still
clear the DTM bar (24.90) by a wide margin; they are simply farther
from it than the deployed numbers implied.

| claim | value | source |
|---|---|---|
| the codec fit reproduces the recorded binary floors | 128: disagree-ev 0.0169 / relMSE-ev 0.0788; 64: 0.0266/0.1046; two-level 0.0068/0.0464; dead 0 | `17_fit_binary_codec.py` |
| teach checks | G lift +0.2213 (128) / +0.2560 (64), patch G2 +0.1712; R repair 0.2926 / 0.3344, R2 0.4274, joint 0.3499 | `18_teach_binary_banks.py` |
| T = 0.1 collapses binary diversity at the screen | one-level div 0.1296 against the 0.75 x 0.1811 rule (real bar critic 0.8680, div 0.1811) | `19_binary_verdict.py` |
| the window draw screens out | div 0.1025 against a 0.1358 bar, critic 0.7890; T_max median 0.014 (ctx 0) to 0.043 (ctx 127) | `19_binary_verdict.py` |
| two-level beats one-level | 9.82 against 12.47, 2.7 FID | `06_binary_arm.py` |
| the two-level arm sits above its own codec ceiling by | 0.25 FID (9.82 against 9.57) | `06_binary_arm.py` |
| binary two-level energy matches grayscale post-decode | 1.871e-11 J with periphery, within 1% of 1.87e-11 | `22_binary_energy.py` |
| the one-level energy flag | SnCr device bounded 3.64e-12 (128) / 2.31e-12 (64), about 2.5x below the grayscale 6.97e-12 / 3.47e-12 on like terms; unexplained, stated in Sec. V B | `22_binary_energy.py` |

The DTM comparison points are decoded from the vector figure in that
paper's arXiv source; of the decoded series only the VAE points can be
cross-checked (its Appendix F Table III, matched to the two
significant figures that column prints). The GAN/VAE/MEBM/DDPM models
were trained and scored by them, not re-run here; see
`reference/dtm_figure1_data.py` and the provenance paragraph of
Sec. V C.

**The head-to-head audit** — scripts
`head_to_head/30_bernoulli_baseline.py`, `31_free_draw.py`,
`head_to_head/32_audit_score.py`, `head_to_head/33_binary_memorization.py`.

| claim | value | source |
|---|---|---|
| replication clamped-vs-free, all 44 shipped runs | clamped best < free best in 44/44, median gap 3.0 FID; 8-step DTM-ACP 26.84 free / 25.82 clamped | `vendor/dtm-replication/figures/*.csv` |
| free-label draw, binary two-level P6 | 11.05 (against 11.10 clamped) | `31_free_draw.py`, `32_audit_score.py` |
| free-label draw, grayscale two-level P3 | 18.32 (against 18.73 clamped) | `31_free_draw.py`, `32_audit_score.py` |
| Bernoulli calibration, class-conditional / pooled | 180.51 / 258.10 | `30_bernoulli_baseline.py`, `32_audit_score.py` |
| binary memorization, two-level P6 | 5120/5120 distinct, 1 exact train copy (held-out real: 13/10,000), NN-Hamming median 24 bits against held-out real 23 | `33_binary_memorization.py` |
| binary memorization, one-level 128 / 64 | 5120/5120 distinct, 0 exact copies each | `33_binary_memorization.py` |

## Figures

Numbers are the compiled order (labels in parentheses are the stable
reference); Fig. 2 (`fig:arch`) and Fig. 7 (`fig:contrast`) are drawn
in TeX and have no script. `14_figures.py` also writes `fig-teach.png`,
which the paper no longer includes.

| figure | script | needs |
|---|---|---|
| Fig. 1 `fig-noise.png` (`fig:noise`) | `14_figures.py` | 11 |
| Fig. 3 `fig-samples.png` (`fig:samples`) | `14_figures.py` | 07 |
| Fig. 4 `fig-comparator-sweep.png` (`fig:comparatorsweep`) | `29_comparator_figure.py` | 23, 24 (+ `--extend`) |
| Fig. 5 `fig-backbone.png` (`fig:backbone`) | `14_figures.py` | 06 `--all` |
| Fig. 6 `fig-codec.png` (`fig:codec`) | `14_figures.py` | 01 `--all`, 02 |
| Fig. 8 `fig-binarization.png` (`fig:binarization`) | `16_binarization_figure.py` | the head-to-head at the operating point (25, 06) |
| Fig. 9 `fig-energy.png` (`fig:energy`) | `15_energy_figure.py` | 13 and the head-to-head |
