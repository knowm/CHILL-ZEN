# Every number the paper states, and where it comes from

One row per claim: the value to expect, the script that prints it, and
what to run first. Values are from the runs the paper reports.

The seed ranges describe generation variation of selected banks, not
confidence intervals or cross-machine tolerances. Historical controls
below are marked separately from final comparator-setting results.
Saved logs are preserved in `evidence/`; absent fill-sweep logs are not
reconstructed from the summary.

---

## Setup

| claim | value | source |
|---|---|---|
| split | 68,000 train / 2,000 evaluation of the seed-0 permutation of all 70,000 images | `chill_zen/data.py` |
| deployed code | 128 x 16 backbone at keep-p 0.5 (512 bits) + 2 x 16 residual on 49 slots (392 bits) = 904 bits, plus the clamped label | `chill_zen/config.py` |
| judge critic | 784 -> 256 -> ReLU -> 10, raw held-out accuracy 0.8820 | `00_train_critic.py` |
| evaluation noise at n = 1000, measured over three seeds | critic spread (max − min) 0.0110 reconstruction / 0.0310 end-to-end (`21`, both arms at P4); 0.0180 on the P4 row of `23` | `21_reporting.py`, `23_comparator_sweep.py` |

## Sec. IV B — the two-level result (Table II, Fig. 3)

Table II carries the three-seed mean (range) from `21_reporting.py`, both arms with their hot reads at the operating point (P4, 10 mV / 2 mV), seeds 960-962:

| arm | critic (spread) | div (spread) | tone | std-ratio |
|---|---|---|---|---|
| real references | 0.8860 | 0.4866 | 2.362 | 1.00 |
| reconstruction | **0.8963** (0.0110) | 0.5184 (0.0015) | 2.004 | 1.13 |
| end-to-end (P4) | **0.9007** (0.0310) | 0.5487 (0.0052) | 1.590 | 1.14 |
| end-to-end (P4, `23_comparator_sweep.py`, seeds 950-952) | 0.8900 (0.0180) | 0.5479 (0.0108) | 1.510 | 1.14 |
| reconstruction, fixed-gain patch read (superseded 2026-09-06) | 0.8837 (0.0070) | 0.5055 (0.0006) | 2.023 | 1.12 |
| end-to-end (fixed-gain T=0.1, superseded) | 0.8297 (0.0160) | 0.4830 (0.0053) | 2.489 | 1.26 |

Both Table II rows come from `21_reporting.py` with every hot read at
P4 (`V = 10 mV, v_n = 2 mV`, register code 95); the `23` row is the
same draw on other seeds and agrees within spread. The two superseded
rows are kept as the controls they were: the reconstruction arm's
patch read went through the fixed-gain emulator read until
2026-09-06, and the fixed-gain `T = 0.1` end-to-end draw is the
control of Sec. IV C. No fixed-gain number is presented as a result in
the paper.

The historical fixed-gain single-seed verdict, `07_verdict.py` after
01-05 (used for the draw analysis; no longer the source of Fig. 3):

| arm | critic | div | tone | seam | std-ratio |
|---|---|---|---|---|---|
| real references | 0.8860 | 0.4866 | 2.362 | 1.067 | 1.00 |
| reconstruction | **0.8840** | 0.5053 | 2.020 | 1.045 | 1.12 |
| end-to-end | 0.8340 | 0.4621 | 2.390 | 1.064 | 1.24 |

Memorization and the sweep's flip rate, `21_reporting.py`:

| claim | value | source |
|---|---|---|
| no memorization signature | 5120/5120 distinct joint AND backbone codes; exact copies 0; NN-to-train L2 q01/q50 generated 1.859/3.704 vs held-out real 1.465/3.433 | `21_reporting.py` |
| one dose is a reconciliation, not a fixed point | dose 2 flips 82.1/226 addresses per image (100% of images), dose 3 43.9, dose 4 25.6 (99.96%) | `21_reporting.py` |

The reconstruction arm bypasses both the autoregressive draw and R1.

**The sweep as a chain, `37_dose_chain.py`** (Sec. III B, "runs once by
design"): the joint sweep chained to eight doses on Table II's hot draws
(seeds 960-962, n = 1000), under six sweep read conditions; `--quench`
follows each noisy dose with one cold dose.

| claim | value | source |
|---|---|---|
| noiseless chain, end-to-end critic at dose 1 / 4 / 8 | 0.9007 / 0.849 / 0.813 (paired; every seed falls) | `37` |
| noiseless chain, div and tone at dose 1 -> 8 | div 0.549 -> 0.443 (real 0.487); tone 1.59 -> 1.41 | `37` |
| not a fixed point at dose 8 | 7.1 flips/226 per image; 8% of images unchanged by dose 8 | `37` |
| real encoded codes drift the same way | reconstruction arm 0.892 -> 0.813 over eight cold doses | `37` |
| sweep noise at the floor (50 mV, code 0 / code 45) | same curve as noiseless within 0.01 | `37` |
| sweep noise at v_n/V = 0.10 (50 mV code 245 or 10 mV code 45) | dose 1 0.807, then a stationary ~0.85 with ~150/226 flips per dose | `37` |
| sweep at the draw's operating point (10 mV, code 95, v_n/V = 0.20) | washes out: critic 0.26, std-ratio 0.78, both arms | `37` |
| quench: k doses at 0.10 then one cold dose | critic flat 0.89-0.91 for k = 1..8; div 0.518 -> 0.442, tone 1.56 -> 1.17 | `37 --quench` |
| the one recipe that edges one cold dose (not adopted) | one dose at 0.10 then one cold: 0.909 vs 0.901 (+0.009 / +0.018 / -0.001 by seed), div 0.518 vs 0.549 | `37 --quench` |

`08_stack_arms.py`, n = 320, same batch across arms:

| arm | critic | div | tone | seam | std-ratio |
|---|---|---|---|---|---|
| real | 0.9031 | 0.4823 | 2.360 | 1.100 | 1.00 |
| backbone | 0.8344 | 0.5223 | 2.585 | 1.068 | 1.21 |
| patch level | 0.8250 | 0.5250 | 2.621 | 1.078 | 1.22 |
| joint | 0.8250 | **0.4910** | 2.678 | 1.077 | 1.24 |
| joint-recon | 0.9031 | 0.5045 | 2.080 | 1.066 | 1.12 |

| claim | value | source |
|---|---|---|
| the joint sweep moves diversity toward the real bar | 0.4910 against 0.4823, from 0.5250 before the sweep | `08_stack_arms.py` |
| doubling backbone capacity moves the backbone-only critic | 0.8125 -> 0.8344 (64 -> 128 books at keep-p 0.5); the two-level stack on a 64-book backbone (`08 --backbone 64x16@p0.5`, 0.8062 -> 0.8344 on the dial-taught banks) was not re-measured after the retrain | `06_backbone_sweep.py --all` |

## Sec. IV D — where the remaining error is (localization)

`07_verdict.py` and `10_draw_analysis.py`.

| claim | value | source |
|---|---|---|
| reconstruction critic matches the real bar within seed spread | 0.8963 ± 0.0110 against 0.8860 | `21_reporting.py` |
| the fixed-gain control's end-to-end gap | 0.8273 ± 0.0120 against 0.8860 | `23_comparator_sweep.py` (deployed control) |
| the physical end-to-end draw exceeds both, inside its own spread | 0.9007 ± 0.0310 against reconstruction 0.8963 and real 0.8860 | `21_reporting.py` |
| foreground contrast, fixed-gain control / reconstruction | 1.26 / 1.13 | `23_comparator_sweep.py`, `21_reporting.py` |
| foreground contrast, end-to-end (physical, P4) | 1.14 | `23_comparator_sweep.py` |
| backbone decode, drawn codes / real codes | 1.209 to 1.237 / 1.039 | `10_draw_analysis.py`, probe 5 |
| the patch level adds on top | about 0.01 | `10_draw_analysis.py`, probe 5 |

## Sec. IV D — controls and the earlier draw diagnosis

`10_draw_analysis.py` (deployed draw from 07, prefix-schedule draw from
09) and `09_prefix_schedule.py`. The paper quotes the
deployed (control) column for the distributional probes (pass 9; it quoted the prefix-schedule column before 2026-09-06).

| claim | deployed | prefix-schedule | bar |
|---|---|---|---|
| per-book marginal TV against train | 0.2058 | **0.2106** | 0.0454 |
| worst book | 0.3318 at book 126 | **0.3755 at book 5** | — |
| pair-joint TV, adjacent | 0.3966 | **0.3798** | 0.1622 |
| pair-joint TV, offset-7 | 0.3989 | 0.3820 | 0.1641 |
| re-encode self-consistency, overall | 0.7976 | **0.7742** | real codes 0.6888 |
| re-encode, first half of the chain | 0.8312 | **0.8058** | real codes 0.7143 |
| correlation of over-pick with atom contrast | +0.048 | +0.047 | — |

| claim | value | source |
|---|---|---|
| the over-picked symbols are near-zero-contrast atoms in the first books | top over-picks at contrast quantile 0.00 in books 0, 1, 2 | `10_draw_analysis.py`, probe 2 |
| reteaching the draw bank on its run-time schedule moves the critic | 0.8340 -> 0.8330, i.e. -0.001 (`logs/09_prefix_schedule.log` line 24) | `09_prefix_schedule.py` |
| that reteach did work | next-book accuracy 0.3720 against a class-majority prior of 0.2550; Bernoulli fill holds at 0.2954 against 0.3065 | `09_prefix_schedule.py` |

**The draw ablations** — `20_draw_ablations.py`, three seeds per arm at
n = 1000, real bar critic 0.8860 / div 0.4866. Mean over seeds; the
parenthesis is the critic's seed spread (max − min). The flat-sigma
row is quoted in Sec. V A (the calibration scope); the schedule rows
close Sec. IV D.

| arm | critic (spread) | div | tone | std-ratio |
|---|---|---|---|---|
| deployed | 0.8273 (0.012) | 0.4803 | 2.487 | 1.26 |
| flat sigma, matched median | 0.8393 (0.004) | 0.4520 | 2.424 | 1.30 |
| schedule S-lin, T 0.20 -> 0.05 | 0.8200 (0.018) | 0.5154 | 1.920 | 1.17 |
| schedule S-eff, constant effective noise | 0.8263 (0.019) | 0.4370 | 2.199 | 1.33 |

## Sec. IV E — design studies

The held label and the single repair dose were settled by ablation on a
per-patch predecessor of this pipeline. Those runs are not reproducible
from `experiments/` and no number from them appears in the paper; the
paper states the two choices and their rationale (Sec. III B, Sec. IV A)
without quoting a value. Nothing below depends on them.

**(a) Capacity and alphabet scaling** — `06_backbone_sweep.py --all`,
after `01 --all` and `03 --all`.

| books x symbols @ keep-p | bits | critic | div | tone | seam |
|---|---|---|---|---|---|
| 16 x 16 @ p0.5 | 64 | 0.7375 | 0.4997 | 1.656 | 0.929 |
| 32 x 16 @ p0.5 | 128 | 0.7937 | 0.5073 | 1.767 | 0.971 |
| 64 x 16 @ p0.5 | 256 | 0.8125 | 0.5408 | 2.160 | 1.028 |
| 128 x 16 @ p0.5 | 512 | **0.8344** | 0.5223 | 2.585 | 1.068 |
| 64 x 32 @ p0.5 | 320 | **0.7875** | 0.5538 | 1.927 | 1.032 |
| 64 x 16 @ p1.0 | 256 | 0.7437 | 0.6220 | 1.567 | 1.068 |
| 16 / 32 / 64 / 128 x 16 @ p0.75 (Fig. 5b only) | 64-512 | 0.7406 / 0.8188 / 0.8438 / 0.8031 | 0.551 / 0.543 / 0.556 / 0.543 | — | 1.005 / 0.968 / 1.028 / 1.013 |

| claim | value | source |
|---|---|---|
| deepening the alphabet at matched bits measures worse | 64x32 at 320 bits reaches 0.7875; 128x16 at 512 bits reaches 0.8344 | `06_backbone_sweep.py` |
| a global code removes tile seams by construction | 0.93 to 1.07 across the sweep, real 1.100 | `06_backbone_sweep.py` |
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

## Sec. IV A — teaching (no figure; the numbers behind the banks)

The G fills that the R banks and the joint bank teach against are drawn
through `chill_zen.physical` at the read the system deploys: `read_noise`
0.02 (the device), `V = 10 mV` (the sense floor), and the comparator's
trim register supplying the temperature at **code 245** (`v_cmp = 5 mV`,
`levels.FILL_V_CMP`; `CHILLZEN_FILL_VCMP` overrides it). Every bank's
config records the fill read as `fill_read`, so banks taught at another
level never match and are never silently reused. The G banks do not
teach against fills and reproduce bit-identically across fill levels.

Code 245 was chosen during development and carried to the binary stack.
Intermediate fill-sweep logs were not retained; the compared runs also
used different generation-noise settings. The former summary is not
an auditable isolated sweep and its intermediate scores are omitted.
The final recipe and final-bank probes are reproducible below.

| claim | value | source |
|---|---|---|
| G1 fill on held-out holes | 0.3065 against a 0.1358 prior, lift +0.1708 | `03_teach_backbone_banks.py` |
| R1 repair / preserve | 0.2693 / 0.5980, pool wrong 0.556 | `03_teach_backbone_banks.py` |
| G2 fill | 0.3959 against a 0.3121 prior, gap probe 0.3562 | `04_teach_patch_banks.py` |
| R2 repair / preserve | 0.4232 / 0.5312, pool wrong 0.557 | `04_teach_patch_banks.py` |
| joint repair, overall / backbone / patch | 0.3150 / 0.3141 / 0.3162, eval pool wrong 0.558 | `05_teach_joint_bank.py` |
| the joint bank beats the bank it replaces on backbone addresses, at a matched 64-book configuration | 0.372 against 0.287 (dial-taught banks; after the retrain `03`'s 64-book R1 repairs 0.2907 and the `05 --backbone` side was not re-run) | `05 --backbone 64x16@p0.5` and `03 --points 64x16@p0.5` |

## Sec. V B — the energy model (Table V)

`11_operating_point.py`, `12_energy_model.py`, `13_energy_frontier.py`,
`22_binary_energy.py`; the model is `chill_zen/energy_geom.py` and
Appendix B of the paper gives every number. All of it is a physical
estimate from a stated geometry, never a measurement.

| claim | value | source |
|---|---|---|
| the emulator's default noise gain implies a Johnson source of | 1.2 MOhm at the reference read | `11` |
| draw noise, median over deployed contexts | 0.0293 backbone (0.147 at step 0 to 0.0161 at step 127), 0.0165 patch | `11` |
| cold verdict margin over the physical flicker floor, on the joint sweep's *output* states | 0.0913 against 2.16e-03, a factor of 42 | `11` |
| the same margin on the states the sweeps actually *read* (their input), against the combined floor 3.0e-03 | R1 median 0.027, joint 0.032, about ten floors; 7.3% / 6.5% of groups within one floor, 21% / 18% within three. The paper cites these, not the 42 | `35_cold_noise_control.py` |
| settling, 5RC across the band | 0.093 to 1.87 ns at 100 fF | `11` |
| the thermodynamic floor at the draw noise | 2.3e3 kBT backbone, 7.4e3 kBT patch | `11` |
| census per image | 2048 + 2048 + 1568 + 3616 + 784 (decode) = 10,064 lane reads over 10,080 lanes in five banks; 1,781,920 (space, lane) address assertions; 2.85e7 synapse pairs, 5.70e7 memristors | `12` |
| geometry | 20 nm via on a 50 nm crossbar pitch; one 4x4 unit crossbar per (lane, space); two 4:1 pass-transistor muxes, the a and b sides sharing select lines; transistor pitch 0.20 um; lane pitch 1.00 um; C_seg 0.80 fF; C_node 161 fF (K = 134) / 278 fF (K = 232) | `chill_zen/energy_geom.py`, Appendix B |
| E_sel, address delivery at 0.8 V (91% of the total) | 1.82e-9 J | `12` |
| E_read, the divider reads at 5 C_node V^2 (1%) | 1.97e-11 J | `12` |
| E_readout, comparator + lane buffer at 15 fJ (8%) | 1.51e-10 J | `12` |
| total per image, nominal | 2.00e-9 J, 7.9x below the DTM chain's 1.568e-8 J | `12` |
| low-swing select, 0.5 V | 8.83e-10 J (17.7x) | `12` |
| 7 nm-class periphery / with 0.5 V | 8.24e-10 J (19.0x) / 4.24e-10 J (37.0x) | `12` |
| pessimistic corner (16x16 arrays, separate a/b select lines, 1.0 V) | 8.07e-9 J (1.9x less) | `12` |
| single-knob sensitivities, as DTM/ours | separate a/b lines 6.4; 16x16 4.6; readout 45 fJ 6.8; wire 0.3 fF/um + gate 0.5 fF 5.0; crossbar pitch 100 nm 7.5; transistor pitch 0.4 um 6.6 | `12` |
| bounds on the excluded terms | rails 1.43 nF over 10,080 lanes, pulsed per read step: 5.4e-12 J (0.3%); leakage 12 off gates per space on a DC path, 2.14e7 gates, 2.5e-13 J at 5 pA and 10 ns pulses (held 30 us at 50 mV would be 1.6e-10 J, 8%); kickback 4 mV on 131 fF | `12` (excluded_bounds), Appendix B |
| their cell, for comparison | E_cell = 2 fJ per conditional update, E_comm about half; RNG 2e-15 J per Bernoulli sample | their Eq. 13 |
| digital-decode alternative, carried in Appendix B's basis of comparison for a reader who does not grant the lane decode | 784 x 226 = 177,184 MACs at 10-50 fJ per 8-bit MAC = 1.8-8.9 nJ on the 2.0 nJ total | arithmetic; the lane decode is the design and is priced as 784 cold reads |

Per-image energy across the frontier, `13`; the model charges counts
only, so the binary-trained arms (`22`) equal their grayscale twins:

| system | nominal | low-swing 0.5 V | pessimistic corner |
|---|---|---|---|
| 16 books | 4.93e-11 | 3.13e-11 | 1.48e-10 |
| 32 books | 9.88e-11 | 5.56e-11 | 3.35e-10 |
| 64 books | 2.49e-10 | 1.24e-10 | 9.27e-10 |
| 128 books | 7.51e-10 | 3.41e-10 | 2.98e-9 |
| deployed two-level | 2.00e-9 | 8.83e-10 | 8.07e-9 |

The one-level rows carry the draw, the sweep, and the 784-lane decode.
The retired model (100 fF lane, 1 fJ comparator, no address delivery;
1.87e-11 J, 838x) is in `logs/12_energy_model.log` before 2026-09-06.

**The readout priced as a circuit, `38_readout_options.py`** (Appendix
B, readout table). The model prices one continuous-time comparator per
lane, 200 nA from 0.8 V for a 62.5 ns window = 10 fJ, plus the 100 nA
lane buffer = 5 fJ.

| readout circuit | per lane read | readout share | total | ratio |
|---|---|---|---|---|
| continuous-time, 200 nA for 62.5 ns (as priced) | 15 fJ | 8% | 2.00e-9 | 7.9x |
| continuous-time, 1 uA for 62.5 ns | 55 fJ | 23% | 2.40e-9 | 6.5x |
| clocked, binary search on the winner, 5 strobes at 10 fJ | 55 fJ | 23% | 2.40e-9 | 6.5x |
| clocked, full ramp, 16 strobes | 165 fJ | 47% | 3.50e-9 | 4.5x |
| clocked tournament, 15 pairwise decisions per 16-lane group | 14.4 fJ | 7% | 1.99e-9 | 7.9x |

| claim | value | source |
|---|---|---|
| continuous-time noise-energy bound, E = 4kT gamma (n V_T) V_dd (BT) / v_n^2 (gamma 1, n V_T 35 mV, 0.8 V, BT 3) | 0.35 fJ at 2 mV (code 95); 1.39 fJ at 1 mV (code 45); 15.5 fJ at 300 uV (code 10); 139 fJ at 100 uV (code 0) | `38` |
| illustrative ramp condition for one noise sample per lane | slope >= 6 v_n 2 pi B = 38 mV/ns at 2 mV and 500 MHz; the 20 mV range in 0.53 ns | `38` |

The BT=3 noise-energy examples are not the same circuit point as
500 MHz over 62.5 ns (BT=31.25, 10.4 times greater energy in that
formula). They do not verify the priced comparator.

**Sneak paths in the unit crossbar, `36_sneak_paths.py`** (Limitation 2
and Appendix B). A nodal solve with every device at a common conductance
G, one row and one column driven, the other lines floating; then the
energy model at each way out, on the deployed census.

| claim | value | source |
|---|---|---|
| 4x4 floating-line sneak network | 3G / 9G / 3G in series = 1.286G in parallel with the selected device; the node sees 2.286G; selected share 7/16 = 0.4375 | `36` |
| first-order weight of each unselected device in the read (selected = 1) | 0.184 for each of the 6 row/column neighbours, 0.020 for each of the 9 others; shares 44% / 48% / 8% | `36` |
| fraction of a pulse the unselected devices see | 3/7 = 0.429 (6 neighbours), 1/7 = 0.143 (9 others) | `36` |
| 16x16 (the pessimistic corner's array) | sneak 7.258G; selected share 31/256 = 0.121 | `36` |
| node-side lines at fixed ground | differential read exact; shunt 2.25G per side, read gain 0.31, hot read 32.5 mV to restore 10 mV at the comparator; driver sinks 3 G V^2 per side | `36` |
| node-side lines track V_y | read exact; replica driver sinks 3 G V^2 per side, 3KGV per lane (unpriced; reintroduces G) | `36` |
| 16x1 selector column | no sneak path; no driver; 16 select lines per space instead of 8; leakage paths per space 30 instead of 12 (2.5x); held-rail leakage would be 25-44% of its total, pulsed 0.04-0.07% | `36`, `energy_geom.excluded_bounds` |

| geometry (Limitation 2 table) | E_sel | E_read | total | ratio |
|---|---|---|---|---|
| floating lines, as priced | 1.82e-9 | 1.97e-11 | 2.00e-9 | 7.9x |
| node-side lines at fixed ground, hot read 32.5 mV | 2.74e-9 | 1.29e-10 | 3.02e-9 | 5.2x |
| node-side lines track V_y (buffer unpriced) | 2.74e-9 | 1.13e-10 | 3.00e-9 | 5.2x |
| 16x1 selector column, pass gates beside the array (p = 3.25 um) | 1.43e-9 | 5.3e-12 | 1.58e-9 | 9.9x |
| 16x1 selector column, pass gates stacked along V_y (p = 0.25 um) | 7.41e-10 | 2.10e-11 | 9.13e-10 | 17.2x |

The pull-down rows charge a third line per assertion (the node-side
mux's complement line) and double the node-side mux stack (C_node 230 /
399 fF); E_read carries the drivers' 3x sink. The selector-column rows
charge one line per assertion and have no mux on the node.

**Physical-operating-point reconciliation, `28_physical_energy.py`.**
Confirms the hot-read operating point (10 mV, settling-limited) the
comparator sweeps use: the earlier joule model
already clamped the operating voltage to `V_FLOOR` whenever the
emulator-`T`-derived target wanted a sub-floor read, which is every
hot-read class reported here, so `e_bound = V_FLOOR^2 G_sum t_b`
either way (ratio to the old value 1.0000 at every class checked).

| claim | value | source |
|---|---|---|
| grayscale composite sigma at the physical point (P4, v_n=2mV) | 0.2032 (ctx 0) to 0.2007 (patch), matching the sweep's own composition to three digits | `28_physical_energy.py` |
| lane-only sigma (no comparator) | 0.017-0.036 across the chain | `28_physical_energy.py` |
| binary composite sigma at the physical point (P6, v_n=3mV) | 0.3014 (ctx 0) to 0.3005 (ctx 127), matching the binary sweep's own composition exactly | `28_physical_energy.py` |
| reference-read row (P5 twin, V=50mV, v_n=5mV) | lane 0.0050, comparator 0.1000 -- comparator-dominated, matches P5's measured 97-100% share | `28_physical_energy.py` |

## Sec. IV C — the comparator sets the draw temperature (Fig. 4)

The grid, the per-point noise composition (thermal / flicker /
comparator variance shares) and both curves are printed by
`23_comparator_sweep.py` and `24_binary_comparator_sweep.py`
(`--extend` adds the last two binary points); the argument is
Sec. IV C of the paper and the read model is `chill_zen/physical.py`.

Every level is a code on the emulator's 8-bit comparator register
(`COMPARATOR_V_MIN` 100 uV, `COMPARATOR_V_STEP` 20 uV, codes 0-255,
ceiling 5.20 mV): `chill_zen.physical.register_level` raises on any
level the register does not reach, so a swept point is representable by the modeled register. This does
not demonstrate a physical part. `v_n/V = 0` is the modeling switch -- the
comparator term is not applied -- and not code 0, which is the quietest
comparator modeled.

| $v_n/V$ | code | grayscale end-to-end critic | binary-trained two-level FID |
|---|---|---|---|
| 0 (P0) | -- | 0.8163 | 22.14 |
| 0.016 (P1) | 3 | 0.8453 | 20.53 |
| 0.05 (P2) | 20 | 0.8647 | 14.86 |
| 0.10 (P3) | 45 | 0.8840 | 10.99 |
| **0.20 (P4)** | 95 | **0.8900** | 10.04 |
| **0.30 (P6)** | 145 | -- | **9.69** |
| 0.50 (P7) | 245 | -- | 11.57 |
| deployed (emulator $T$) | -- | 0.8273 | 10.63 (`06_binary_arm.py` records 10.89 at T = 0.3 on its own seed block) |

The 50 mV twin P5 (`V = 50 mV, v_n = 5 mV`, code 245, `v_n/V = 0.10`):
grayscale 0.8837, binary 10.73. Every value above is from the banks
taught on the physical fill read (Sec. III B); the same sweep on the
dial-taught banks peaked at P3 (0.9043) rather than P4, which is why
the operating point moved when the training path was made physical.

At $V = 10$ mV the register's ceiling caps $v_n/V$ at 0.52, which is
where the sweep stops. A ninth point at $v_n = 7.5$ mV ($v_n/V = 0.75$,
binary FID 12.27) was swept before the level was a register and is not
reported: 7.5 mV is above the ceiling, and above every comparator class
of Table III. The turnover it was declared to find is already in
P6 -> P7.

| claim | value | source |
|---|---|---|
| P3 = P5 (same v_n/V, 5x different V) | grayscale critic 0.8840 vs 0.8837; binary FID 10.99 vs 10.73 -- the comparator's share sets the temperature, not V | `23_comparator_sweep.py`, `24_binary_comparator_sweep.py` |
| the binary turnover basin | P4-P6 differ by 0.35 FID, about the seed-block scatter on the deployed arm (10.63 vs 10.89) | `24_binary_comparator_sweep.py --extend` |
| the comparator's share of the read variance at the optimum | P4: 97% at ctx0 rising to 99% from ctx16 on; P3 was 88-97% | `23_comparator_sweep.py` (composition table) |
| comparator classes modeled against | raw dynamic latch 0.5-2mV rms / 5-20mV offset; auto-zeroed 50-200uV / 0.1-1mV; trimmed: underlying class noise / tens of uV offset | Table III of the paper; sources Razavi 2015, Pelgrom 1989 |
| T_max, lanes alone at the 10mV floor | ~0.037 empty context, ~0.017 full context | `23_comparator_sweep.py` (the `v_n = 0` point's composition) |
| lanes-alone flicker share across the chain | 70% at ctx0, then 27 / 9 / 5 / 5% at ctx16 / ctx64 / ctx127 / patch -- flicker dominates the first read, thermal the rest | `23_comparator_sweep.py` (the `v_n = 0` point's composition) |
| the median-convention note | grayscale backbone read noise 0.023 (median over steps) vs 0.030 in the energy log (mean of per-step medians) | `11_operating_point.py` against `23_comparator_sweep.py` |

**The cold-read control, `35_cold_noise_control.py`.** The deployed
generation runs the R1 sweep and the joint sweep as noiseless argmax
reads. The control runs both through `chill_zen.physical` at 50 mV,
device terms on at the settling-limited pulse, comparator at register
code 0 / 10 / 45 (100 uV / 300 uV / 1 mV; v_n/V = 0.002 / 0.006 /
0.02), on the same hot draws (P4, seeds 990-992, n = 1000). "chain" is
the whole cold path with noise (noisy R1, patch read on that g, noisy
joint); "end-to-end" and "recon" isolate the joint sweep on the
reported inputs. Flip fractions are against the noiseless sweep on the
identical input.

| cold read | chain critic | chain div | chain tone | recon critic | R1 flips (of 128) | joint flips (of 226) |
|---|---|---|---|---|---|---|
| noiseless (reported) | 0.8903 +-0.012 | 0.5490 | 1.553 | 0.8967 +-0.008 | -- | -- |
| code 0, 100 uV | 0.8897 +-0.016 | 0.5497 | 1.555 | 0.8957 | 9.5 | 12.4 |
| code 10, 300 uV | 0.8910 +-0.010 | 0.5496 | 1.553 | 0.8973 | 13.7 | 19.8 |
| code 45, 1 mV | 0.8927 +-0.024 | 0.5544 | 1.529 | 0.9023 | 33.7 | 51.7 |

The recon rows here draw the patch level through the physical read at
P4, as `21_reporting.py` now does for both of Table II's arms (since
2026-09-06; on its own seeds it reads 0.8963). Before that date Table
II's reconstruction row drew the patch level through the fixed-gain
emulator read (T = 0.1, 50 mV, 1 us, no comparator) and read 0.8837.

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

| system | FID | div | J/image (nominal) |
|---|---|---|---|
| real train images (control) | 1.907 | 0.1806 | — |
| 128-book codec reconstruction reference | 14.205 | 0.1634 | — |
| **two-level (deployed)** | **19.431** | 0.1643 | 2.00e-9 |
| one-level, 128 books | 21.975 | 0.1640 | 7.51e-10 |
| one-level, 64 books | 23.811 | 0.1597 | 2.49e-10 |
| one-level, 32 books | 30.300 | 0.1397 | 9.88e-11 |
| one-level, 16 books | 45.296 | 0.1288 | 4.93e-11 |
| DTM, 8 steps | 24.90 | — | 1.568e-08 |
| DTM, 2 steps | 32.60 | — | 3.921e-09 |

Table above is superseded (no T-drawn number stands as a
result). Table VII carries each grayscale arm at its best level on the
register grid: `head_to_head/34_grayscale_grid.py` renders the two-level
stack and the one-level 16/32/64/128-book systems at P0, P1, P2, P3, P4
and P6 (`V = 10 mV`) and, with `--fifty`, the one-level systems at
Q0 and Q1 (`V = 50 mV`, codes 0 and 10, v_n/V 0.002 and 0.006);
`07_comparator_score.py` scores them in the DTM environment and prints
each arm's best, and `15_energy_figure.py`, `16_binarization_figure.py`
and `31_free_draw.py` read that selection. P0 is the lanes-only
modeling switch, not a register setting, and is never selected as an
arm's level. The full grid (FID; the P3 and P4 columns reproduce the
single-point renders of `25_comparator_headtohead.py` bit for bit):

| arm | P0 (0) | P1 (0.016) | P2 (0.05) | P3 (0.10) | P4 (0.20) | P6 (0.30) | Q0 (0.002 @ 50 mV) | Q1 (0.006 @ 50 mV) |
|---|---|---|---|---|---|---|---|---|
| two-level | 29.67 | 25.46 | 18.65 | 18.09 **best** | 21.95 | 23.48 | — | — |
| one-level-128 | 34.94 | 28.51 | 22.11 **best** | 25.30 | 32.23 | 34.72 | 41.32 | 39.63 |
| one-level-64 | 26.52 | 24.67 | 22.62 **best** | 25.91 | 29.01 | 30.61 | 30.60 | 29.79 |
| one-level-32 | 24.81 | 25.76 **best** | 31.20 | 33.82 | 36.05 | 36.66 | 27.67 | 26.00 |
| one-level-16 | 60.65 | 54.79 | 45.71 | 44.15 **best** | 44.64 | 44.56 | 70.22 | 66.69 |

The 50 mV points were run because the 32-book FID was still falling at
P1 and the 10 mV register has no physical point below it; every arm
scores worse at 50 mV than at its 10 mV optimum, so the 32-book system's
best physical point is P1.

The rows Table VII carries:

| system | level | FID | div | J/image (nominal) |
|---|---|---|---|---|
| **two-level (physical)** | P3, v_n/V 0.10 | **18.09** | 0.1558 | 2.00e-9 |
| one-level, 128 books | P2, 0.05 | 22.11 | 0.1520 | 7.51e-10 |
| one-level, 64 books | P2, 0.05 | 22.62 | 0.1426 | 2.49e-10 |
| one-level, 32 books | P1, 0.016 | 25.76 | 0.1199 | 9.88e-11 |
| one-level, 16 books | P3, 0.10 | 44.15 | 0.1162 | 4.93e-11 |

**Each arm has its own comparator optimum, and smaller codes want a
quieter comparator.** At those levels the frontier is monotone in book
count and the 128- and 64-book one-level scores are below the published DTM
reference; the 32- and 16-book systems are not. Rendering every arm at the
two-level stack's critic optimum (P4) instead gave 21.95 / 32.23 /
29.01 / 36.05 / 44.64 -- a kink at 128 books, which is what prompted the
grid. Energies are unchanged: the bounded figures do not depend on draw
quality.

**The critic and the FID disagree on the two-level stack's level.**
The critic (Table II) peaks at P4, 0.8900 against 0.8840 at P3; the FID
peaks at P3, 18.09 against 21.95 at P4. Table II reports the critic's
optimum, Table VII the FID's, both on the same declared grid, and the
selection is by the reported metric itself (Limitations). The
superseded fixed-gain draw scores 19.43 on this protocol.

| claim | value | source |
|---|---|---|
| energy ratio, nominal | 7.9x less | `12_energy_model.py` |
| energy ratio, low-swing 0.5 V select | 17.7x less | `12_energy_model.py` |
| energy ratio, 7 nm-class periphery | 19.0x less (37.0x with 0.5 V) | `12_energy_model.py` |
| energy ratio, pessimistic corner | 1.9x less | `12_energy_model.py` |
| reconstruction reference scope | 14.205 is backbone-only; it is neither a full-stack reference nor a lower bound on generator FID | `02_generate.py`, `03_score.py` |
| diversity against real | deployed point 0.164 against 0.181 (91%); physical two-level point (P3) 0.156 (86%); the codec's own render of real codes is 0.163 | `03_score.py`, `07_comparator_score.py` |
| their GPU baselines | VAE best 17.9, GAN best 26.3, MEBM best 33.5; DDPM best decoded 12.2 (clipped by their y-axis; best visible 16.4) | `reference/dtm_figure1_data.py` |

**The binary-trained arms** — fit by `17_fit_binary_codec.py`, taught
by `18_teach_binary_banks.py`, screened and rendered by
`19_binary_verdict.py`, scored by `head_to_head/06_binary_arm.py`,
energy census by `22_binary_energy.py`. The temperature sweep
{0.1, 0.2, 0.3, 0.5} was declared before the run; Table VII carries the
physical-point rows below, not these.

| system | FID | div | J/image bounded |
|---|---|---|---|
| binary codec reconstruction reference, 128 books | 9.566 | 0.1675 | — |
| binary stack reconstruction (real codes, both levels) | 2.832 | 0.1757 | — |
| binary real-train control | 1.907 | 0.1806 | — |
| binary two-level, T = 0.3 (superseded) | 10.889 | 0.1891 | 2.00e-9 |
| binary two-level, T = 0.5 / 0.2 / 0.1 (superseded) | 13.192 / 8.969 / 15.070 | 0.192 / 0.168 / 0.136 | — |
| binary one-level 128, T = 0.2 (superseded) | 12.275 | 0.1541 | 7.51e-10 |
| binary one-level 64, T = 0.2 (superseded) | 12.665 | 0.1640 | 2.49e-10 |

Physical-point rows (P6 for two-level and for one-level; both
10 mV/3 mV, `v_n/V = 0.30`), the ones Table VII carries:

| system | FID | div | J/image (nominal) |
|---|---|---|---|
| **binary two-level (physical, P6)** | **9.69** | 0.1472 | 2.00e-9 |
| binary one-level 128 (physical, P6) | 15.46 | 0.1303 | 7.51e-10 |
| binary one-level 64 (physical, P6) | 15.10 | 0.1294 | 2.49e-10 |

The physical two-level point now beats the superseded fixed-gain
control (9.69 against 10.63 on the same seed block), where on the
dial-taught banks it trailed it (11.10 against 9.82).

**Flagged, not softened:** the one-level rows get worse at the
physical point too (12.28 -> 15.46, 12.67 -> 15.10), the same
direction as the grayscale frontier and for the same reason: P6
was found by sweeping the two-level stack. Both one-level rows still
clear the DTM bar (24.90) by a wide margin; they are simply farther
from it than the deployed numbers implied.

| claim | value | source |
|---|---|---|
| the codec fit reproduces the recorded binary floors | 128: disagree-ev 0.0169 / relMSE-ev 0.0788; 64: 0.0266/0.1046; two-level 0.0068/0.0464; dead 0 | `17_fit_binary_codec.py` |
| teach checks | G lift +0.2213 (128) / +0.2560 (64), patch G2 +0.1712; R repair 0.2980 / 0.3356, R2 0.5281, joint 0.3746 | `18_teach_binary_banks.py` |
| T = 0.1 collapses binary diversity at the screen | one-level div 0.1330 against the 0.75 x 0.1811 rule (real bar critic 0.8680, div 0.1811) | `19_binary_verdict.py` |
| the window draw screens out | div 0.1025 against a 0.1358 bar, critic 0.7890; T_max median 0.014 (ctx 0) to 0.043 (ctx 127) | `19_binary_verdict.py` |
| two-level beats one-level | 9.69 against 15.46 at the physical point, 5.8 FID (8.97 against 12.28 on the superseded best-T rows) | `07_comparator_score.py`, `06_binary_arm.py` |
| full binary two-level reconstruction reference | 2.832; 9.566 is backbone-only. These FIDs do not bound the generator or give an additive error decomposition | `19_binary_verdict.py`, `06_binary_arm.py` |
| binary two-level energy equals grayscale | 2.00e-9 J; the model charges counts only and the census is identical | `22_binary_energy.py` |
| binary one-level energies equal their grayscale twins | 7.51e-10 (128) / 2.49e-10 (64); the earlier 2.5x deviation was an artifact of the retired magnitude-based model | `22_binary_energy.py` |

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
| free-label draw, binary two-level P6 | 10.10 (against 9.69 clamped; free is now slightly worse, where on the dial-taught banks it was slightly better) | `31_free_draw.py`, `32_audit_score.py` |
| free-label draw, grayscale two-level at its grid best (P3) | 18.27 (against 18.09 clamped); at P4 it was 21.92 against 21.95 | `31_free_draw.py`, `32_audit_score.py` |
| Bernoulli calibration, class-conditional / pooled | 180.51 / 258.10 | `30_bernoulli_baseline.py`, `32_audit_score.py` |
| binary memorization, two-level P6 | 5120/5120 distinct, 0 exact copies against actual 68k fitting split; excluded-from-fit real: 5/2,000; NN-Hamming median 25 versus 23, generated q01 4 | `33_binary_memorization.py` |
| binary memorization, one-level 128 / 64 | 5120/5120 distinct, 0 exact copies each | `33_binary_memorization.py` |

## Figures

Numbers are the compiled order (labels in parentheses are the stable
reference); Fig. 2 (`fig:arch`) is drawn
in TeX and have no script. `14_figures.py` also writes `fig-teach.png`,
which the paper no longer includes.

| figure | script | needs |
|---|---|---|
| Fig. 1 `fig-noise.png` (`fig:noise`) | `14_figures.py` | 11 |
| Fig. 3 `fig-samples.png` (`fig:samples`) | `14_figures.py` replays `21_reporting.gen_states`, seed 960, first 6/class from n=1000 | final banks; no refit |
| Fig. 4 `fig-comparator-sweep.png` (`fig:comparatorsweep`) | `29_comparator_figure.py` | 23, 24 (+ `--extend`) |
| Fig. 5 `fig-backbone.png` (`fig:backbone`) | `14_figures.py` | 06 `--all` |
| Fig. 6 `fig-codec.png` (`fig:codec`) | `14_figures.py` | 01 `--all`, 02 |
| Fig. 7 `fig-binarization.png` (`fig:binarization`) | `16_binarization_figure.py` | the grayscale grid renders and their scores (34, 07_comparator_score) |
| Fig. 8 `fig-energy.png` (`fig:energy`) | `15_energy_figure.py` | 13, 22, and the physical-point scores (34, 27, 07_comparator_score) |
