# Astra editorial rewrite

The manuscript is `main-astra-rewrite.tex`; its compiled PDF is `main-astra-rewrite.pdf`.
The PDF has 10 pages, including appendices and references. The source manuscript's existing PDF has 19 pages.

## Build

Run `./build.sh` from this directory (or invoke it by its path). It builds with Tectonic, keeps intermediate files and logs in `build/`, and copies the final PDF beside the source. It runs no experiments. The script uses only cached TeX packages, which were sufficient on this machine. On a machine without those packages, a first Tectonic run without `--only-cached` will need to download them.

## Editorial approach

The prose was written anew from `../main.tex`, checked against `../NUMBERS.md`, and organized around representation, generation, image quality, comparator noise, and hardware implementation. The appendix retains the local instruction rules and the energy calculation. Four existing figures were copied without modification. The bibliography was copied from the original.

The rewrite consolidates repeated explanations and qualifications. It retains the distinction between emulator results and projected hardware performance, the validation and training-split limitations, the differing critic and FID noise settings, the nominal crossbar's sneak paths, and the unverified comparator and decoder requirements. Secondary baseline comparisons, duplicate energy tables, and the acronym list were omitted to shorten the presentation.

Notation and reporting clarifications:

- Dropout masks, generated symbols, and physical temperature have separate notation.
- Bank size, lane reads, noisy reads, and address assertions are distinguished.
- Cold-read spread is labeled as the maximum-minus-minimum seed range, as specified in `../experiments/35_cold_noise_control.py`.
- Tone units and foreground definitions follow `../chill_zen/judge.py`.
- Binary one-level rows are identified as using the two-level model's selected noise setting, rather than as individually optimized results.
- Historical representation studies are distinguished from the final comparator-setting results.

No experiments, training, rescoring, or plot regeneration were performed. Existing repository files were not edited.

## Verification

The final PDF was rendered and all ten pages inspected. All 18 distinct internal reference targets and all 27 cited bibliography entries resolve; all four figures match their originals byte for byte. All four tables and all four figures appear in the PDF, with no overfull boxes or clipped content. REVTeX emits recoverable float-placement warnings, and the bibliography emits spacing/style warnings; the final rendered pages were checked directly. Build logs and the PDF checksum are in `build/` and `qa/checks.json`; the checked page renders are in `qa/`.
