# CHILL ZEN finalization pass

The emulated generative result remains worth publishing. I found no reason to retrain the models to support the narrower claim that this architecture generates Fashion-MNIST with the recorded quality under its stated protocol. The paper-to-paper comparison supports approximately 7.9 times lower modeled energy and lower reported FID under the stated assumptions. This is a comparison of estimates and published results; it does not require finished hardware. Identical training/evaluation conditions and a measured energy advantage are stronger claims that this work does not make.

I read `repo/main.tex` before the hostile review, then checked the relevant generation, training, evaluation and figure code against saved results. The review's resolved labels were treated as prior assessments, not proof that an issue was closed.

## Changes that matter

1. **The reconstruction reference was misidentified.** The 9.57 binary FID is for backbone-only reconstruction. The existing full two-level reference is 2.83. The generator's 9.69 therefore is not “0.1 above its own ceiling.” Moreover, reconstruction FID is not a mathematical lower bound on generator FID, and FID differences cannot be assigned additively to codec and draw errors. The revised table includes the full two-level reference and removes the error-budget inference. No FID was recomputed.

2. **The reconstruction arm does not isolate the draw.** `21_reporting.gen_states` supplies real backbone codes directly to the patch generator, bypassing both the autoregressive draw and R1. The paper now describes that actual control. Its classifier agreement cannot localize every remaining error to the draw.

3. **The selected models were evaluated during development.** Epoch selection uses the 2,000-image partition. Fill and comparator settings were also selected during development, and reported FIDs select the best tested settings on their own scores. This is disclosed where the measurements are introduced. Seed ranges are now described as ranges, not confidence intervals or tests of equivalence. The grayscale critic sweep's best point is its endpoint, not an interior optimum.

4. **The binary memorization audit used the wrong split.** The corrected script compares saved renders against the actual permuted 68,000 fitting images. All 5,120 two-level renders remain distinct; none is an exact fitting-image copy. Median nearest-neighbor Hamming distance remains 25 bits, versus 23 for the 2,000 excluded-from-fitting reference images. Five reference images exactly duplicate fitting images. Those reference images were used for selection, so the paper no longer calls this an untouched test. The canonical/OpenML correspondence was verified exactly over all 70,000 images.

5. **The paper's sample figure used a retired generator.** The figure script loaded `07_verdict` fixed-gain states while its caption described comparator-driven generation. It now replays reporting seed 960 at the stated setting and shows the first six images per class. That replay reproduced critic 0.9210 and 0.8910 for the two arms, matching the original reporting log. No model was fitted or selected in this replay.

6. **The DTM evaluation attribution was too strong.** The near-zero reference gate validates our implementation of the replication pipeline. It does not prove that the published DTM scores used it. Captions, the abstract, and the comparison text now distinguish replication-scored CHILL ZEN results from transcribed DTM results. The different fitting splits remain disclosed. The bibliography identifies [arXiv v2](https://arxiv.org/abs/2510.23972v2), the version used for the comparison, rather than an unverified journal publication.

7. **Several hardware conclusions exceeded the model.** The nominal floating-line crossbar does not implement isolated reads. Its approximately 2 nJ estimate is retained as a conditional calculation; the already-priced selector-column alternative is stated separately. The readout discussion now identifies the incompatible bandwidth-time examples: 500 MHz over 62.5 ns gives BT=31.25, whereas the quoted noise-energy numbers use BT=3. The kickback estimate also omitted the first read's small node: 0.5 fC on about 12 fF implies about 42 mV, not the 4 mV quoted for 131 fF. These are unresolved circuit requirements, not failures of the emulated generation result.

8. **The model description now says what is implemented.** Noisy argmax defines a conditional choice distribution; the local rule does not establish calibrated data conditionals. The Beta interpretation is restricted to forward accumulation, with conductance increments expressed as pseudo-counts. Reverse feedback does not preserve a monotone evidence count or prove a 1/n schedule. “No gradients anywhere” now applies to bank teaching, with host-fitted codebooks and Adam-trained critic stated separately.

## Reproducibility work

The revision preserves existing logs in `evidence/`, corrects the binary audit and figure scripts, pins installation to the emulator revision recorded by the installed environment, and fixes stale documentation. The paper build now depends on its figure files as well as LaTeX and bibliography. A snapshot manifest records current artifact and render hashes.

Those hashes are retrospective. They cannot prove which earlier bank produced an old render. Intermediate fill-sweep logs remain missing; the paper no longer presents those runs as an isolated, auditable experiment. Historical control sections remain explicitly historical. Large fitted artifacts and experimental weights were not changed.

## What still needs a decision before public release

- **Scope:** publish this as an emulated model with a conditional hardware analysis. A claim of matched DTM superiority needs a matched training/evaluation protocol. A claim of a realized energy advantage needs a consistent readout, decoder and array implementation. Wording cannot supply that evidence.
- **Historical reproducibility:** the saved logs support auditing the reported numbers, and one reporting seed was replayed successfully. A full clean rebuild was not run, so this pass does not certify that every historical score can be regenerated from the current artifact cache. Establishing that stronger claim would require a fresh dependency-tracked run, not invented provenance.
- **Public references:** the private companion repository's intended public URL should be checked when it is released. The neural-lane blog citation still needs a stable, accessible chapter permalink; its current generic URL was not verified. The fetched replication has no license file at the recorded revision; it is not redistributed.

## Checks performed

- Corrected binary nearest-neighbor audit on all three saved arms, with the true 68k/2k split.
- Full equality check between canonical IDX and OpenML image arrays under the documented permutation.
- One reporting-seed replay for the figure, matching both recorded classifier scores.
- Hamming-distance implementation checked against direct XOR distances, including an exact copy and multiple chunks.
- Existing geometry and readout arithmetic scripts rerun; edited Python files parsed.
- LaTeX compilation and visual inspection of the resulting pages; figures regenerated from existing models or saved images as appropriate.

No training rerun, full FID extraction, clean end-to-end rebuild, push, or publication was performed. The main reported FIDs, bank weights, teaching recipe and energy-model constants remain unchanged.


## Write-like-a-pro prose pass

The local composite rose from 89.355 to 90.077. Word count fell from 12,711 to 12,521 (190 words); topical retention is 99.5%, with no lost-terminology warning. Twenty sentence-level edits were accepted through three individually tested batches. The earlier baseline was archived and reset to the substantively expanded hostile-review revision, not to evade the word gate.

The four existing committee checkpoints scored 77, 81, 85, and 81. Their call budget was already exhausted, so no new external review was made. Five of the latest checkpoint's 19 quotations remain: the contributions lead-in, generation lead-in, audit heading, emulator disclosure, and device-variability sentence. The first four provide useful navigation or scope; tested alternatives to the fifth lowered the composite and were rejected under the skill's rule. The phrasing axis remains below the corpus band (63.9); this pass does not claim that every stylistic finding is resolved. The score measures resemblance to the supplied academic corpus, not scientific validity.

The full prose diff was reviewed. The PDF rebuilt successfully to 20 pages, with no unresolved references or overfull boxes; opening pages were visually inspected. The anti-gaming suite passed: 144 passed, 1 skipped. A duplicated-word scan found no matches. The acknowledgments were preserved, including OpenAI Astra-6 and Claude Fable. No experiments were rerun during this prose pass.

Dashboard: http://127.0.0.1:8781/
