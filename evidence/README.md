# Evidence preserved during finalization

`original-logs/` contains the local experiment logs as found, including
historical configurations. `scoring/` preserves the scoring JSON and logs that are otherwise
ignored by git. `verification/` contains the corrected
binary memorization audit and short checks made during finalization.
The full canonical-IDX/OpenML image arrays were compared after the
seed-0 permutation and were exactly equal for all 70,000 images.

The figure replay uses reporting seed 960, n=1000, and reproduces
critic 0.9210 (end-to-end) and 0.8910 (reconstruction) from the
original reporting log. It refits no weights.

`snapshot.json` records hashes of files present during this pass.
These are retrospective inventory hashes, not evidence that a given
historical bank produced a saved render. The old generation cache
does not record that relationship. Missing intermediate fill-sweep
logs remain missing. No clean full-pipeline rebuild was performed.

The reported emulator installation records commit
`6899ca6717377a952782724c33c85b0c4e580e56` in its direct_url.json;
the installation instructions now pin that revision.
