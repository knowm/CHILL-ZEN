"""chill_zen — the CHILL ZEN companion package.

Everything the paper's experiments are built from: the data split, the
dropout-AQ codecs, the lane bank runtime, the AAT layouts of the two
levels and the joint bank, the teaching routines, the generation
recipe, the judge, and the energy model.

The lane emulator itself is not here. It is the public
``ktram-neural-core`` package (github.com/knowm/ktram-neural-core);
every read and every weight update in this repository goes through it.
"""

__all__ = ["config", "data", "codec", "lanes", "levels", "teach",
           "generate", "judge", "physical", "energy", "artifacts"]
