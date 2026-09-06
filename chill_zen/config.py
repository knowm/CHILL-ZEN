"""The deployed operating point, and the sweeps the paper reports.

One place for every configuration string, so a script never invents a
code string and a reader never has to grep for what "deployed" means.

The deployed system is a 904-bit code: a 128-book, 16-symbol backbone
over the whole image at keep-p 0.5 (512 bits) plus a 2-book, 16-symbol
residual code on each of 49 4x4-pixel slots (392 bits), with the label
clamped alongside.
"""

BACKBONE = "128x16@p0.5"          # the deployed backbone code: books x symbols @ keep-p
PATCH = "2x16@p0.5"               # the deployed residual patch code: books per slot x symbols @ keep-p

# The backbone codec sweep of Sec. IV E (a) and Fig. 5(b). Tuples are
# (books, symbols, keep-p).
BACKBONE_SWEEP = [
    (16, 16, 0.75), (16, 16, 1.0), (32, 16, 0.75), (32, 16, 1.0),
    (64, 16, 0.75), (64, 16, 1.0), (128, 16, 0.75), (128, 16, 1.0),
    (16, 16, 0.5), (32, 16, 0.5), (64, 16, 0.5), (128, 16, 0.5),
    (16, 32, 0.5), (32, 32, 0.5), (64, 32, 0.5),
]

# The backbone codes that get banks taught and are generated from.
BACKBONE_POINTS = [
    "16x16@p0.75", "32x16@p0.75", "64x16@p0.75", "128x16@p0.75",
    "16x16@p0.5", "32x16@p0.5", "64x16@p0.5", "128x16@p0.5",
    "16x32@p0.5", "32x32@p0.5", "64x32@p0.5",
    "64x16@p1.0",                 # the keep-p = 1 control (plain AQ)
]

# The one-level frontier of Sec. V C and Fig. 9.
FRONTIER = ["16x16@p0.5", "32x16@p0.5", "64x16@p0.5", "128x16@p0.5"]

# The residual patch codec sweep of Sec. IV E (b) and Fig. 6(b):
# (books per slot, keep-p) at 16 symbols on a 7x7 grid.
PATCH_SWEEP = [(1, 0.5), (1, 0.75), (2, 0.5), (2, 0.75), (3, 0.5), (3, 0.75)]

# Anchors quoted in the paper's rate-distortion discussion.
PER_PATCH_CODEC_FLOOR = 0.0576    # a per-patch code at the same 512 bits
