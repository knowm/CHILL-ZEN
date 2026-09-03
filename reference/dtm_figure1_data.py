"""The comparison points of Fig. 1 of Jelincic et al., arXiv:2510.23972v2.

Decoded losslessly from the vector PDF `Images/eff_fig.pdf` inside the
arXiv source tarball (https://arxiv.org/e-print/2510.23972): polyline /
marker coordinates mapped through the axis gridlines (x: 1e-8..1e0 J,
log; y: 0.00..0.07 FID^-1, linear). Validated: the VAE points reproduce
their Appendix F Table III *theoretical* column (2.3e-5, 0.434e-4,
1.7e-3 J; FID 30.5, 27.4, 17.9) to the two significant figures that
column prints — Fig. 1 plots theoretical GPU energies.

Each entry: (energy_J_per_sample, FID). FID^-1 = 1/FID.
"""

DTM = [  # orange, "Ours"; energies = 1.961 nJ x {1, 2, 4, 6, 8} (chain depth)
    (1.961e-9, 78.4),
    (3.921e-9, 32.6),
    (7.843e-9, 28.5),
    (1.176e-8, 26.5),
    (1.568e-8, 24.9),
]

MEBM = [
    (8.60e-8, 143.3),
    (4.65e-7, 104.1),
    (1.69e-6, 69.1),
    (3.59e-6, 38.2),
    (1.04e-5, 33.5),
]

GAN = [
    (6.93e-5, 28.8),
    (2.52e-4, 28.4),
    (1.94e-3, 26.3),
]

VAE = [
    (2.30e-5, 30.5),
    (4.34e-5, 27.4),
    (1.72e-3, 17.9),
]

DDPM = [
    (1.410, 120.4),
    (2.820, 88.6),
    (3.525, 59.2),
    (4.231, 49.8),
    (4.936, 22.3),
    (5.642, 16.4),
    # Their y-axis (FID^-1 <= 0.07) clips this vertex; it is their best
    # DDPM point and the paper's DDPM comparisons are stated against it.
    (5.952, 12.2),
]

SERIES = {"DTM": DTM, "MEBM": MEBM, "GAN": GAN, "VAE": VAE, "DDPM": DDPM}
