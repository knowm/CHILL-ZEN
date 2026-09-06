"""The energy model on the binary-trained arms.

Paper Sec. V B and Table VII. The model of experiment 12
(chill_zen/energy_geom.py) charges counts only: lanes and spaces per
bank, lane reads and their voltage. The binary-trained two-level stack
and the 128- and 64-book one-level arms have the same banks, spaces and
read census as their grayscale twins, so their energies are identical
to them: 2.00e-9 J (two-level), 7.5e-10 J (128 books), 2.1e-10 J
(64 books) at the nominal point. This script records that under the
keys `15_energy_figure.py` reads.

Cost: seconds. Requires nothing.

    python experiments/22_binary_energy.py
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts                                    # noqa: E402
from chill_zen.energy_geom import (all_points, fmt_row,             # noqa: E402
                                   one_level_census, two_level_census)
from chill_zen.levels import parse_tag                             # noqa: E402

ARMS = {"two-level": None, "128x16@p0.5": 128, "64x16@p0.5": 64}


def main():
    log, fh = artifacts.make_log("22_binary_energy")
    log("binary-trained arms: same census as the grayscale twins, so the "
        "same energy (the model charges counts only)")
    results = {}
    for tag, M in ARMS.items():
        banks = two_level_census() if M is None else one_level_census(M)
        pts = all_points(banks)
        log(f"\n{tag}:")
        for name, e in pts.items():
            log("  " + fmt_row(name, e))
        results[tag] = dict(nominal=pts["nominal"]["total"],
                            pessimistic=pts["pessimistic corner"]["total"],
                            low_swing=pts["low-swing 0.5 V"]["total"],
                            points={k: v["total"] for k, v in pts.items()})
    torch.save(dict(results=results), artifacts.path("binary_energy"))
    log("\nsaved: binary-energy-results.pt")
    fh.close()


if __name__ == "__main__":
    main()
