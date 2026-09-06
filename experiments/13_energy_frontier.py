"""Per-image energy for each point on the one-level frontier.

Paper Sec. V B, Table VII and Fig. 9. The model of experiment 12
(chill_zen/energy_geom.py) evaluated for the backbone-only systems at
16, 32, 64 and 128 books, whose census is M x 16 hot autoregressive
reads with 6 + M spaces asserted, M x 16 cold sweep reads, and the 784
cold decode reads. The deployed two-level point is carried over from
experiment 12. Only counts enter, so the binary-trained arms of
experiment 22 share these numbers with their grayscale twins.

Expected, joules per image at the nominal point (pessimistic corner):

    16 books      4.9e-11  (1.6e-10)
    32 books      9.4e-11  (3.3e-10)
    64 books      2.1e-10  (7.7e-10)
    128 books     7.5e-10  (2.9e-09)
    two-level     2.0e-09  (8.1e-09)

Cost: seconds. Requires 12.

    python experiments/13_energy_frontier.py
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.energy_geom import all_points, fmt_row, one_level_census  # noqa: E402
from chill_zen.levels import parse_tag                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="*", default=config.FRONTIER)
    args = ap.parse_args()
    log, fh = artifacts.make_log("13_energy_frontier")

    out = {}
    for tag in args.points:
        M, _, _ = parse_tag(tag)
        pts = all_points(one_level_census(M))
        log(f"\n{M} books:")
        for name, e in pts.items():
            log("  " + fmt_row(name, e))
        out[M] = dict(M=M, lo=pts["nominal"]["total"],
                      hi=pts["pessimistic corner"]["total"],
                      low_swing=pts["low-swing 0.5 V"]["total"],
                      points={k: v["total"] for k, v in pts.items()})

    e12 = torch.load(artifacts.path("energy_model"))["totals"]
    out["deployed"] = dict(lo=e12["lo"], hi=e12["hi"],
                           low_swing=e12["low_swing"])
    log(f"\ndeployed two-level: nominal {e12['lo']:.3e} J  pessimistic "
        f"{e12['hi']:.3e} J  low-swing {e12['low_swing']:.3e} J")
    torch.save(out, artifacts.path("energy_frontier"))
    log(f"saved: {artifacts.path('energy_frontier').name}")
    fh.close()


if __name__ == "__main__":
    main()
