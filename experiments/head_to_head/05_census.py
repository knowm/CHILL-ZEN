"""Verify the instruction census for the arm that was scored, then
assemble the head-to-head table.

The energy rows only describe the scored arm if the recipe that
rendered it performs the reads the census counts. That is checked here
from the banks' address and lane counts rather than assumed, and the run stops if it does
not come out. Expected:

    stage                        lane reads
    backbone draw (hot)          128 x 16 = 2048
    patch read (hot)              98 x 16 = 1568
    backbone sweep (cold)        128 x 16 = 2048
    joint sweep (cold)           226 x 16 = 3616
    total                                  9280   over 226 addresses

Then it joins the FID of `03_score.py` to the energy of experiment 13
and prints the table of the paper's head-to-head section, with all
three energy accountings, including the one at which the deployed
system costs more per image than the DTM chain does.

Runs in the emulator environment (torch), not the JAX one.

    python experiments/head_to_head/05_census.py
"""
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import dtm_fid as ev                                        # noqa: E402
from chill_zen import artifacts, config                            # noqa: E402
from chill_zen.levels import S, load_patch_level, JointLevel       # noqa: E402

ROWS = [("one-level-16", 16), ("one-level-32", 32), ("one-level-64", 64),
        ("one-level-128", 128), ("two-level", "deployed")]


def main():
    log, fh = ev.make_log("05_census")
    log(f"\n=== census and head-to-head table "
        f"({time.strftime('%Y-%m-%d %H:%M')}) ===")

    pl = load_patch_level(config.BACKBONE, config.PATCH)
    jl = JointLevel(pl)
    NG, NPU, NU = pl.NG, pl.NPU, jl.NU
    log(f"counts read from the deployed levels: {NG} backbone addresses, "
        f"{NPU} patch addresses, {NU} joint targets")

    census = [
        ("backbone draw (hot)", NG * S,
         f"{NG} sequential steps x {S} lanes of the group being placed"),
        ("patch read (hot)", NPU * S,
         f"{NPU} patch addresses x {S} lanes, one parallel read"),
        ("backbone sweep (cold)", NG * S,
         f"{NG} addresses x {S} lanes, one cold parallel pass"),
        ("joint sweep (cold)", NU * S,
         f"{NU} addresses x {S} lanes, one cold parallel pass"),
        ("decode (cold)", 784,
         "784 output lanes, one cold parallel read (charged as a lane "
         "operation; runs host-side today)"),
    ]
    total = sum(n for _, n, _ in census)
    log(f"\n  {'stage':<26}{'lane reads':>12}   what")
    for name, n, what in census:
        log(f"  {name:<26}{n:>12}   {what}")
    log(f"  {'TOTAL':<26}{total:>12}")
    log(f"  addresses swept: {NG} backbone + {NPU} patch = {NG + NPU}")

    model = torch.load(artifacts.path("energy_model"))
    expected = sum(b["reads"] for b in model["census"])
    if total != expected or NG + NPU != NU:
        log(f"\n  MISMATCH against the energy model's census ({expected} "
            "reads). The energy rows do NOT describe the scored arm; stop "
            "here and recount before reporting any energy number.")
        fh.close()
        return
    log(f"\n  matches the energy model's census ({expected} reads over "
        f"{NU} addresses), so the modeled energy describes the scored arm "
        "unchanged.")

    frontier = torch.load(artifacts.path("energy_frontier"))
    scores = json.loads((HERE / "score-results.json").read_text())
    th = ev.THEIR_THRESH

    log(f"\n-- the frontier, their protocol (their code, their reference, "
        f"their {th} binarization, n = {ev.THEIR_N_GEN}) --")
    log(f"  {'arm':<16}{'FID':>8}{'div':>8}{'nominal J':>12}"
        f"{'low-swing':>13}{'pessimistic':>12}{'vs 24.9':>10}")
    pts = []
    for name, key in ROWS:
        k = f"{name}@{th}"
        if k not in scores or key not in frontier:
            continue
        f = scores[k]["theirs"]
        e = frontier[key]
        pts.append((name, e["lo"], f, e["low_swing"], e["hi"]))
        log(f"  {name:<16}{f:>8.2f}{scores[k]['div']:>8.4f}"
            f"{e['lo']:>12.2e}{e['low_swing']:>13.2e}{e['hi']:>12.2e}"
            f"{f - ev.PRIMARY_BAR:>+10.2f}")
    log(f"  {'their DTM, 8':<16}{ev.PRIMARY_BAR:>8.2f}{'—':>8}"
        f"{ev.DTM_CHAIN[-1][1]:>12.2e}{'—':>13}{'—':>12}")
    for tag, key in (("real train images", "real-train"),
                     ("codec ceiling", "codec-ceiling")):
        k = f"{key}@{th}"
        if k in scores:
            log(f"  {tag:<16}{scores[k]['theirs']:>8.2f}"
                f"{scores[k]['div']:>8.4f}   (not a generator)")

    best = min(pts, key=lambda p: p[2])
    log("\n-- the result --")
    log(f"  {best[0]}: FID {best[2]:.2f} against their {ev.PRIMARY_BAR} "
        f"({ev.PRIMARY_BAR - best[2]:.2f} better), at {best[1]:.2e} J per "
        f"image at the nominal point against their "
        f"{ev.DTM_CHAIN[-1][1]:.3e} — a factor of "
        f"{ev.DTM_CHAIN[-1][1] / best[1]:.0f}.")
    log(f"  low-swing select {best[3]:.2e} J "
        f"({ev.DTM_CHAIN[-1][1] / best[3]:.0f}x); pessimistic corner "
        f"{best[4]:.2e} J ({ev.DTM_CHAIN[-1][1] / best[4]:.1f}x).")
    log("  All three belong in the report.")
    fh.close()


if __name__ == "__main__":
    main()
