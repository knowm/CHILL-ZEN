"""Score the frozen arms with their FID. The decisive step.

Reads what `02_generate.py` rendered and scores it with their code
against two references:

  their reference   the shipped bw_fashion_mnist_train.npz -- 60,000
                    train images binarized at 0.1, their convention,
                    confirmed by `01_gate.py`
  midpoint          the SAME 60,000 images binarized at 0.5, features
                    taken with their extractor. Their protocol in every
                    respect except the binarization, and reported only
                    to show what the convention alone is worth

Diversity is the within-class mean pairwise pixel disagreement, 512
pairs per class, measured on the same binarized images that were
scored. It travels with the FID because a mode-collapsed generator can
score well on FID and this one is under real: the number belongs beside
the headline rather than in a footnote.

Expected, against their reference at their 0.1, n = 5120:

    arm                  FID    on-fraction   diversity
    real train images   1.907        0.4492      0.1806
    codec ceiling      14.205        0.4949      0.1634
    two-level          18.931        0.4747      0.1540
    one-level-128      22.320        0.5083      0.1583
    one-level-64       23.867        0.5369      0.1512
    one-level-32       28.752        0.5347      0.1420
    one-level-16       46.218        0.5603      0.1262
    their 8-step DTM    24.90             --          --

Neither the control nor the ceiling is a generator, and neither is
eligible for the bar.

Cost: about an hour on first run; features are cached, so re-scoring is
free.

    .venv-dtm/bin/python experiments/head_to_head/03_score.py
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

ARMS = ["two-level", "one-level-128", "one-level-64", "one-level-32",
        "one-level-16", "codec-ceiling", "real-train"]
NOT_A_SYSTEM = {"real-train", "codec-ceiling"}


def diversity(b, per=512):
    """Within-class mean pairwise pixel disagreement, 512 pairs a class."""
    rng = np.random.default_rng(0)
    out = []
    for c in range(len(b) // per):
        blk = b[c * per:(c + 1) * per].reshape(per, -1)
        i, j = rng.integers(0, per, 512), rng.integers(0, per, 512)
        m = i != j
        out.append(np.abs(blk[i[m]] - blk[j[m]]).mean())
    return float(np.mean(out))


def main():
    log, fh = ev.make_log("03_score")
    log(f"\n=== scoring ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    g0 = np.load(ev.HERE / "gate-results.npz")
    th = float(g0["thresh"])
    floor = float(g0["floor_5120"])
    log(f"gate: threshold {th}; self-consistency {float(g0['fid_self']):.6f}; "
        f"real-data floor at n = {ev.THEIR_N_GEN} {floor:.3f}")

    refs = {"theirs": ev.ref_stats()}
    x, _ = ev.fashion("train")
    act = ev.their_features(ev.binarize(x, ev.OUR_THRESH),
                             tag=f"train-th{ev.OUR_THRESH}")
    refs["midpoint"] = ev.stats_of(act)
    del act
    log(f"references: theirs = their shipped statistics (60,000 train at "
        f"{th}); midpoint = the same 60,000 at {ev.OUR_THRESH}, their "
        "extractor")

    sets = {}
    for t in (th, ev.OUR_THRESH):
        for a in ARMS:
            p = ev.GEN / f"{a}.npy"
            if p.exists():
                im = np.load(p)
                sets[f"{a}@{t}"] = ((im > t).astype(np.float32), a, t)
    if not sets:
        raise SystemExit("no arms rendered — run 02_generate.py first")
    log(f"arms: {len(sets)} (arm, binarization) sets, {ev.THEIR_N_GEN} "
        "images each")

    res = {}
    log(f"\n  {'arm @ binarization':<26}{'on-frac':>9}{'div':>9}"
        f"{'vs theirs':>12}{'vs midpoint':>13}")
    for key, (im, a, t) in sets.items():
        f = ev.their_features(im, tag=f"{a}-th{t}")
        mu, sig = ev.stats_of(f)
        row = {rn: ev.frechet(mu, sig, mr, sr)[0]
               for rn, (mr, sr) in refs.items()}
        row["on"] = float(im.mean())
        row["div"] = diversity(im)
        res[key] = row
        log(f"  {key:<26}{row['on']:>9.4f}{row['div']:>9.4f}"
            f"{row['theirs']:>12.3f}{row['midpoint']:>13.3f}")

    log(f"\n-- the head-to-head: their code, their reference, their {th} "
        "binarization, their n --")
    log(f"   the bar is FID {ev.PRIMARY_BAR} at "
        f"{ev.DTM_CHAIN[-1][1]:.3e} J/sample (their 8-step DTM). "
        f"Real-data floor {floor:.3f}.")
    log(f"   {'arm':<18}{'FID':>9}{'on-frac':>10}{'div':>9}{'under bar':>11}")
    winners = []
    for a in ARMS:
        k = f"{a}@{th}"
        if k not in res:
            continue
        f = res[k]["theirs"]
        under = "YES" if (f < ev.PRIMARY_BAR and a not in NOT_A_SYSTEM) else ""
        if under:
            winners.append((a, f))
        log(f"   {a:<18}{f:>9.3f}{res[k]['on']:>10.4f}{res[k]['div']:>9.4f}"
            f"{under:>11}")

    log("\n-- verdict --")
    if winners:
        best = min(winners, key=lambda t_: t_[1])
        log(f"   {best[0]} scores {best[1]:.3f} against their "
            f"{ev.PRIMARY_BAR}; {len(winners)} arms clear the bar.")
    else:
        ce = res.get(f"codec-ceiling@{th}", {}).get("theirs")
        log(f"   no generator under {ev.PRIMARY_BAR}.")
        if ce is not None:
            log(f"   read the ceiling first: the codec's render of real "
                f"codes scores {ce:.3f}. " + (
                    "No generator through this codec can clear the bar; "
                    "the codec is the wall."
                    if ce >= ev.PRIMARY_BAR else
                    "The codec clears the bar on real codes, so the wall "
                    "is the draw, not the codec."))

    (ev.HERE / "score-results.json").write_text(json.dumps(res, indent=1))
    log("saved: score-results.json")
    fh.close()


if __name__ == "__main__":
    main()
