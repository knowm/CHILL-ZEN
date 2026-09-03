"""Stand their FID up and prove it. Nothing downstream counts until this
passes.

Three steps.

*The threshold probe.* Their loader binarizes at 0.1, ours at the
midpoint. Because the whole comparison turns on that, the threshold is
confirmed from their statistics as well as from their source: the mean
2048-dimensional feature of 2000 train images at four thresholds,
compared against their shipped mu. Expected -- 0.1 wins by a factor of
8.1 in relative L2 over the next candidate (0.1: 0.01407; 0.3: 0.11370;
0.5: 0.22605; 0.0: 0.24048).

*Check 1, reference self-consistency.* The 60,000-image train split
from the canonical idx files, binarized at 0.1, scored against their
shipped mu and sigma with their own code. Expected FID 0.000039, with
maximum absolute differences of 6.0e-04 in mu and 3.0e-04 in sigma.
Agreement at that level across a 2048x2048 covariance pins down the
reference identity, the threshold, the data order, the scaling, the
resize, the extractor and the statistics, all at once.

*Check 2, the held-out floor.* The test split at the same threshold:
2.142 at n = 5120 and 1.194 at n = 10,000. Real held-out data does not
score zero at these sample sizes, which is the scale on which every
other number here should be read.

The same 60,000 train images binarized at the midpoint instead score
29.75 -- worse than their generator. That number is reported in the
paper and it comes out of this script.

Cost: about 2.5 hours (four feature passes over 60,000 and 10,000
images at roughly 25 images/s). Features are cached, so re-scoring is
free.

    .venv-dtm/bin/python experiments/head_to_head/01_gate.py
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dtm_fid as ev                                        # noqa: E402

PROBE_N = 2000
PROBE_THRESH = [0.0, 0.1, 0.3, 0.5]
SELF_TOL = 0.5


def main():
    log, fh = ev.make_log("01_gate")
    log(f"\n=== the gate ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    commit = (ev.HERE / "vendor" / "COMMIT")
    if commit.exists():
        log(f"vendored evaluation code at commit "
            f"{commit.read_text().strip()}")
    log(f"reference  {ev.REF_BW_FASHION.name} "
        f"sha256 {ev.sha256(ev.REF_BW_FASHION)[:16]}")
    mu_r, sig_r = ev.ref_stats()
    log(f"reference  mu {mu_r.shape} {mu_r.dtype}, sigma {sig_r.shape} "
        f"{sig_r.dtype}; trace {np.trace(sig_r):.4f}")

    xtr, _ = ev.fashion("train")
    xte, _ = ev.fashion("test")
    log(f"data       train {xtr.shape} test {xte.shape}, canonical idx files")
    res = {}

    log(f"\n-- threshold probe (mean feature against their mu, "
        f"n = {PROBE_N} train images) --")
    log(f"  {'thresh':>7} {'on-frac':>8} {'rel L2':>10} {'cosine':>9}")
    probe = {}
    for th in PROBE_THRESH:
        imgs = ev.binarize(xtr[:PROBE_N], th)
        act = ev.their_features(imgs, tag=f"probe-tr{PROBE_N}-th{th}")
        mu = act.mean(0)
        rel = float(np.linalg.norm(mu - mu_r) / np.linalg.norm(mu_r))
        cos = float(mu @ mu_r / (np.linalg.norm(mu) * np.linalg.norm(mu_r)))
        probe[th] = (rel, cos, float(imgs.mean()))
        log(f"  {th:>7.2f} {imgs.mean():>8.4f} {rel:>10.5f} {cos:>9.6f}")
    best = min(probe, key=lambda t: probe[t][0])
    runner = sorted(probe, key=lambda t: probe[t][0])[1]
    log(f"  best {best} (rel L2 {probe[best][0]:.5f}); next {runner} "
        f"({probe[runner][0]:.5f}) — "
        f"{probe[runner][0] / probe[best][0]:.1f}x worse")
    if best != ev.THEIR_THRESH:
        log(f"  NOTE: the probe picks {best} where their source says "
            f"{ev.THEIR_THRESH}; check 1 decides")
    res["probe"] = np.array([[t, *probe[t]] for t in PROBE_THRESH])

    log(f"\n-- check 1: the 60,000 train images at threshold {best}, "
        "against their reference --")
    t0 = time.time()
    act_tr = ev.their_features(ev.binarize(xtr, best), tag=f"train-th{best}")
    mu, sig = ev.stats_of(act_tr)
    fid_self, t1, t2 = ev.frechet(mu, sig, mu_r, sig_r)
    log(f"  FID = {fid_self:.6f}   [mean term {t1:.6f}, covariance term "
        f"{t2:.6f}]   ({time.time() - t0:.0f}s)")
    log(f"  mu     max abs diff {np.abs(mu - mu_r).max():.3e}")
    log(f"  sigma  max abs diff {np.abs(sig - sig_r).max():.3e}")
    res["fid_self"] = fid_self
    res["thresh"] = best
    gate1 = fid_self < SELF_TOL
    log(f"  CHECK 1: {'PASS' if gate1 else 'FAIL'} (bar: under {SELF_TOL})")

    if best != ev.OUR_THRESH:
        act_ours = ev.their_features(ev.binarize(xtr, ev.OUR_THRESH),
                                      tag=f"train-th{ev.OUR_THRESH}")
        mo, so = ev.stats_of(act_ours)
        fid_ours, _, _ = ev.frechet(mo, so, mu_r, sig_r)
        log(f"  for contrast: the SAME 60,000 images binarized at "
            f"{ev.OUR_THRESH} score {fid_ours:.2f} against their "
            f"reference — worse than their generator")
        res["fid_train_ourthresh"] = fid_ours

    log(f"\n-- check 2: the held-out floor at threshold {best} --")
    act_te = ev.their_features(ev.binarize(xte, best), tag=f"test-th{best}")
    floors = {}
    for n in (ev.THEIR_N_GEN, 10000):
        mu_n, sig_n = ev.stats_of(act_te[:n])
        f, _, _ = ev.frechet(mu_n, sig_n, mu_r, sig_r)
        floors[n] = f
        log(f"  FID(test[:{n}]) = {f:.3f}")
    res["floor_5120"] = floors[ev.THEIR_N_GEN]
    res["floor_10000"] = floors[10000]
    gate2 = np.isfinite(floors[10000]) and floors[10000] > 0

    log("\n-- gate --")
    log(f"  check 1 (self-consistency, under {SELF_TOL}): {fid_self:.6f}  "
        f"{'PASS' if gate1 else 'FAIL'}")
    log(f"  check 2 (held-out floor, finite and positive): "
        f"{floors[10000]:.3f}  {'PASS' if gate2 else 'FAIL'}")
    log(f"  GATE: {'PASS' if (gate1 and gate2) else 'FAIL'}")
    log(f"  real-data floor at their n ({ev.THEIR_N_GEN}): "
        f"{floors[ev.THEIR_N_GEN]:.3f}")
    log(f"  the bar to beat: {ev.PRIMARY_BAR} (their 8-step DTM)")

    np.savez(ev.HERE / "gate-results.npz", **res)
    log("saved: gate-results.npz")
    _figure(probe, fid_self, floors, best, res.get("fid_train_ourthresh"))
    fh.close()


def _figure(probe, fid_self, floors, best, fid_ours):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figu, ax = plt.subplots(1, 2, figsize=(11, 4))
    ts = sorted(probe)
    ax[0].plot(ts, [probe[t][0] for t in ts], "o-", color="#b03030")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("binarization threshold")
    ax[0].set_ylabel("relative L2, mean feature against their mu")
    ax[0].set_title(f"their reference was binarized at {best}")
    ax[0].grid(alpha=0.3)

    names, vals, cols = ["train\n(check 1)"], [max(fid_self, 1e-4)], ["#2a7f2a"]
    if fid_ours is not None:
        names.append(f"train at {ev.OUR_THRESH}\n(the midpoint)")
        vals.append(fid_ours)
        cols.append("#888888")
    names += ["test 5,120\n(floor at their n)", "test 10,000\n(floor)"]
    vals += [floors[ev.THEIR_N_GEN], floors[10000]]
    cols += ["#3060b0", "#3060b0"]
    ax[1].bar(range(len(vals)), vals, color=cols)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(vals)))
    ax[1].set_xticklabels(names, fontsize=8)
    ax[1].axhline(ev.PRIMARY_BAR, color="#b03030", ls="--",
                  label=f"the bar to beat ({ev.PRIMARY_BAR})")
    ax[1].set_ylabel("FID, their protocol")
    ax[1].set_title("real data scored under their harness")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax[1].text(i, v * 1.15, f"{v:.3g}", ha="center", fontsize=8)
    figu.suptitle("The evaluation harness, standing up and checked",
                  fontsize=11)
    figu.tight_layout()
    figu.savefig(ev.HERE / "gate.png", dpi=140)
    print("saved: gate.png", flush=True)


if __name__ == "__main__":
    main()
