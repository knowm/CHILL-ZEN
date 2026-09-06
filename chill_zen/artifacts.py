"""Where the frozen artifacts are kept, and what they are called.

Every experiment reads and writes through here so that a run can be
pointed at a different directory (`CHILLZEN_ARTIFACTS`) without editing a
script. Names are the paper's, not a run identifier: what a file holds
should be readable from what it is called.
"""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = pathlib.Path(os.environ.get("CHILLZEN_ARTIFACTS", ROOT / "artifacts"))
FIGURES = pathlib.Path(os.environ.get("CHILLZEN_FIGURES", ROOT / "figures"))
LOGS = pathlib.Path(os.environ.get("CHILLZEN_LOGS", ROOT / "logs"))

NAMES = {
    "backbone_books": "backbone-books.pt",
    "backbone_codes": "backbone-codes.pt",
    "backbone_codec_results": "backbone-codec-results.pt",
    "backbone_banks": "backbone-banks.pt",
    "backbone_teach_results": "backbone-teach-results.pt",
    "patch_books": "patch-books.pt",
    "patch_codes": "patch-codes.pt",
    "patch_codec_results": "patch-codec-results.pt",
    "patch_banks": "patch-banks.pt",
    "patch_teach_results": "patch-teach-results.pt",
    "joint_pool": "joint-pool.pt",
    "joint_bank": "joint-bank.pt",
    "joint_teach_results": "joint-teach-results.pt",
    "backbone_sweep": "backbone-sweep-results.pt",
    "verdict": "verdict-states.pt",
    "draw_analysis": "draw-analysis-results.pt",
    "prefix_bank": "prefix-schedule-bank.pt",
    "prefix_verdict": "prefix-schedule-verdict.pt",
    "draw_ablations": "draw-ablations-results.pt",
    "reporting_results": "reporting-results.pt",
    "binary_backbone_books": "binary-backbone-books.pt",
    "binary_backbone_codes": "binary-backbone-codes.pt",
    "binary_codec_results": "binary-codec-results.pt",
    "binary_backbone_banks": "binary-backbone-banks.pt",
    "binary_patch_books": "binary-patch-books.pt",
    "binary_patch_codes": "binary-patch-codes.pt",
    "binary_patch_banks": "binary-patch-banks.pt",
    "binary_joint_bank": "binary-joint-bank.pt",
    "binary_verdict": "binary-verdict-results.pt",
    "binary_energy": "binary-energy-results.pt",
    "comparator_sweep": "comparator-sweep-results.pt",
    "binary_comparator_sweep": "binary-comparator-sweep-results.pt",
    "binary_comparator_sweep_ext": "binary-comparator-sweep-ext-results.pt",
    "critic": "critic.pt",
    "operating_point": "operating-point.pt",
    "energy_model": "energy-model.pt",
    "energy_frontier": "energy-frontier.pt",
    "sneak_paths": "sneak-paths.pt",
}


def path(key):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / NAMES[key]


def figure(name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name


def make_log(name):
    """Tee a log file next to the artifacts; returns (log, close)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(LOGS / f"{name}.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        print(msg, file=fh, flush=True)
    return log, fh
