"""
abstain_curve.py
Task A2.2 Part 1: choose the abstain threshold tau_conf from data, before it touches the agent.

confidence = max(p_angry, p_happy, p_sad). For each tau, a clip is kept if confidence >= tau.
Per split (val, test): coverage, accuracy on the kept clips, and the happy->distress rate
(share of kept happy clips with p_distress >= 0.5).

Rule, fixed in advance: tau_conf = the tau with the highest val accuracy among taus with
val coverage >= 0.70 (ties -> the smallest tau, i.e. the most coverage).

Outputs: results/metrics/abstain_curve.csv, results/figures/abstain_coverage.png
         (with a non-default --min-coverage, e.g. 0.85: abstain_curve_cov085.csv / abstain_coverage_cov085.png,
          so the committed rule-A outputs are never overwritten)
Usage: python scripts/abstain_curve.py [--min-coverage 0.85]
"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Config
CODE_DIR = Path(__file__).resolve().parent.parent
PROBS_CSV = CODE_DIR / "results" / "metrics" / "clip_probabilities.csv"
OUT_CSV = CODE_DIR / "results" / "metrics" / "abstain_curve.csv"
OUT_FIG = CODE_DIR / "results" / "figures" / "abstain_coverage.png"
CLASSES = ["angry", "happy", "sad"]
TAUS = np.round(np.arange(0.34, 0.95 + 1e-9, 0.01), 2)
DEFAULT_MIN_COVERAGE = 0.70   # rule A (shipped tau_conf = 0.82); rule B uses 0.85
DISTRESS_CUT = 0.5


def stats(df, tau):
    kept = df[df["confidence"] >= tau]
    happy = df[df["label"] == "happy"]
    kept_happy = kept[kept["label"] == "happy"]
    return {
        "n": len(df),
        "n_kept": len(kept),
        "coverage": len(kept) / len(df),
        "accuracy": (kept["pred"] == kept["label"]).mean() if len(kept) else np.nan,
        "n_happy_kept": len(kept_happy),
        "happy_coverage": len(kept_happy) / len(happy),
        "happy_to_distress": (kept_happy["p_distress"] >= DISTRESS_CUT).mean() if len(kept_happy) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description="Coverage vs accuracy curve; picks tau_conf on val.")
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                    help=f"minimum val coverage for an eligible tau (default {DEFAULT_MIN_COVERAGE})")
    args = ap.parse_args()
    min_cov = args.min_coverage
    suffix = "" if np.isclose(min_cov, DEFAULT_MIN_COVERAGE) else f"_cov{round(min_cov * 100):03d}"
    out_csv = OUT_CSV.with_stem(OUT_CSV.stem + suffix)
    out_fig = OUT_FIG.with_stem(OUT_FIG.stem + suffix)

    df = pd.read_csv(PROBS_CSV)
    probs = df[[f"p_{c}" for c in CLASSES]].to_numpy()
    df["confidence"] = probs.max(axis=1)
    df["pred"] = np.array(CLASSES)[probs.argmax(axis=1)]

    rows = []
    for tau in [0.0, *TAUS]:
        row = {"tau": tau}
        for split in ["val", "test"]:
            for k, v in stats(df[df["split"] == split], tau).items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    curve = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    curve.round(4).to_csv(out_csv, index=False)

    # Selection rule (val only)
    grid = curve[curve["tau"] > 0]
    eligible = grid[grid["val_coverage"] >= min_cov]
    best = eligible.loc[eligible["val_accuracy"].idxmax()]   # idxmax -> first (smallest) tau on ties
    tau_conf = float(best["tau"])
    base = curve[curve["tau"] == 0].iloc[0]

    print(f"Rule: max val accuracy s.t. val coverage >= {min_cov:.2f}  ->  tau_conf = {tau_conf:.2f}\n")
    cols = ["coverage", "accuracy", "happy_coverage", "happy_to_distress"]
    print(f"{'':16s}" + "".join(f"{c:>19s}" for c in cols))
    for name, r in [("tau=0 (no abst)", base), (f"tau={tau_conf:.2f}", best)]:
        for split in ["val", "test"]:
            vals = "".join(f"{r[f'{split}_{c}']:19.3f}" for c in cols)
            print(f"{name:12s} {split:4s}{vals}   kept {int(r[f'{split}_n_kept'])}/{int(r[f'{split}_n'])}")

    # Figure: accuracy vs coverage
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, color in [("val", "tab:blue"), ("test", "tab:orange")]:
        ax.plot(curve[f"{split}_coverage"], curve[f"{split}_accuracy"], marker=".", color=color, label=split)
        ax.scatter(best[f"{split}_coverage"], best[f"{split}_accuracy"], s=120, facecolors="none",
                   edgecolors=color, linewidths=2, zorder=3)
    ax.axvline(min_cov, color="grey", ls=":", lw=1, label=f"min val coverage {min_cov:.2f}")
    ax.annotate(f"tau_conf = {tau_conf:.2f}", (best["val_coverage"], best["val_accuracy"]),
                xytext=(-12, -18), textcoords="offset points", ha="right")
    ax.set_xlabel("coverage (share of clips kept)")
    ax.set_ylabel("accuracy on kept clips")
    ax.set_title("Abstain: coverage vs accuracy (XGBoost, top-1 confidence)")
    ax.invert_xaxis()   # left = keep everything, right = abstain more
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"\nSaved {out_csv}\nSaved {out_fig}")


if __name__ == "__main__":
    main()
