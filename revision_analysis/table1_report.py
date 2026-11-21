#!/usr/bin/env python3
"""Replacement values for manuscript Table S41, Figure 5 and the abstract.

These come from the two configurations the paper actually reports (Table 1), retrained
under nested evaluation across three seeds -- not from the twelve per-cell champions.
See REVISION_GUIDE, the warning block before Part 1, for why that distinction matters.

Reported metric is `val/mcc_from_probs` (decision of 2026-11-12, REVISION_GUIDE 0.3a),
with accuracy and F1 from the matching `_from_probs` columns so the whole table uses one
convention. Means are taken over folds within a seed, then over seeds; the quoted
spread is the across-seed standard deviation.

Usage:
    python table1_report.py --runs /path/to/runs/table1
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

# Published Table S41 MCCs, for the side-by-side
PUBLISHED = {
    ("pIC50", "S1"): {5: 0.46, 6: 0.61, 7: 0.58},
    ("pIC50", "TL"): {5: 0.43, 6: 0.62, 7: 0.52},
    ("pKi", "S1"): {5: 0.48, 6: 0.47, 7: 0.66},
    ("pKi", "TL"): {5: 0.61, 6: 0.58, 7: 0.72},
}
ARMS = {"no_pretrain_encoder_task": "S1", "pretrain_ccsa_then_encoder_task": "TL"}
METRICS = [("val/mcc_from_probs", "MCC"),
           ("val/accuracy_from_probs", "Accuracy"),
           ("val/f1_from_probs", "F1")]


def load(root: Path) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(root / "**/*summary.csv"), recursive=True):
        m = re.search(r"/(pic50|pk)_t(\d)_.*_s(\d+)/", f)
        if not m:
            continue
        d = pd.read_csv(f)
        d["endpoint"] = "pIC50" if m.group(1) == "pic50" else "pKi"
        d["threshold"] = int(m.group(2))
        d["seed"] = int(m.group(3))
        rows.append(d)
    if not rows:
        raise SystemExit(f"no summary files under {root}")
    d = pd.concat(rows, ignore_index=True)
    d = d[d.method.isin(ARMS)].copy()
    d["arm"] = d.method.map(ARMS)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path.home() / "matthijs/runs/table1")
    args = ap.parse_args()
    d = load(args.runs)

    n = d.groupby(["endpoint", "threshold", "arm", "seed"]).ngroups
    print(f"loaded {len(d)} fold-rows across {n} (endpoint, threshold, arm, seed) groups")
    print("expected 36 = 2 endpoints x 3 thresholds x 2 arms x 3 seeds\n")

    for col, label in METRICS:
        if col not in d.columns:
            print(f"  [{label}: column {col} absent]")
            continue
        per_seed = d.groupby(["endpoint", "threshold", "arm", "seed"])[col].mean()
        agg = per_seed.groupby(["endpoint", "threshold", "arm"]).agg(["mean", "std"])
        print("=" * 66)
        print(f"{label} — nested, mean ± s.d. across three seeds")
        print("=" * 66)
        print(f"{'':16s} {'θ5':>16} {'θ6':>16} {'θ7':>16}")
        for endpoint in ("pIC50", "pKi"):
            for arm in ("S1", "TL"):
                cells = []
                for thr in (5, 6, 7):
                    try:
                        r = agg.loc[(endpoint, thr, arm)]
                        cells.append(f"{r['mean']:.3f} ± {r['std']:.3f}")
                    except KeyError:
                        cells.append("—")
                print(f"{endpoint + ' ' + arm:16s} " + " ".join(f"{c:>16}" for c in cells))
        if label == "MCC":
            print("\n  against the published Table S41 values:")
            for (endpoint, arm), pub in PUBLISHED.items():
                deltas = []
                for thr in (5, 6, 7):
                    try:
                        got = agg.loc[(endpoint, thr, arm), "mean"]
                        deltas.append(f"{pub[thr]:.2f}→{got:.3f} ({got - pub[thr]:+.3f})")
                    except KeyError:
                        deltas.append("—")
                print(f"    {endpoint + ' ' + arm:10s} " + "  ".join(deltas))
            print("\n  TL − S1 per cell:")
            for endpoint in ("pIC50", "pKi"):
                parts = []
                for thr in (5, 6, 7):
                    try:
                        gap = (agg.loc[(endpoint, thr, "TL"), "mean"]
                               - agg.loc[(endpoint, thr, "S1"), "mean"])
                        pub_gap = PUBLISHED[(endpoint, "TL")][thr] - PUBLISHED[(endpoint, "S1")][thr]
                        parts.append(f"θ{thr} {gap:+.3f} (was {pub_gap:+.2f})")
                    except KeyError:
                        parts.append(f"θ{thr} —")
                print(f"    {endpoint:6s} " + "   ".join(parts))
        print()


if __name__ == "__main__":
    main()
