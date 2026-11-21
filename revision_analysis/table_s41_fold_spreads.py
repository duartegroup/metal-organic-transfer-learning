#!/usr/bin/env python3
"""Per-fold accuracy, F1 and MCC for the Table S41 arms, so the whole table can quote
fold spreads.

Why this exists. Table S41 as first drafted quoted the S1 and TL spread across the three
seeds (folds averaged within a seed), while the baseline column it sits beside quotes the
spread across the five folds. Those are different quantities, and the seed spread is much
the smaller of the two because between-fold variation has been averaged out of it: at
pKi theta=6 the TL MCC reads 0.509 +/- 0.009 across seeds against 0.51 +/- 0.27 across
folds, beside a baseline at 0.50 +/- 0.14. Printed side by side that invites the reader
to conclude the neural arms are an order of magnitude more stable than the baseline,
which is not what the data say.

`table1_fold_metrics_nested.csv` already holds the seed-averaged per-fold MCC, but not
accuracy or F1. This script re-extracts all three from the run summaries, builds the
per-fold values the same way (average over seeds within a fold), and reports mean +/- s.d.
across the five folds. It cross-checks its own MCC against the existing file, which is
what makes the extraction trustworthy rather than merely plausible.

Metric convention is `*_from_probs` throughout (decided 2026-11-12).

Usage:
    python table_s41_fold_spreads.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = {"no_pretrain_encoder_task": "S1", "pretrain_ccsa_then_encoder_task": "TL"}
COLS = {"Accuracy": "val/accuracy_from_probs", "F1": "val/f1_from_probs",
        "MCC": "val/mcc_from_probs"}
METRICS = ["Accuracy", "F1", "MCC"]
THRS = (5, 6, 7)
FOLDS = (1, 2, 3, 4, 5)


def load_per_fold(runs: Path) -> dict:
    """{(endpoint, threshold, fold, arm, metric): seed-averaged value}"""
    acc = defaultdict(list)
    for f in glob.glob(str(runs / "**/*summary.csv"), recursive=True):
        m = re.search(r"/(pic50|pk)_t(\d)_.*_s(\d+)/", f)
        if not m:
            continue
        endpoint = "pIC50" if m.group(1) == "pic50" else "pKi"
        thr = int(m.group(2))
        for row in csv.DictReader(open(f)):
            arm = ARMS.get(row["method"])
            if not arm:
                continue
            try:
                fold = int(row["fold"])
            except ValueError:
                continue
            for label, col in COLS.items():
                if row.get(col):
                    acc[(endpoint, thr, fold, arm, label)].append(float(row[col]))
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=here / "results/table1")
    ap.add_argument("--check", type=Path,
                    default=here / "results/table1_fold_metrics_nested.csv")
    ap.add_argument("--out", type=Path,
                    default=here / "results/table_s41_fold_spreads.csv")
    args = ap.parse_args()

    pf = load_per_fold(args.runs)

    # ---- completeness -------------------------------------------------------------
    missing = [(e, t, f, a, m) for e in ("pKi", "pIC50") for t in THRS for f in FOLDS
               for a in ("S1", "TL") for m in METRICS if (e, t, f, a, m) not in pf]
    seeds = {n for _, n in pf.values()}
    print(f"cells found: {len(pf)} of {2*3*5*2*3} expected; seeds per cell: {sorted(seeds)}")
    if missing:
        print(f"  MISSING {len(missing)}: {missing[:6]}")

    # ---- cross-check MCC against the existing per-fold file -----------------------
    ref = {}
    for r in csv.DictReader(open(args.check)):
        ref[(r["endpoint"], int(float(r["threshold"])), int(r["fold"]),
             r["model"])] = float(r["mcc"])
    diffs = [abs(pf[(e, t, f, a, "MCC")][0] - v)
             for (e, t, f, a), v in ref.items() if (e, t, f, a, "MCC") in pf]
    print(f"cross-check vs {args.check.name}: {len(diffs)} cells, "
          f"max |delta| = {max(diffs):.6f}" if diffs else "cross-check: no overlap")

    # ---- fold spreads --------------------------------------------------------------
    rows = []
    print("\nmean +/- s.d. ACROSS THE FIVE FOLDS (seed-averaged within each fold)")
    for e in ("pKi", "pIC50"):
        print(f"\n{e}")
        print(f"{'metric':10}" + "".join(f"{'S1 t'+str(t):>18}" for t in THRS)
              + "".join(f"{'TL t'+str(t):>18}" for t in THRS))
        for lab in METRICS:
            cells = []
            for arm in ("S1", "TL"):
                for t in THRS:
                    v = np.array([pf[(e, t, f, arm, lab)][0] for f in FOLDS])
                    cells.append(f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}")
                    # eight decimals: the table is typeset at three, and rounding a
                    # shorter record again drifts on the half-way cases (pIC50 theta=6 S1 F1
                    # has s.d. 0.21550008, which must not become 0.215).
                    rows.append(dict(endpoint=e, threshold=t, arm=arm, metric=lab,
                                     mean_over_folds=round(float(v.mean()), 8),
                                     sd_across_folds=round(float(v.std(ddof=1)), 8),
                                     n_folds=len(v)))
            print(f"{lab:10}" + "".join(f"{c:>18}" for c in cells))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
