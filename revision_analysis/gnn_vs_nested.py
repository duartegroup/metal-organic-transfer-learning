#!/usr/bin/env python3
"""The GNN control compared against NESTED S1 and TL, on matching protocols.

Why this exists. The §1.10 comparison table sets the nested GNN against the *published*
S1 and TL values, and keeps a second "matched" GNN arm (re-run with selection on the test
fold) so that like could be compared with like. That was correct while the paper reported
published numbers. It is no longer: after the Table 1 retraining the manuscript reports
nested values throughout, so the comparison the paper needs is nested-vs-nested, and the
matched arm's reason for existing is gone.

The guide anticipated this -- §1.10 carries the caveat "if the champions are re-run with
nested selection, this table must be rebuilt from those numbers with all three columns
nested". This is that rebuild.

Inputs, all local:
  results/gnn_control_pKi_final_t{5,6,7}.0.csv   nested GNN, per fold, seed-averaged
  results/table1/**/*summary.csv                 nested S1 and TL (Table 1 pKi config)

The GNN replaces the molecular representation used by S1, so the S1 arm of the Table 1
pKi configuration is the correct comparator: same folds, same thresholds, same endpoint,
same nested protocol, three seeds each.

Usage:
    python gnn_vs_nested.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = {"no_pretrain_encoder_task": "S1", "pretrain_ccsa_then_encoder_task": "TL"}
METRIC = "val/mcc_from_probs"
THRS = (5, 6, 7)

# What §1.10 currently prints, for the side-by-side.
PUBLISHED = {"S1": {5: 0.475, 6: 0.467, 7: 0.664},
             "TL": {5: 0.608, 6: 0.583, 7: 0.715}}
GNN_MATCHED = {5: 0.486, 6: 0.594, 7: 0.643}


def load_gnn(results: Path) -> dict:
    """{(threshold, fold): mcc} for the nested GNN."""
    out = {}
    for t in THRS:
        p = results / f"gnn_control_pKi_final_t{t}.0.csv"
        for r in csv.DictReader(open(p)):
            out[(t, int(r["fold"]))] = float(r["mcc"])
    return out


def load_table1(runs: Path) -> dict:
    """{(threshold, fold, arm): mcc} for pKi, averaged over seeds."""
    acc = defaultdict(list)
    for f in glob.glob(str(runs / "**/*summary.csv"), recursive=True):
        m = re.search(r"/(pic50|pk)_t(\d)_.*_s(\d+)/", f)
        if not m or m.group(1) != "pk":
            continue
        t = int(m.group(2))
        for r in csv.DictReader(open(f)):
            arm = ARMS.get(r["method"])
            if not arm or not r.get(METRIC):
                continue
            try:
                fold = int(r["fold"])
            except ValueError:
                continue
            acc[(t, fold, arm)].append(float(r[METRIC]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=here / "results")
    ap.add_argument("--runs", type=Path, default=here / "results/table1")
    args = ap.parse_args()

    gnn, t1 = load_gnn(args.results), load_table1(args.runs)
    folds = sorted({f for (_, f) in gnn})
    print(f"GNN folds per threshold: {folds}; Table 1 pKi cells: {len(t1)}\n")

    print("=" * 78)
    print("ALL THREE ARMS NESTED — pKi, mean test MCC over five folds")
    print("=" * 78)
    print(f"{'theta':>6}{'GNN':>9}{'S1':>9}{'TL':>9}   "
          f"{'GNN-S1':>8}{'GNN-TL':>8}    {'(old table: GNN matched / S1 pub / TL pub)':>12}")
    for t in THRS:
        g = np.mean([gnn[(t, f)] for f in folds])
        s = np.mean([t1[(t, f, "S1")] for f in folds])
        l = np.mean([t1[(t, f, "TL")] for f in folds])
        print(f"{t:>6}{g:>9.3f}{s:>9.3f}{l:>9.3f}   {g-s:>+8.3f}{g-l:>+8.3f}"
              f"      {GNN_MATCHED[t]:.3f} / {PUBLISHED['S1'][t]:.3f} / "
              f"{PUBLISHED['TL'][t]:.3f}")
    print()

    for other in ("S1", "TL"):
        a = np.array([gnn[(t, f)] for t in THRS for f in folds])
        b = np.array([t1[(t, f, other)] for t in THRS for f in folds])
        d = a - b
        try:
            p = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            p = float("nan")
        rng = np.random.default_rng(0)
        boot = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"GNN − {other} (nested, paired over {len(d)} fold×threshold cells): "
              f"mean {d.mean():+.3f}, median {np.median(d):+.3f}, "
              f"95% CI [{lo:+.3f}, {hi:+.3f}], GNN wins {int((d>0).sum())}/{len(d)}, "
              f"Wilcoxon p={p:.4f}")
    print()
    print("Per-fold detail (nested):")
    print(f"{'theta':>6}{'fold':>5}{'GNN':>9}{'S1':>9}{'TL':>9}")
    for t in THRS:
        for f in folds:
            print(f"{t:>6}{f:>5}{gnn[(t,f)]:>9.3f}{t1[(t,f,'S1')]:>9.3f}"
                  f"{t1[(t,f,'TL')]:>9.3f}")


if __name__ == "__main__":
    main()
