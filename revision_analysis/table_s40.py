#!/usr/bin/env python3
"""Replacement values for SI Table S40 — the model-family comparison.

Table S40 pools the five folds over the three activity thresholds and compares the
non-neural baseline (B), the from-scratch neural baseline (S1) and the transfer-learned
model (TL). It is the source of the "0.54 to 0.64" figure the reviewer quotes in comment
1, and of the "+33.5%" neural advantage over conventional machine learning.

**Why it has to be rebuilt.** Its S1 and TL columns are the published, non-nested values
that §1.9 replaces. Its B column does not change: §0.2e verified that the non-neural
baselines were nested throughout. So the published table compares a biased estimate
against an unbiased one, which §0.2 flagged at the time and this script now quantifies.

**What this script can and cannot do.** The S1 and TL columns, and the TL-vs-S1
comparison including its Wilcoxon p-value, are computed here from `results/table1/`.
The TL-vs-B and S1-vs-B *percentages* are computed against the published B means read
from the SI table at two decimal places. Their **p-values cannot be recomputed**, because
the per-fold values of the non-neural baseline are not deposited in the repository and
are not on the cluster — only the launcher scripts under
`extracted/launchers/results/supervised_learning_results/` survive. Re-running
`src/supervised_learning_src/supervised_learning.py` on aleph would produce them; it is a
CPU job and needs the target-domain HDF5s.

Usage:
    python table_s40.py
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
METRICS = [("val/accuracy_from_probs", "Accuracy"),
           ("val/f1_from_probs", "F1"),
           ("val/mcc_from_probs", "MCC")]
THRS = (5, 6, 7)

# Published Table S40, read from si_revised.pdf.
PUBLISHED_B = {"pKi": {"Accuracy": 0.79, "F1": 0.72, "MCC": 0.48},
               "pIC50": {"Accuracy": 0.74, "F1": 0.72, "MCC": 0.44}}
PUBLISHED_S1_TL = {"pKi": {"Accuracy": (0.83, 0.87), "F1": (0.76, 0.84),
                           "MCC": (0.54, 0.64)},
                   "pIC50": {"Accuracy": (0.80, 0.80), "F1": (0.78, 0.78),
                             "MCC": (0.55, 0.52)}}
PUBLISHED_TL_VS_B = {"pKi": {"Accuracy": "+9.7%", "F1": "+17.3%", "MCC": "+33.5%"},
                     "pIC50": {"Accuracy": "+8.5%", "F1": "+8.0%", "MCC": "+20.0%"}}


def load(runs: Path) -> dict:
    """{(endpoint, threshold, fold, arm, metric): mcc} averaged over seeds."""
    acc = defaultdict(list)
    for f in glob.glob(str(runs / "**/*summary.csv"), recursive=True):
        m = re.search(r"/(pic50|pk)_t(\d)_.*_s(\d+)/", f)
        if not m:
            continue
        e = "pIC50" if m.group(1) == "pic50" else "pKi"
        t = int(m.group(2))
        for r in csv.DictReader(open(f)):
            arm = ARMS.get(r["method"])
            if not arm:
                continue
            try:
                fold = int(r["fold"])
            except ValueError:
                continue
            for col, _ in METRICS:
                if r.get(col):
                    acc[(e, t, fold, arm, col)].append(float(r[col]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=here / "results/table1")
    ap.add_argument("--out", type=Path, default=here / "results/table_s40_nested.csv")
    args = ap.parse_args()
    avg = load(args.runs)

    rows = []
    print("TABLE S40 REPLACEMENT — pooled over three thresholds, 15 fold-pairs, nested")
    print("=" * 104)
    print(f"{'':10}{'B':>7}{'S1 pub':>8}{'S1 new':>9}{'TL pub':>8}{'TL new':>9}"
          f"{'TLvsS1 pub':>12}{'TLvsS1 new':>12}{'p':>9}{'S1vsB':>9}{'TLvsB':>9}"
          f"{'TLvsB pub':>11}")
    for e in ("pKi", "pIC50"):
        print(e)
        for col, lab in METRICS:
            s1 = np.array([avg[(e, t, f, "S1", col)] for t in THRS for f in range(1, 6)])
            tl = np.array([avg[(e, t, f, "TL", col)] for t in THRS for f in range(1, 6)])
            p = float(stats.wilcoxon(tl, s1).pvalue)
            b = PUBLISHED_B[e][lab]
            ps1, ptl = PUBLISHED_S1_TL[e][lab]
            r = dict(endpoint=e, metric=lab, B_published=b,
                     S1_published=ps1, S1_nested=round(s1.mean(), 3),
                     TL_published=ptl, TL_nested=round(tl.mean(), 3),
                     TL_vs_S1_published_pct=round((ptl - ps1) / ps1 * 100, 1),
                     TL_vs_S1_nested_pct=round((tl.mean() - s1.mean()) / s1.mean() * 100, 1),
                     TL_vs_S1_wilcoxon_p=round(p, 4),
                     S1_vs_B_nested_pct=round((s1.mean() - b) / b * 100, 1),
                     TL_vs_B_nested_pct=round((tl.mean() - b) / b * 100, 1),
                     TL_vs_B_published_pct=PUBLISHED_TL_VS_B[e][lab],
                     note="p-values against B need per-fold B, not deposited")
            rows.append(r)
            print(f"  {lab:8}{b:>7.2f}{ps1:>8.2f}{s1.mean():>9.3f}{ptl:>8.2f}"
                  f"{tl.mean():>9.3f}{(ptl-ps1)/ps1*100:>11.1f}%"
                  f"{(tl.mean()-s1.mean())/s1.mean()*100:>11.1f}%{p:>9.4f}"
                  f"{(s1.mean()-b)/b*100:>8.1f}%{(tl.mean()-b)/b*100:>8.1f}%"
                  f"{PUBLISHED_TL_VS_B[e][lab]:>11}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    print("""
THE HEADLINE. On MCC the neural models no longer beat the conventional baseline:
  pKi   B 0.48 : S1 0.434 (-9.6%) : TL 0.461 (-4.0%)   [published TL vs B: +33.5%]
  pIC50 B 0.44 : S1 0.399 (-9.3%) : TL 0.441 (+0.2%)   [published TL vs B: +20.0%]
This is what §0.2 predicted when it said the published table "puts a biased estimate
against an unbiased one". It is now measured.

Two things keep this from being the whole story, and both belong in the write-up:
  * B is a CHAMPION -- the best of its own configuration search -- while S1 and TL here
    are the single pre-specified Table 1 configuration. The published table had the same
    asymmetry, so this is not new, but it matters for how the sentence is phrased.
  * Both searches selected configurations on optimistic estimates. The neural search
    covered 432 configurations against a handful for B, so the neural numbers retain the
    LARGER configuration-selection advantage. Correcting that too would move the
    comparison further against the neural models, not back toward them.
""")


if __name__ == "__main__":
    main()
