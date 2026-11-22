#!/usr/bin/env python3
"""Table S40, fully rebuilt: B, S1 and TL all nested, with the paired tests.

This supersedes `table_s40.py`, which could compute the percentages but not the tests
against the non-neural baseline B, because B's per-fold values were not deposited
anywhere. They now exist: the two B champion configurations named in
`results/statistical_tests/champion/champion_config.json` were re-run on aleph under the
published nested search protocol (job 797788), and the per-fold values are in
`results/baseline_b_per_fold.csv`.

    pIC50 champion B: SOAP / inhibitor_protein / standard scaler / target SMOTE / no PCA
                      -> KNN
    pKi   champion B: MACE / inhibitor / no scaler / no SMOTE / PCA 50
                      -> RandomForest

**The protocol is genuinely nested**, which is worth recording because §0.2e asserted it
from a reading of the code and this is the direct check. In `run_hyperparameter_search`,
each outer fold holds out its test fold, hyperparameters are chosen by RandomizedSearchCV
over a PredefinedSplit of the *remaining* folds only, and the stored MCC is then computed
on the held-out outer fold. (By contrast the `--evaluate-from-file` path is NOT nested:
it picks `best_params` by `idxmax` over the per-fold test MCC. The published B numbers
come from the search path -- the reproduction below confirms it.)

Table S40's design, reproduced here: pool the five folds over the three thresholds into
15 fold-pairs per endpoint; report relative differences with Holm-corrected Wilcoxon
signed-rank p-values over the two planned comparisons (TL vs B, TL vs S1); report S1 vs B
as a difference only.

Usage:
    python table_s40_full.py
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
NEURAL_COL = {"Accuracy": "val/accuracy_from_probs", "F1": "val/f1_from_probs",
              "MCC": "val/mcc_from_probs"}
B_COL = {"Accuracy": "accuracy", "F1": "f1", "MCC": "mcc"}
METRICS = ["Accuracy", "F1", "MCC"]
THRS = (5, 6, 7)

# Published Table S40, from si_revised.pdf.
PUB = {
    "pKi": {"Accuracy": dict(B=0.79, S1=0.83, TL=0.87, TLvB="+9.7% (0.007)",
                             TLvS1="+3.8% (0.018)", S1vB="+5.7%"),
            "F1": dict(B=0.72, S1=0.76, TL=0.84, TLvB="+17.3% (0.002)",
                       TLvS1="+10.8% (0.008)", S1vB="+5.9%"),
            "MCC": dict(B=0.48, S1=0.54, TL=0.64, TLvB="+33.5% (0.025)",
                        TLvS1="+18.6% (0.025)", S1vB="+12.5%")},
    "pIC50": {"Accuracy": dict(B=0.74, S1=0.80, TL=0.80, TLvB="+8.5% (0.017)",
                               TLvS1="-0.4% (0.45, ns)", S1vB="+9.0%"),
              "F1": dict(B=0.72, S1=0.78, TL=0.78, TLvB="+8.0% (0.17, ns)",
                         TLvS1="-0.0% (0.93, ns)", S1vB="+8.1%"),
              "MCC": dict(B=0.44, S1=0.55, TL=0.52, TLvB="+20.0% (0.030)",
                          TLvS1="-4.7% (0.42, ns)", S1vB="+26.0%")},
}
# Published per-threshold B means, from champion_config.json -- the reproduction gate.
B_GATE = {"pIC50": {"MCC": (0.2929, 0.4783, 0.5357), "Accuracy": (0.6934, 0.7713, 0.7504),
                    "F1": (0.7021, 0.7444, 0.7071)},
          "pKi": {"MCC": (0.3881, 0.4987, 0.5408), "Accuracy": (0.8240, 0.7253, 0.8157),
                  "F1": (0.8893, 0.6360, 0.6235)}}


def load_neural(runs: Path) -> dict:
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
            for lab, col in NEURAL_COL.items():
                if r.get(col):
                    acc[(e, t, fold, arm, lab)].append(float(r[col]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def load_b(path: Path) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        for lab, col in B_COL.items():
            out[(r["endpoint"], int(float(r["threshold"])), int(r["fold"]), "B", lab)] = \
                float(r[col])
    return out


def holm(p: dict) -> dict:
    out, run = {}, 0.0
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (len(p) - i) * v))
        out[k] = run
    return out


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=here / "results/table1")
    ap.add_argument("--baseline", type=Path,
                    default=here / "results/baseline_b_per_fold.csv")
    ap.add_argument("--out", type=Path, default=here / "results/table_s40_full.csv")
    args = ap.parse_args()

    v = load_neural(args.runs)
    v.update(load_b(args.baseline))

    def vec(e, arm, lab):
        return np.array([v[(e, t, f, arm, lab)] for t in THRS for f in range(1, 6)])

    # ---- reproduction gate -------------------------------------------------------
    print("REPRODUCTION GATE — re-run B against the published per-threshold means")
    print("(champion_config.json; these are the values behind Table S40's B column)\n")
    worst = 0.0
    for e in ("pIC50", "pKi"):
        for lab in METRICS:
            got = [np.mean([v[(e, t, f, "B", lab)] for f in range(1, 6)]) for t in THRS]
            exp = B_GATE[e][lab]
            d = max(abs(a - b) for a, b in zip(got, exp))
            worst = max(worst, d)
            flag = "OK " if d < 0.02 else "*** CHECK ***"
            print(f"  {flag} {e:6} {lab:9} got "
                  f"{'/'.join(f'{x:.4f}' for x in got)}  expected "
                  f"{'/'.join(f'{x:.4f}' for x in exp)}  max|Δ|={d:.4f}")
    print(f"\n  worst deviation across all 18 cells: {worst:.4f}\n")

    # ---- the table ----------------------------------------------------------------
    raw_p, cells = {}, {}
    for e in ("pIC50", "pKi"):
        for lab in METRICS:
            b, s1, tl = vec(e, "B", lab), vec(e, "S1", lab), vec(e, "TL", lab)
            cells[(e, lab)] = dict(
                B=b.mean(), S1=s1.mean(), TL=tl.mean(),
                TLvB=(tl.mean() - b.mean()) / b.mean() * 100,
                TLvS1=(tl.mean() - s1.mean()) / s1.mean() * 100,
                S1vB=(s1.mean() - b.mean()) / b.mean() * 100,
                p_TLvB=float(stats.wilcoxon(tl, b).pvalue),
                p_TLvS1=float(stats.wilcoxon(tl, s1).pvalue))
            raw_p[(e, lab, "TLvB")] = cells[(e, lab)]["p_TLvB"]
            raw_p[(e, lab, "TLvS1")] = cells[(e, lab)]["p_TLvS1"]

    # Holm over the two planned comparisons, within endpoint and metric, as published.
    adj = {}
    for e in ("pIC50", "pKi"):
        for lab in METRICS:
            a = holm({k: raw_p[(e, lab, k)] for k in ("TLvB", "TLvS1")})
            adj[(e, lab)] = a

    def fmt(pct, p):
        return f"{pct:+.1f}% ({p:.3f}{', ns' if p > 0.05 else ''})"

    print("=" * 108)
    print("TABLE S40 REPLACEMENT — all three arms nested, 15 fold-pairs, "
          "Holm over the two planned TL comparisons")
    print("=" * 108)
    rows = []
    for e in ("pIC50", "pKi"):
        print(f"\n{e}")
        print(f"{'Metric':10}{'B':>7}{'S1':>8}{'TL':>8}   {'TL vs B':>20}"
              f"{'TL vs S1':>20}{'S1 vs B':>10}     published TL vs B")
        for lab in METRICS:
            c = cells[(e, lab)]
            a = adj[(e, lab)]
            print(f"{lab:10}{c['B']:>7.3f}{c['S1']:>8.3f}{c['TL']:>8.3f}   "
                  f"{fmt(c['TLvB'], a['TLvB']):>20}{fmt(c['TLvS1'], a['TLvS1']):>20}"
                  f"{c['S1vB']:>9.1f}%     {PUB[e][lab]['TLvB']}")
            rows.append(dict(endpoint=e, metric=lab,
                             B=round(c["B"], 3), S1=round(c["S1"], 3),
                             TL=round(c["TL"], 3),
                             TL_vs_B_pct=round(c["TLvB"], 1),
                             TL_vs_B_p_holm=round(a["TLvB"], 4),
                             TL_vs_S1_pct=round(c["TLvS1"], 1),
                             TL_vs_S1_p_holm=round(a["TLvS1"], 4),
                             S1_vs_B_pct=round(c["S1vB"], 1),
                             published_B=PUB[e][lab]["B"],
                             published_S1=PUB[e][lab]["S1"],
                             published_TL=PUB[e][lab]["TL"],
                             published_TL_vs_B=PUB[e][lab]["TLvB"],
                             published_TL_vs_S1=PUB[e][lab]["TLvS1"],
                             published_S1_vs_B=PUB[e][lab]["S1vB"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
